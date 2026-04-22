"""Phase C Step 10 walk-forward PCA plus K-means clustering baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


APPROVED_FEATURE_COLUMNS = [
    "spy_ret_1d",
    "spy_vol_20d_ann",
    "qqq_ret_1d",
    "qqq_vol_20d_ann",
    "tlt_ret_1d",
    "tlt_vol_20d_ann",
    "ief_ret_1d",
    "ief_vol_20d_ann",
    "gld_ret_1d",
    "gld_vol_20d_ann",
    "uso_ret_1d",
    "uso_vol_20d_ann",
    "vix_close",
    "vix_change_5d",
    "yield_curve_slope_pct",
    "spy_minus_tlt_ret_5d",
    "qqq_minus_spy_ret_5d",
    "gld_minus_uso_ret_5d",
]

DEFAULT_PCA_COMPONENTS = 5
DEFAULT_RANDOM_STATE = 17
DEFAULT_N_INIT = 50
DEFAULT_MIN_TRAINING_DAYS = 756
DEFAULT_REFIT_FREQUENCY = "monthly"


@dataclass(frozen=True)
class RefitBlockSummary:
    refit_date: str
    training_end_date: str
    block_end_date: str
    training_rows: int
    signal_rows: int
    pca_explained_variance_ratio: list[float]
    ordered_raw_clusters: list[int]
    cluster_profile: list[dict[str, object]]


@dataclass(frozen=True)
class ClusteringRunSummary:
    input_path: str
    labels_path: str
    metadata_path: str
    notes_path: str
    row_count: int
    start_date: str
    end_date: str
    feature_columns: list[str]
    label_columns: list[str]
    pca_components: int
    explained_variance_ratio: list[float]
    cluster_counts: dict[int, int]
    same_as_previous_rate: float
    hmm_comparison_available: bool
    min_training_days: int
    refit_frequency: str
    refit_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the Phase C Step 10 walk-forward PCA plus K-means baseline."
    )
    parser.add_argument(
        "--input",
        default="data/processed/model_features.parquet",
        help="Path to the approved modeling feature table parquet file.",
    )
    parser.add_argument(
        "--labels-output",
        default="results/regimes/cluster_labels.parquet",
        help="Output parquet path for walk-forward clustering labels.",
    )
    parser.add_argument(
        "--metadata-output",
        default="results/models/clustering_baseline_metadata.json",
        help="Output JSON path for reproducibility metadata.",
    )
    parser.add_argument(
        "--notes-output",
        default="docs/clustering_baseline_notes.md",
        help="Output Markdown path for concise clustering notes.",
    )
    parser.add_argument(
        "--hmm-labels-input",
        default="results/regimes/hmm_walkforward_state_labels.parquet",
        help="Optional walk-forward HMM labels parquet path for a high-level comparison section.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters. Phase C Step 10 requires 3.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help="Number of PCA components used before K-means.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for deterministic K-means initialization.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=DEFAULT_N_INIT,
        help="Number of K-means restarts to run with the fixed random seed.",
    )
    parser.add_argument(
        "--min-training-days",
        type=int,
        default=DEFAULT_MIN_TRAINING_DAYS,
        help="Minimum expanding-window training sample before the first walk-forward cluster block.",
    )
    parser.add_argument(
        "--refit-frequency",
        default=DEFAULT_REFIT_FREQUENCY,
        choices=["monthly", "quarterly"],
        help="Calendar cadence for re-estimating scaler, PCA, and K-means.",
    )
    return parser


def load_feature_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")

    frame = pd.read_parquet(path)
    required_columns = ["date", *APPROVED_FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Feature table is missing required approved columns: "
            + ", ".join(missing_columns)
        )

    frame = frame[required_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Feature table contains invalid dates.")

    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)

    if frame["date"].duplicated().any():
        raise ValueError("Feature table contains duplicate dates after sorting.")

    if frame[APPROVED_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature table contains missing values in approved inputs.")

    if len(frame) < 30:
        raise ValueError("Feature table is too short for a stable 3-cluster baseline.")

    return frame


def period_label(timestamp: pd.Timestamp, refit_frequency: str) -> pd.Period:
    code = "M" if refit_frequency == "monthly" else "Q"
    return timestamp.to_period(code)


def build_refit_schedule(
    feature_table: pd.DataFrame,
    min_training_days: int,
    refit_frequency: str,
) -> list[int]:
    if min_training_days < 252:
        raise ValueError("Minimum training sample should be at least 252 trading days.")
    if min_training_days >= len(feature_table):
        raise ValueError("Minimum training sample leaves no holdout dates for walk-forward labels.")

    schedule: list[int] = []
    for index in range(min_training_days, len(feature_table)):
        if index == min_training_days:
            schedule.append(index)
            continue

        current_period = period_label(feature_table.loc[index, "date"], refit_frequency)
        previous_period = period_label(feature_table.loc[index - 1, "date"], refit_frequency)
        if current_period != previous_period:
            schedule.append(index)

    if not schedule:
        raise ValueError("Failed to build a non-empty walk-forward clustering schedule.")
    return schedule


def standardize_features(feature_table: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_table[APPROVED_FEATURE_COLUMNS])

    zero_scale_columns = [
        column
        for column, scale in zip(APPROVED_FEATURE_COLUMNS, scaler.scale_, strict=True)
        if np.isclose(scale, 0.0)
    ]
    if zero_scale_columns:
        raise ValueError(
            "Cannot standardize constant feature columns: " + ", ".join(zero_scale_columns)
        )

    return scaled_values, scaler


def build_embedding(
    standardized_values: np.ndarray,
    requested_components: int,
) -> tuple[np.ndarray, PCA]:
    if requested_components < 2:
        raise ValueError("PCA components must be at least 2 for the Step 10 baseline.")

    effective_components = min(
        requested_components,
        standardized_values.shape[0],
        standardized_values.shape[1],
    )
    if effective_components < 2:
        raise ValueError("Input table does not have enough information for PCA.")

    pca = PCA(n_components=effective_components, svd_solver="full")
    embedding = pca.fit_transform(standardized_values)
    return embedding, pca


def fit_kmeans(
    embedding: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple[np.ndarray, KMeans]:
    if n_clusters != 3:
        raise ValueError("Phase C Step 10 requires exactly 3 clusters.")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        algorithm="lloyd",
    )
    raw_assignments = model.fit_predict(embedding)
    observed_clusters = np.unique(raw_assignments)
    if observed_clusters.size != n_clusters:
        raise ValueError(
            "K-means did not produce exactly 3 observed clusters: "
            + ", ".join(str(int(cluster)) for cluster in observed_clusters)
        )

    return raw_assignments.astype(int), model


def build_cluster_order(
    feature_table: pd.DataFrame,
    raw_assignments: np.ndarray,
) -> tuple[list[int], dict[int, int], pd.DataFrame]:
    profile = (
        feature_table.assign(raw_cluster=raw_assignments)
        .groupby("raw_cluster", as_index=False)
        .agg(
            observations=("raw_cluster", "size"),
            mean_spy_ret_1d=("spy_ret_1d", "mean"),
            mean_qqq_ret_1d=("qqq_ret_1d", "mean"),
            mean_tlt_ret_1d=("tlt_ret_1d", "mean"),
            mean_vix_close=("vix_close", "mean"),
            mean_yield_curve_slope_pct=("yield_curve_slope_pct", "mean"),
        )
        .sort_values(
            by=["mean_spy_ret_1d", "mean_vix_close", "raw_cluster"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )

    ordered_raw_clusters = profile["raw_cluster"].astype(int).tolist()
    raw_to_ordered = {
        raw_cluster: ordered_cluster
        for ordered_cluster, raw_cluster in enumerate(ordered_raw_clusters)
    }
    profile["ordered_cluster"] = [
        raw_to_ordered[int(raw_cluster)] for raw_cluster in profile["raw_cluster"]
    ]
    profile["sample_share"] = profile["observations"] / float(len(feature_table))
    profile = profile[
        [
            "ordered_cluster",
            "raw_cluster",
            "observations",
            "sample_share",
            "mean_spy_ret_1d",
            "mean_qqq_ret_1d",
            "mean_tlt_ret_1d",
            "mean_vix_close",
            "mean_yield_curve_slope_pct",
        ]
    ]
    return ordered_raw_clusters, raw_to_ordered, profile


def remap_cluster_assignments(
    raw_assignments: np.ndarray,
    raw_to_ordered: dict[int, int],
) -> np.ndarray:
    return np.array([raw_to_ordered[int(cluster)] for cluster in raw_assignments], dtype=int)


def build_labels_output(
    signal_frame: pd.DataFrame,
    ordered_assignments: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": signal_frame["date"].to_numpy(),
            "cluster_label": ordered_assignments,
        }
    )


def save_dataframe(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def save_metadata(metadata: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def format_markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        formatted_row: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted_row.append(f"{value:.6f}")
            else:
                formatted_row.append(str(value))
        lines.append("| " + " | ".join(formatted_row) + " |")
    return "\n".join(lines)


def compute_run_statistics(assignments: np.ndarray) -> dict[str, float | int]:
    if assignments.size == 0:
        raise ValueError("Cannot compute run statistics for an empty assignment array.")

    same_as_previous_rate = (
        float(np.mean(assignments[1:] == assignments[:-1])) if assignments.size > 1 else 1.0
    )

    run_lengths: list[int] = []
    current_value = int(assignments[0])
    current_length = 1
    for assignment in assignments[1:]:
        assignment_int = int(assignment)
        if assignment_int == current_value:
            current_length += 1
        else:
            run_lengths.append(current_length)
            current_value = assignment_int
            current_length = 1
    run_lengths.append(current_length)

    run_lengths_array = np.array(run_lengths, dtype=float)
    return {
        "same_as_previous_rate": same_as_previous_rate,
        "average_run_length": float(run_lengths_array.mean()),
        "median_run_length": float(np.median(run_lengths_array)),
        "max_run_length": int(run_lengths_array.max()),
        "run_count": int(run_lengths_array.size),
    }


def load_hmm_comparison(
    hmm_labels_path: str | Path,
    labels_frame: pd.DataFrame,
) -> dict[str, object] | None:
    path = Path(hmm_labels_path)
    if not path.exists():
        return None

    hmm_frame = pd.read_parquet(path)
    required_columns = ["date", "hmm_state"]
    missing_columns = [column for column in required_columns if column not in hmm_frame.columns]
    if missing_columns:
        raise ValueError(
            "HMM labels parquet is missing required columns: " + ", ".join(missing_columns)
        )

    hmm_frame = hmm_frame[required_columns].copy()
    hmm_frame["date"] = pd.to_datetime(hmm_frame["date"]).dt.normalize()
    merged = labels_frame.merge(hmm_frame, on="date", how="inner")
    if len(merged) < 2:
        raise ValueError("HMM labels comparison did not produce a usable overlap window.")

    hmm_statistics = compute_run_statistics(merged["hmm_state"].to_numpy(dtype=int))
    crosstab = pd.crosstab(
        merged["cluster_label"],
        merged["hmm_state"],
        normalize="index",
    ).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0.0)

    dominant_overlap: dict[str, int] = {}
    for cluster_label, row in crosstab.iterrows():
        dominant_overlap[f"cluster_{int(cluster_label)}"] = int(row.astype(float).idxmax())

    return {
        "hmm_labels_path": str(path),
        "hmm_row_count": int(len(hmm_frame)),
        "overlap_row_count": int(len(merged)),
        "overlap_start_date": merged["date"].min().date().isoformat(),
        "overlap_end_date": merged["date"].max().date().isoformat(),
        "hmm_persistence": hmm_statistics,
        "cluster_to_hmm_overlap": json.loads(crosstab.to_json()),
        "dominant_hmm_state_by_cluster": dominant_overlap,
    }


def build_output_state_profile(
    feature_table: pd.DataFrame,
    labels_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = feature_table.merge(labels_frame, on="date", how="inner", validate="one_to_one")
    profile = (
        merged.groupby("cluster_label", as_index=False)
        .agg(
            observations=("cluster_label", "size"),
            mean_spy_ret_1d=("spy_ret_1d", "mean"),
            mean_qqq_ret_1d=("qqq_ret_1d", "mean"),
            mean_tlt_ret_1d=("tlt_ret_1d", "mean"),
            mean_vix_close=("vix_close", "mean"),
            mean_yield_curve_slope_pct=("yield_curve_slope_pct", "mean"),
        )
        .sort_values("cluster_label")
        .reset_index(drop=True)
    )
    profile["sample_share"] = profile["observations"] / float(len(labels_frame))
    profile["state_name"] = profile["cluster_label"].map(
        {0: "risk_off", 1: "transition", 2: "risk_on"}
    )
    return profile[
        [
            "cluster_label",
            "state_name",
            "observations",
            "sample_share",
            "mean_spy_ret_1d",
            "mean_qqq_ret_1d",
            "mean_tlt_ret_1d",
            "mean_vix_close",
            "mean_yield_curve_slope_pct",
        ]
    ]


def describe_cluster(row: pd.Series) -> str:
    mean_spy_ret = float(row["mean_spy_ret_1d"])
    mean_vix = float(row["mean_vix_close"])
    mean_tlt_ret = float(row["mean_tlt_ret_1d"])

    if mean_spy_ret < 0.0 and mean_vix >= 20.0:
        tone = "stress or drawdown days"
    elif mean_spy_ret > 0.0 and mean_vix >= 20.0:
        tone = "rebound days with elevated volatility"
    else:
        tone = "calmer trend days"

    bond_note = (
        "Treasuries tended to help" if mean_tlt_ret > 0.0 else "Treasuries did not cushion much"
    )
    return f"Cluster {int(row['cluster_label'])}: {tone}; {bond_note}."


def average_explained_variance_ratio(refit_blocks: list[RefitBlockSummary]) -> list[float]:
    ratio_matrix = np.array(
        [block.pca_explained_variance_ratio for block in refit_blocks],
        dtype=float,
    )
    return ratio_matrix.mean(axis=0).tolist()


def write_notes(
    output_path: str | Path,
    labels_frame: pd.DataFrame,
    state_profile: pd.DataFrame,
    cluster_statistics: dict[str, float | int],
    hmm_comparison: dict[str, object] | None,
    refit_blocks: list[RefitBlockSummary],
    min_training_days: int,
    refit_frequency: str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile_markdown = format_markdown_table(state_profile)
    cluster_descriptions = "\n".join(
        f"- {describe_cluster(row)}"
        for _, row in state_profile.sort_values("cluster_label").iterrows()
    )
    average_training_rows = sum(block.training_rows for block in refit_blocks) / len(refit_blocks)
    average_explained = average_explained_variance_ratio(refit_blocks)
    explained_text = ", ".join(f"{value:.4f}" for value in average_explained)

    hmm_section = ""
    if hmm_comparison is None:
        hmm_section = (
            "## High-Level Comparison With HMM\n\n"
            "- No walk-forward HMM labels file was available, so the Step 10 notes stop at the clustering profile.\n"
        )
    else:
        hmm_stats = hmm_comparison["hmm_persistence"]
        overlap_frame = pd.DataFrame(hmm_comparison["cluster_to_hmm_overlap"])
        overlap_frame.index.name = "cluster_label"
        overlap_frame.columns = [f"hmm_state_{column}" for column in overlap_frame.columns]
        overlap_markdown = format_markdown_table(overlap_frame.reset_index())
        dominant_overlap = hmm_comparison["dominant_hmm_state_by_cluster"]

        hmm_section = (
            "## High-Level Comparison With HMM\n\n"
            f"- Overlap window: {hmm_comparison['overlap_start_date']} to {hmm_comparison['overlap_end_date']} "
            f"({hmm_comparison['overlap_row_count']} rows).\n"
            f"- Walk-forward K-means labels stayed the same as the previous day {cluster_statistics['same_as_previous_rate']:.4f} of the time.\n"
            f"- Walk-forward HMM labels stayed the same as the previous day {hmm_stats['same_as_previous_rate']:.4f} of the time.\n"
            "- Interpretation: the walk-forward PCA plus K-means baseline is less temporally stable because it has no transition model, "
            "so it reacts to each day in isolation even after all preprocessing is fit on past data only.\n"
            f"- Dominant walk-forward HMM overlap by ordered cluster: {dominant_overlap}.\n\n"
            "### Cluster-to-HMM Overlap\n\n"
            f"{overlap_markdown}\n"
        )

    content = (
        "# Clustering Baseline Notes\n\n"
        "- Model: walk-forward z-score the approved features, reduce them with PCA, then fit 3-cluster K-means.\n"
        "- Inputs: the approved 18-feature table in `data/processed/model_features.parquet`.\n"
        f"- Training window: expanding, with a minimum warm-up of {min_training_days} trading days.\n"
        f"- Refit cadence: {refit_frequency}.\n"
        "- Cluster ordering: at each refit, raw clusters are remapped from weakest to strongest training-window mean `spy_ret_1d`.\n"
        "- Real-time timing note: the exported walk-forward cluster label for date `t` is produced by a past-only fit applied to same-day features `x_t`, so the backtest must lag the signal by one day before trading.\n"
        f"- Signal sample: {labels_frame['date'].min().date().isoformat()} to {labels_frame['date'].max().date().isoformat()} "
        f"({len(labels_frame)} rows).\n"
        f"- Refit count: {len(refit_blocks)}.\n"
        f"- Average training rows per refit: {average_training_rows:.1f}.\n"
        f"- Average PCA explained variance ratio: {explained_text} "
        f"(cumulative {float(sum(average_explained)):.4f}).\n"
        f"- Cluster same-as-previous-day rate: {cluster_statistics['same_as_previous_rate']:.4f}.\n"
        f"- Average cluster run length: {cluster_statistics['average_run_length']:.2f} days.\n\n"
        "## Output State Profile\n\n"
        "The table below summarizes same-day realized feature outcomes conditional on the exported walk-forward cluster label.\n\n"
        f"{profile_markdown}\n\n"
        "## Plain-Language Interpretation\n\n"
        f"{cluster_descriptions}\n\n"
        f"{hmm_section}"
        "## Validation Notes\n\n"
        f"- The first walk-forward cluster block appears only after the {min_training_days}-day warm-up.\n"
        "- Each refit block fits the scaler, PCA embedding, and K-means model using the training window only.\n"
        "- Cluster labels in each signal block are produced by applying `transform` and `predict` to the holdout block only.\n"
        "- Any trading strategy using these labels must lag them by one day because the label for date `t` uses same-day features.\n"
    )

    path.write_text(content, encoding="utf-8")
    return path


def build_metadata(
    input_path: str | Path,
    labels_path: str | Path,
    metadata_path: str | Path,
    notes_path: str | Path,
    hmm_labels_path: str | Path,
    labels_frame: pd.DataFrame,
    refit_blocks: list[RefitBlockSummary],
    cluster_statistics: dict[str, float | int],
    hmm_comparison: dict[str, object] | None,
    pca_components: int,
    random_state: int,
    n_init: int,
    min_training_days: int,
    refit_frequency: str,
) -> dict[str, object]:
    cluster_counts = {
        int(cluster): int(count)
        for cluster, count in labels_frame["cluster_label"].value_counts().sort_index().items()
    }
    return {
        "input_path": str(Path(input_path)),
        "labels_path": str(Path(labels_path)),
        "metadata_path": str(Path(metadata_path)),
        "notes_path": str(Path(notes_path)),
        "optional_hmm_labels_path": str(Path(hmm_labels_path)),
        "row_count": int(len(labels_frame)),
        "start_date": labels_frame["date"].min().date().isoformat(),
        "end_date": labels_frame["date"].max().date().isoformat(),
        "approved_feature_columns": APPROVED_FEATURE_COLUMNS,
        "walkforward_spec": {
            "training_window": "expanding",
            "min_training_days": min_training_days,
            "refit_frequency": refit_frequency,
            "standardization_rule": "fit StandardScaler on the training window only at each refit",
            "pca_rule": "fit PCA on the training-window standardized features only at each refit",
            "cluster_fit_rule": "fit KMeans on the training-window PCA embedding only at each refit",
            "ordering_rule": "re-label raw clusters from lowest to highest training-window mean spy_ret_1d",
            "signal_definition": "For date t, export the ordered cluster label assigned by the past-only scaler/PCA/KMeans fit to same-day features x_t. Any trading rule must lag this label to date t+1.",
        },
        "pca_spec": {
            "requested_components": int(pca_components),
            "average_explained_variance_ratio": average_explained_variance_ratio(refit_blocks),
        },
        "clustering_spec": {
            "model": "KMeans",
            "n_clusters": 3,
            "n_init": int(n_init),
            "random_state": int(random_state),
            "algorithm": "lloyd",
        },
        "refit_count": len(refit_blocks),
        "refit_blocks": [asdict(block) for block in refit_blocks],
        "cluster_counts": cluster_counts,
        "cluster_persistence": cluster_statistics,
        "optional_hmm_comparison": hmm_comparison,
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
    }


def validate_outputs(
    labels_path: str | Path,
    expected_row_count: int,
    expected_start_date: str,
    expected_end_date: str,
) -> pd.DataFrame:
    labels_frame = pd.read_parquet(labels_path)
    required_columns = ["date", "cluster_label"]
    missing_columns = [column for column in required_columns if column not in labels_frame.columns]
    if missing_columns:
        raise ValueError(
            "Validated labels output is missing required columns: "
            + ", ".join(missing_columns)
        )

    if len(labels_frame) != expected_row_count:
        raise ValueError("Validated labels output row count does not match the walk-forward expectation.")

    if labels_frame["date"].min().date().isoformat() != expected_start_date:
        raise ValueError("Labels output start date does not match the expected signal window.")
    if labels_frame["date"].max().date().isoformat() != expected_end_date:
        raise ValueError("Labels output end date does not match the expected signal window.")

    if labels_frame["date"].duplicated().any():
        raise ValueError("Validated labels output contains duplicate dates.")

    if labels_frame["cluster_label"].isna().any():
        raise ValueError("Validated labels output contains missing cluster assignments.")

    unsupported_clusters = sorted(
        set(labels_frame["cluster_label"].astype(int).unique().tolist()) - {0, 1, 2}
    )
    if unsupported_clusters:
        raise ValueError(
            "Validated labels output contains unsupported ordered clusters: "
            + ", ".join(str(cluster) for cluster in unsupported_clusters)
        )

    return labels_frame


def run_clustering_pipeline(
    input_path: str | Path = "data/processed/model_features.parquet",
    labels_output_path: str | Path = "results/regimes/cluster_labels.parquet",
    metadata_output_path: str | Path = "results/models/clustering_baseline_metadata.json",
    notes_output_path: str | Path = "docs/clustering_baseline_notes.md",
    hmm_labels_path: str | Path = "results/regimes/hmm_walkforward_state_labels.parquet",
    n_clusters: int = 3,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
    min_training_days: int = DEFAULT_MIN_TRAINING_DAYS,
    refit_frequency: str = DEFAULT_REFIT_FREQUENCY,
) -> ClusteringRunSummary:
    if n_clusters != 3:
        raise ValueError("Phase C Step 10 requires exactly 3 clusters.")

    feature_table = load_feature_table(input_path)
    refit_schedule = build_refit_schedule(feature_table, min_training_days, refit_frequency)

    label_frames: list[pd.DataFrame] = []
    refit_blocks: list[RefitBlockSummary] = []

    for block_index, start_index in enumerate(refit_schedule):
        end_index = (
            refit_schedule[block_index + 1] - 1
            if block_index + 1 < len(refit_schedule)
            else len(feature_table) - 1
        )
        training_frame = feature_table.iloc[:start_index].copy()
        signal_frame = feature_table.iloc[start_index : end_index + 1].copy()

        standardized_training, scaler = standardize_features(training_frame)
        standardized_signal = scaler.transform(signal_frame[APPROVED_FEATURE_COLUMNS])
        training_embedding, pca = build_embedding(
            standardized_training,
            requested_components=pca_components,
        )
        signal_embedding = pca.transform(standardized_signal)

        raw_training_assignments, cluster_model = fit_kmeans(
            embedding=training_embedding,
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        )
        ordered_raw_clusters, raw_to_ordered, cluster_profile = build_cluster_order(
            feature_table=training_frame,
            raw_assignments=raw_training_assignments,
        )

        raw_signal_assignments = cluster_model.predict(signal_embedding).astype(int)
        ordered_signal_assignments = remap_cluster_assignments(
            raw_assignments=raw_signal_assignments,
            raw_to_ordered=raw_to_ordered,
        )
        label_frames.append(
            build_labels_output(signal_frame=signal_frame, ordered_assignments=ordered_signal_assignments)
        )

        refit_blocks.append(
            RefitBlockSummary(
                refit_date=signal_frame["date"].iloc[0].date().isoformat(),
                training_end_date=training_frame["date"].iloc[-1].date().isoformat(),
                block_end_date=signal_frame["date"].iloc[-1].date().isoformat(),
                training_rows=int(len(training_frame)),
                signal_rows=int(len(signal_frame)),
                pca_explained_variance_ratio=[
                    float(value) for value in pca.explained_variance_ratio_
                ],
                ordered_raw_clusters=ordered_raw_clusters,
                cluster_profile=json.loads(cluster_profile.to_json(orient="records")),
            )
        )

    labels_frame = pd.concat(label_frames, ignore_index=True)
    labels_path = save_dataframe(labels_frame, labels_output_path)

    cluster_statistics = compute_run_statistics(labels_frame["cluster_label"].to_numpy(dtype=int))
    hmm_comparison = load_hmm_comparison(hmm_labels_path, labels_frame)
    state_profile = build_output_state_profile(feature_table, labels_frame)

    metadata = build_metadata(
        input_path=input_path,
        labels_path=labels_path,
        metadata_path=metadata_output_path,
        notes_path=notes_output_path,
        hmm_labels_path=hmm_labels_path,
        labels_frame=labels_frame,
        refit_blocks=refit_blocks,
        cluster_statistics=cluster_statistics,
        hmm_comparison=hmm_comparison,
        pca_components=pca_components,
        random_state=random_state,
        n_init=n_init,
        min_training_days=min_training_days,
        refit_frequency=refit_frequency,
    )
    metadata_path = save_metadata(metadata, metadata_output_path)
    notes_path = write_notes(
        output_path=notes_output_path,
        labels_frame=labels_frame,
        state_profile=state_profile,
        cluster_statistics=cluster_statistics,
        hmm_comparison=hmm_comparison,
        refit_blocks=refit_blocks,
        min_training_days=min_training_days,
        refit_frequency=refit_frequency,
    )

    validated_labels = validate_outputs(
        labels_path=labels_path,
        expected_row_count=len(feature_table) - min_training_days,
        expected_start_date=feature_table["date"].iloc[min_training_days].date().isoformat(),
        expected_end_date=feature_table["date"].iloc[-1].date().isoformat(),
    )

    cluster_counts = {
        int(cluster): int(count)
        for cluster, count in validated_labels["cluster_label"].value_counts().sort_index().items()
    }
    average_explained = average_explained_variance_ratio(refit_blocks)

    return ClusteringRunSummary(
        input_path=str(Path(input_path)),
        labels_path=str(labels_path),
        metadata_path=str(metadata_path),
        notes_path=str(notes_path),
        row_count=int(len(validated_labels)),
        start_date=validated_labels["date"].min().date().isoformat(),
        end_date=validated_labels["date"].max().date().isoformat(),
        feature_columns=APPROVED_FEATURE_COLUMNS.copy(),
        label_columns=list(validated_labels.columns),
        pca_components=len(average_explained),
        explained_variance_ratio=average_explained,
        cluster_counts=cluster_counts,
        same_as_previous_rate=float(cluster_statistics["same_as_previous_rate"]),
        hmm_comparison_available=hmm_comparison is not None,
        min_training_days=min_training_days,
        refit_frequency=refit_frequency,
        refit_count=len(refit_blocks),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_clustering_pipeline(
        input_path=args.input,
        labels_output_path=args.labels_output,
        metadata_output_path=args.metadata_output,
        notes_output_path=args.notes_output,
        hmm_labels_path=args.hmm_labels_input,
        n_clusters=args.n_clusters,
        pca_components=args.pca_components,
        random_state=args.random_state,
        n_init=args.n_init,
        min_training_days=args.min_training_days,
        refit_frequency=args.refit_frequency,
    )

    explained = ", ".join(f"{value:.4f}" for value in summary.explained_variance_ratio)
    print(f"Input: {summary.input_path}")
    print(f"Rows: {summary.row_count}")
    print(f"Date range: {summary.start_date} to {summary.end_date}")
    print(f"Features ({len(summary.feature_columns)}): {', '.join(summary.feature_columns)}")
    print(f"Labels output: {summary.labels_path} -> {', '.join(summary.label_columns)}")
    print(f"Metadata output: {summary.metadata_path}")
    print(f"Notes output: {summary.notes_path}")
    print(
        f"PCA components: {summary.pca_components}; average explained variance ratio: {explained}"
    )
    print(
        "Cluster counts: "
        + ", ".join(
            f"cluster_{cluster}={count}" for cluster, count in summary.cluster_counts.items()
        )
    )
    print(
        f"Walk-forward setup: warm-up={summary.min_training_days}, "
        f"refit_frequency={summary.refit_frequency}, refits={summary.refit_count}"
    )
    print(f"Cluster same-as-previous-day rate: {summary.same_as_previous_rate:.4f}")
    print(f"HMM comparison available: {summary.hmm_comparison_available}")


if __name__ == "__main__":
    main()
