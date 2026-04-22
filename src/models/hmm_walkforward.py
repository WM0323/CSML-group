"""Walk-forward HMM evaluation with past-only fitting and real-time signals."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.hmm_common import (
    APPROVED_FEATURE_COLUMNS,
    DEFAULT_RANDOM_SEEDS,
    FitAttempt,
    fit_hmm_with_restarts,
    format_markdown_table,
    load_feature_table,
    parse_random_seeds,
)


@dataclass(frozen=True)
class RefitBlockSummary:
    refit_date: str
    training_end_date: str
    block_end_date: str
    training_rows: int
    signal_rows: int
    selected_fit: FitAttempt
    ordered_raw_states: list[int]
    state_profile: list[dict[str, object]]


@dataclass(frozen=True)
class WalkforwardHMMRunSummary:
    input_path: str
    labels_path: str
    probabilities_path: str
    metadata_path: str
    notes_path: str
    row_count: int
    start_date: str
    end_date: str
    feature_columns: list[str]
    labels_columns: list[str]
    probabilities_columns: list[str]
    min_training_days: int
    refit_frequency: str
    refit_count: int
    first_refit_date: str
    last_refit_date: str
    state_persistence: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a walk-forward 3-state Gaussian HMM using only past data and "
            "export real-time predictive regime signals."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/model_features.parquet",
        help="Path to the approved modeling feature table parquet file.",
    )
    parser.add_argument(
        "--labels-output",
        default="results/regimes/hmm_walkforward_state_labels.parquet",
        help="Output parquet path for walk-forward regime labels aligned to tradable dates.",
    )
    parser.add_argument(
        "--probabilities-output",
        default="results/regimes/hmm_walkforward_state_probabilities.parquet",
        help="Output parquet path for walk-forward predictive state probabilities.",
    )
    parser.add_argument(
        "--metadata-output",
        default="results/models/hmm_walkforward_metadata.json",
        help="Output JSON path for walk-forward reproducibility metadata.",
    )
    parser.add_argument(
        "--notes-output",
        default="docs/hmm_walkforward_notes.md",
        help="Output Markdown path for concise walk-forward notes.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of HMM states. This walk-forward pipeline keeps 3 states.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=200,
        help="Maximum EM iterations per restart.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="EM convergence tolerance.",
    )
    parser.add_argument(
        "--covariance-type",
        default="diag",
        choices=["diag", "full", "spherical", "tied"],
        help="Gaussian covariance structure for the HMM.",
    )
    parser.add_argument(
        "--random-seeds",
        default="7,11,19,23,31",
        help="Comma-separated random seeds to try at each refit.",
    )
    parser.add_argument(
        "--min-training-days",
        type=int,
        default=756,
        help="Minimum expanding-window training sample before the first tradable signal.",
    )
    parser.add_argument(
        "--refit-frequency",
        default="monthly",
        choices=["monthly", "quarterly"],
        help="Calendar cadence for re-estimating HMM parameters.",
    )
    return parser


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
        raise ValueError("Failed to build a non-empty refit schedule.")
    return schedule


def validate_scaler(scales: np.ndarray) -> None:
    zero_scale_columns = [
        column
        for column, scale in zip(APPROVED_FEATURE_COLUMNS, scales, strict=True)
        if np.isclose(scale, 0.0)
    ]
    if zero_scale_columns:
        raise ValueError(
            "Cannot standardize constant feature columns in the training window: "
            + ", ".join(zero_scale_columns)
        )


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    weight_total = float(weights.sum())
    if np.isclose(weight_total, 0.0):
        return float("nan")
    return float(np.dot(values, weights) / weight_total)


def build_posterior_weighted_state_order(
    training_frame: pd.DataFrame,
    training_posteriors: np.ndarray,
) -> tuple[list[int], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    spy_values = training_frame["spy_ret_1d"].to_numpy(dtype=float)
    qqq_values = training_frame["qqq_ret_1d"].to_numpy(dtype=float)
    tlt_values = training_frame["tlt_ret_1d"].to_numpy(dtype=float)
    vix_values = training_frame["vix_close"].to_numpy(dtype=float)
    slope_values = training_frame["yield_curve_slope_pct"].to_numpy(dtype=float)

    for raw_state in range(training_posteriors.shape[1]):
        weights = training_posteriors[:, raw_state].astype(float)
        effective_obs = float(weights.sum())
        rows.append(
            {
                "raw_state": int(raw_state),
                "effective_observations": effective_obs,
                "sample_share": effective_obs / float(len(training_frame)),
                "mean_spy_ret_1d": weighted_average(spy_values, weights),
                "mean_qqq_ret_1d": weighted_average(qqq_values, weights),
                "mean_tlt_ret_1d": weighted_average(tlt_values, weights),
                "mean_vix_close": weighted_average(vix_values, weights),
                "mean_yield_curve_slope_pct": weighted_average(slope_values, weights),
            }
        )

    profile = (
        pd.DataFrame(rows)
        .sort_values(
            by=["mean_spy_ret_1d", "mean_vix_close", "raw_state"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )
    ordered_raw_states = profile["raw_state"].astype(int).tolist()
    raw_to_ordered = {raw_state: ordered for ordered, raw_state in enumerate(ordered_raw_states)}
    profile["ordered_state"] = [raw_to_ordered[int(value)] for value in profile["raw_state"]]
    profile = profile[
        [
            "ordered_state",
            "raw_state",
            "effective_observations",
            "sample_share",
            "mean_spy_ret_1d",
            "mean_qqq_ret_1d",
            "mean_tlt_ret_1d",
            "mean_vix_close",
            "mean_yield_curve_slope_pct",
        ]
    ]
    return ordered_raw_states, profile


def normalize_log_probabilities(log_values: np.ndarray) -> np.ndarray:
    max_log_value = float(np.max(log_values))
    shifted = np.exp(log_values - max_log_value)
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Forward recursion produced invalid state weights.")
    return shifted / total


def build_filtered_posterior_block(
    emission_log_likelihoods: np.ndarray,
    initial_state_distribution: np.ndarray,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictive_probabilities = np.zeros_like(emission_log_likelihoods, dtype=float)
    filtered_posteriors = np.zeros_like(emission_log_likelihoods, dtype=float)

    predictive = np.asarray(initial_state_distribution, dtype=float)
    predictive = np.clip(predictive, 1e-300, None)
    predictive = predictive / predictive.sum()

    for row_index, emission_log in enumerate(emission_log_likelihoods):
        predictive_probabilities[row_index] = predictive
        filtered_posteriors[row_index] = normalize_log_probabilities(
            np.log(predictive) + emission_log
        )
        predictive = np.asarray(filtered_posteriors[row_index] @ transition_matrix, dtype=float)
        predictive = np.clip(predictive, 1e-300, None)
        predictive = predictive / predictive.sum()

    return predictive_probabilities, filtered_posteriors


def build_predictive_signal_block(
    emission_log_likelihoods: np.ndarray,
    last_training_filtered: np.ndarray,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictive_probabilities = np.zeros_like(emission_log_likelihoods, dtype=float)
    filtered_posteriors = np.zeros_like(emission_log_likelihoods, dtype=float)

    previous_filtered = np.asarray(last_training_filtered, dtype=float)
    for row_index, emission_log in enumerate(emission_log_likelihoods):
        predictive = np.asarray(previous_filtered @ transition_matrix, dtype=float)
        predictive = np.clip(predictive, 1e-300, None)
        predictive = predictive / predictive.sum()
        predictive_probabilities[row_index] = predictive

        filtered_posteriors[row_index] = normalize_log_probabilities(
            np.log(predictive) + emission_log
        )
        previous_filtered = filtered_posteriors[row_index]

    return predictive_probabilities, filtered_posteriors


def save_dataframe(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def build_output_state_profile(
    feature_table: pd.DataFrame,
    labels_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = feature_table.merge(labels_frame, on="date", how="inner", validate="one_to_one")
    profile = (
        merged.groupby("hmm_state", as_index=False)
        .agg(
            observations=("hmm_state", "size"),
            mean_spy_ret_1d=("spy_ret_1d", "mean"),
            mean_qqq_ret_1d=("qqq_ret_1d", "mean"),
            mean_tlt_ret_1d=("tlt_ret_1d", "mean"),
            mean_vix_close=("vix_close", "mean"),
            mean_yield_curve_slope_pct=("yield_curve_slope_pct", "mean"),
        )
        .sort_values("hmm_state")
        .reset_index(drop=True)
    )
    profile["sample_share"] = profile["observations"] / float(len(labels_frame))
    profile["state_name"] = profile["hmm_state"].map(
        {0: "risk_off", 1: "transition", 2: "risk_on"}
    )
    return profile[
        [
            "hmm_state",
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


def calculate_state_persistence(labels_frame: pd.DataFrame) -> float:
    if len(labels_frame) < 2:
        return float("nan")
    repeated = (
        labels_frame["hmm_state"].astype(int).iloc[1:].to_numpy()
        == labels_frame["hmm_state"].astype(int).iloc[:-1].to_numpy()
    )
    return float(repeated.mean())


def build_metadata(
    input_path: str | Path,
    labels_path: str | Path,
    probabilities_path: str | Path,
    notes_path: str | Path,
    feature_table: pd.DataFrame,
    labels_frame: pd.DataFrame,
    min_training_days: int,
    refit_frequency: str,
    covariance_type: str,
    n_iter: int,
    tol: float,
    random_seeds: list[int],
    refit_blocks: list[RefitBlockSummary],
) -> dict[str, object]:
    return {
        "input_path": str(Path(input_path)),
        "labels_path": str(Path(labels_path)),
        "probabilities_path": str(Path(probabilities_path)),
        "notes_path": str(Path(notes_path)),
        "row_count": int(len(labels_frame)),
        "start_date": labels_frame["date"].min().date().isoformat(),
        "end_date": labels_frame["date"].max().date().isoformat(),
        "approved_feature_columns": APPROVED_FEATURE_COLUMNS,
        "walkforward_spec": {
            "training_window": "expanding",
            "min_training_days": min_training_days,
            "refit_frequency": refit_frequency,
            "signal_type": "one_step_ahead_predictive_probabilities",
            "signal_definition": (
                "For tradable date t, the exported probability vector is "
                "P(z_t | x_1:t-1). Within each refit block the forward recursion "
                "updates P(z_t | x_1:t) after observing x_t, then advances one "
                "step through the transition matrix for the next tradable date."
            ),
            "state_ordering_rule": (
                "At each refit, raw states are re-labeled from lowest to highest "
                "forward-filtered training mean spy_ret_1d."
            ),
            "standardization_rule": (
                "Each refit block uses a StandardScaler fitted only on the "
                "expanding training window available before that block starts."
            ),
        },
        "model_spec": {
            "n_components": 3,
            "covariance_type": covariance_type,
            "n_iter": n_iter,
            "tol": tol,
            "min_covar": 1e-4,
            "random_seeds": random_seeds,
        },
        "refit_count": len(refit_blocks),
        "refit_blocks": [asdict(block) for block in refit_blocks],
        "walkforward_state_persistence": calculate_state_persistence(labels_frame),
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
    }


def save_metadata(metadata: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def write_walkforward_notes(
    output_path: str | Path,
    feature_table: pd.DataFrame,
    labels_frame: pd.DataFrame,
    refit_blocks: list[RefitBlockSummary],
    min_training_days: int,
    refit_frequency: str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_profile = build_output_state_profile(feature_table, labels_frame)
    profile_markdown = format_markdown_table(state_profile)
    average_training_rows = sum(block.training_rows for block in refit_blocks) / len(refit_blocks)
    state_persistence = calculate_state_persistence(labels_frame)

    content = "\n".join(
        [
            "# Walk-Forward HMM Notes",
            "",
            "- Model: 3-state Gaussian HMM with diagonal covariance.",
            "- Inputs: the approved 18-feature table in `data/processed/model_features.parquet`.",
            f"- Training window: expanding, with a minimum warm-up of {min_training_days} trading days.",
            f"- Refit cadence: {refit_frequency}.",
            (
                "- Real-time tradable signal: for each exported date `t`, the probability file stores "
                "`P(z_t | x_1:t-1)`, the one-step-ahead predictive state distribution formed by the "
                "forward recursion using only information available before date `t`."
            ),
            (
                "- Forward update rule: after the model observes `x_t`, it updates the filtered "
                "posterior `P(z_t | x_1:t)` and advances one transition step to form the predictive "
                "distribution for the next tradable date."
            ),
            (
                "- Important methodology point: the tradable signal does not rely on a Viterbi path or "
                "`model.predict()` decoding."
            ),
            (
                "- State ordering: at each refit, raw states are re-labeled from weakest to strongest "
                "forward-filtered training mean `spy_ret_1d`."
            ),
            (
                f"- Signal sample: {labels_frame['date'].min().date().isoformat()} to "
                f"{labels_frame['date'].max().date().isoformat()} ({len(labels_frame)} rows)."
            ),
            f"- Refit count: {len(refit_blocks)}.",
            f"- Average training rows per refit: {average_training_rows:.1f}.",
            f"- Walk-forward state persistence: {state_persistence:.4f}.",
            "",
            "## Output State Profile",
            "",
            (
                "The table below summarizes same-day realized feature outcomes conditional on the "
                "exported predictive signal. Because these are out-of-sample realized values rather "
                "than the training-time ordering metric, they do not need to stay perfectly ordered "
                "from State 0 to State 2."
            ),
            "",
            profile_markdown,
            "",
            "## Validation Notes",
            "",
            (
                f"- The first tradable date appears only after the {min_training_days}-day warm-up, "
                "so no future data is used to form the exported signal rows."
            ),
            "- The walk-forward initialization uses the final forward-filtered training posterior, not a Viterbi path.",
            "- Each refit block uses scaler parameters estimated only on the past training window.",
            "- Within each block, the forward recursion updates daily using only observations available up to that date.",
        ]
    )

    path.write_text(content + "\n", encoding="utf-8")
    return path


def validate_outputs(
    labels_path: str | Path,
    probabilities_path: str | Path,
    expected_row_count: int,
    expected_start_date: str,
    expected_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_frame = pd.read_parquet(labels_path)
    probabilities_frame = pd.read_parquet(probabilities_path)

    if len(labels_frame) != expected_row_count or len(probabilities_frame) != expected_row_count:
        raise ValueError("Validated walk-forward outputs do not match the expected row count.")
    if labels_frame["date"].duplicated().any() or probabilities_frame["date"].duplicated().any():
        raise ValueError("Validated walk-forward outputs contain duplicate dates.")
    if labels_frame["date"].min().date().isoformat() != expected_start_date:
        raise ValueError("Walk-forward labels start date does not match expectation.")
    if labels_frame["date"].max().date().isoformat() != expected_end_date:
        raise ValueError("Walk-forward labels end date does not match expectation.")
    if probabilities_frame["date"].min().date().isoformat() != expected_start_date:
        raise ValueError("Walk-forward probabilities start date does not match expectation.")
    if probabilities_frame["date"].max().date().isoformat() != expected_end_date:
        raise ValueError("Walk-forward probabilities end date does not match expectation.")

    probability_values = probabilities_frame.drop(columns=["date"])
    if not np.allclose(probability_values.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-6):
        raise ValueError("Walk-forward probability rows do not sum to 1 within tolerance.")

    return labels_frame, probabilities_frame


def run_walkforward_hmm_pipeline(
    input_path: str | Path = "data/processed/model_features.parquet",
    labels_output_path: str | Path = "results/regimes/hmm_walkforward_state_labels.parquet",
    probabilities_output_path: str | Path = "results/regimes/hmm_walkforward_state_probabilities.parquet",
    metadata_output_path: str | Path = "results/models/hmm_walkforward_metadata.json",
    notes_output_path: str | Path = "docs/hmm_walkforward_notes.md",
    n_components: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-4,
    random_seeds: list[int] | None = None,
    min_training_days: int = 756,
    refit_frequency: str = "monthly",
) -> WalkforwardHMMRunSummary:
    if n_components != 3:
        raise ValueError("The walk-forward pipeline currently supports exactly 3 HMM states.")

    seeds = random_seeds or DEFAULT_RANDOM_SEEDS
    feature_table = load_feature_table(input_path)
    refit_schedule = build_refit_schedule(feature_table, min_training_days, refit_frequency)

    label_frames: list[pd.DataFrame] = []
    probability_frames: list[pd.DataFrame] = []
    refit_blocks: list[RefitBlockSummary] = []

    for block_index, start_index in enumerate(refit_schedule):
        end_index = (
            refit_schedule[block_index + 1] - 1
            if block_index + 1 < len(refit_schedule)
            else len(feature_table) - 1
        )
        training_frame = feature_table.iloc[:start_index].copy()
        signal_frame = feature_table.iloc[start_index : end_index + 1].copy()

        scaler = StandardScaler()
        standardized_training = scaler.fit_transform(training_frame[APPROVED_FEATURE_COLUMNS])
        validate_scaler(np.asarray(scaler.scale_, dtype=float))
        standardized_signal = scaler.transform(signal_frame[APPROVED_FEATURE_COLUMNS])

        model, selected_fit, _ = fit_hmm_with_restarts(
            standardized_values=standardized_training,
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_seeds=seeds,
        )

        training_emission_log_likelihoods = np.asarray(
            model._compute_log_likelihood(standardized_training),
            dtype=float,
        )
        _, training_filtered_posteriors = build_filtered_posterior_block(
            emission_log_likelihoods=training_emission_log_likelihoods,
            initial_state_distribution=np.asarray(model.startprob_, dtype=float),
            transition_matrix=np.asarray(model.transmat_, dtype=float),
        )
        ordered_raw_states, training_profile = build_posterior_weighted_state_order(
            training_frame=training_frame,
            training_posteriors=training_filtered_posteriors,
        )

        emission_log_likelihoods = np.asarray(
            model._compute_log_likelihood(standardized_signal),
            dtype=float,
        )
        predictive_probabilities_raw, _ = build_predictive_signal_block(
            emission_log_likelihoods=emission_log_likelihoods,
            last_training_filtered=np.asarray(training_filtered_posteriors[-1], dtype=float),
            transition_matrix=np.asarray(model.transmat_, dtype=float),
        )

        predictive_probabilities_ordered = predictive_probabilities_raw[:, ordered_raw_states]
        predictive_probabilities_ordered = (
            predictive_probabilities_ordered
            / predictive_probabilities_ordered.sum(axis=1, keepdims=True)
        )
        ordered_states = np.argmax(predictive_probabilities_ordered, axis=1).astype(int)

        label_frames.append(
            pd.DataFrame(
                {
                    "date": signal_frame["date"].to_numpy(),
                    "hmm_state": ordered_states,
                }
            )
        )
        probability_frame = pd.DataFrame(
            predictive_probabilities_ordered,
            columns=[f"state_{state}_probability" for state in range(n_components)],
        )
        probability_frame.insert(0, "date", signal_frame["date"].to_numpy())
        probability_frames.append(probability_frame)

        refit_blocks.append(
            RefitBlockSummary(
                refit_date=signal_frame["date"].iloc[0].date().isoformat(),
                training_end_date=training_frame["date"].iloc[-1].date().isoformat(),
                block_end_date=signal_frame["date"].iloc[-1].date().isoformat(),
                training_rows=int(len(training_frame)),
                signal_rows=int(len(signal_frame)),
                selected_fit=selected_fit,
                ordered_raw_states=ordered_raw_states,
                state_profile=json.loads(training_profile.to_json(orient="records")),
            )
        )

    labels_frame = pd.concat(label_frames, ignore_index=True)
    probabilities_frame = pd.concat(probability_frames, ignore_index=True)

    labels_path = save_dataframe(labels_frame, labels_output_path)
    probabilities_path = save_dataframe(probabilities_frame, probabilities_output_path)

    metadata = build_metadata(
        input_path=input_path,
        labels_path=labels_path,
        probabilities_path=probabilities_path,
        notes_path=notes_output_path,
        feature_table=feature_table,
        labels_frame=labels_frame,
        min_training_days=min_training_days,
        refit_frequency=refit_frequency,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_seeds=seeds,
        refit_blocks=refit_blocks,
    )
    metadata_path = save_metadata(metadata, metadata_output_path)
    notes_path = write_walkforward_notes(
        output_path=notes_output_path,
        feature_table=feature_table,
        labels_frame=labels_frame,
        refit_blocks=refit_blocks,
        min_training_days=min_training_days,
        refit_frequency=refit_frequency,
    )

    validated_labels, validated_probabilities = validate_outputs(
        labels_path=labels_path,
        probabilities_path=probabilities_path,
        expected_row_count=len(feature_table) - min_training_days,
        expected_start_date=feature_table["date"].iloc[min_training_days].date().isoformat(),
        expected_end_date=feature_table["date"].iloc[-1].date().isoformat(),
    )

    return WalkforwardHMMRunSummary(
        input_path=str(Path(input_path)),
        labels_path=str(labels_path),
        probabilities_path=str(probabilities_path),
        metadata_path=str(metadata_path),
        notes_path=str(notes_path),
        row_count=int(len(validated_labels)),
        start_date=validated_labels["date"].min().date().isoformat(),
        end_date=validated_labels["date"].max().date().isoformat(),
        feature_columns=APPROVED_FEATURE_COLUMNS.copy(),
        labels_columns=list(validated_labels.columns),
        probabilities_columns=list(validated_probabilities.columns),
        min_training_days=min_training_days,
        refit_frequency=refit_frequency,
        refit_count=len(refit_blocks),
        first_refit_date=refit_blocks[0].refit_date,
        last_refit_date=refit_blocks[-1].refit_date,
        state_persistence=calculate_state_persistence(validated_labels),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_walkforward_hmm_pipeline(
        input_path=args.input,
        labels_output_path=args.labels_output,
        probabilities_output_path=args.probabilities_output,
        metadata_output_path=args.metadata_output,
        notes_output_path=args.notes_output,
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        n_iter=args.n_iter,
        tol=args.tol,
        random_seeds=parse_random_seeds(args.random_seeds),
        min_training_days=args.min_training_days,
        refit_frequency=args.refit_frequency,
    )

    print(f"Input: {summary.input_path}")
    print(f"Rows: {summary.row_count}")
    print(f"Signal window: {summary.start_date} to {summary.end_date}")
    print(f"Refits: {summary.refit_count} ({summary.first_refit_date} to {summary.last_refit_date})")
    print(f"Labels output: {summary.labels_path}")
    print(f"Probabilities output: {summary.probabilities_path}")
    print(f"Metadata output: {summary.metadata_path}")
    print(f"Notes output: {summary.notes_path}")
    print(f"State persistence: {summary.state_persistence:.4f}")


if __name__ == "__main__":
    main()
