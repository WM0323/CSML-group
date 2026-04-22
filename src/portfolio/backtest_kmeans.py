"""Backtest the PCA plus K-means baseline and compare it with the walk-forward HMM."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import calculate_performance_metrics
from src.portfolio.backtest_common import (
    BENCHMARK_ALLOCATIONS,
    REGIME_ALLOCATIONS,
    STATE_LABELS,
    TRADED_ASSETS,
    build_allocation_table,
    build_asset_return_frame,
    build_benchmark_table,
    build_metrics_markdown,
    build_metrics_table,
    ensure_parent_directory,
    format_decimal,
    format_percentage,
    load_market_data,
    require_existing_file,
    run_benchmark_backtests,
    run_strategy_backtest,
    validate_allocation_map,
)


@dataclass(frozen=True)
class KMeansBacktestRunSummary:
    market_data_path: str
    cluster_labels_path: str
    clustering_notes_path: str
    strategy_returns_path: str
    benchmark_returns_path: str
    metrics_path: str
    comparison_metrics_path: str
    notes_path: str
    row_count: int
    start_date: str
    end_date: str
    traded_assets: list[str]
    benchmark_names: list[str]
    regime_state_map: dict[int, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PCA plus K-means regime backtest on the walk-forward HMM comparison window."
        )
    )
    parser.add_argument(
        "--market-data-input",
        default="data/processed/aligned_market_data.parquet",
        help="Path to the approved aligned market data parquet file.",
    )
    parser.add_argument(
        "--cluster-labels-input",
        default="results/regimes/cluster_labels.parquet",
        help="Path to the clustering baseline labels parquet file.",
    )
    parser.add_argument(
        "--clustering-notes-input",
        default="docs/clustering_baseline_notes.md",
        help="Path to the clustering baseline notes.",
    )
    parser.add_argument(
        "--comparison-dates-input",
        default="results/regimes/hmm_walkforward_state_labels.parquet",
        help="Optional walk-forward HMM label path used to align the K-means backtest window.",
    )
    parser.add_argument(
        "--comparison-strategy-input",
        default="results/backtests/walkforward_strategy_returns.parquet",
        help="Optional walk-forward HMM strategy return path for direct strategy comparison.",
    )
    parser.add_argument(
        "--strategy-output",
        default="results/backtests/kmeans_strategy_returns.parquet",
        help="Output parquet path for daily K-means strategy returns and weights.",
    )
    parser.add_argument(
        "--benchmarks-output",
        default="results/backtests/kmeans_benchmark_returns.parquet",
        help="Output parquet path for daily benchmark returns on the K-means sample.",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/backtests/kmeans_strategy_metrics.csv",
        help="Output CSV path for K-means strategy and benchmark metrics.",
    )
    parser.add_argument(
        "--comparison-metrics-output",
        default="results/backtests/kmeans_vs_walkforward_strategy_metrics.csv",
        help="Output CSV path for the direct K-means versus walk-forward HMM comparison.",
    )
    parser.add_argument(
        "--notes-output",
        default="docs/backtest_kmeans_notes.md",
        help="Output Markdown path for concise K-means backtest notes.",
    )
    return parser


def load_cluster_labels(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster labels not found: {path}")

    frame = pd.read_parquet(path)
    required_columns = ["date", "cluster_label"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Cluster labels are missing required columns: " + ", ".join(missing_columns)
        )

    frame = frame[required_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Cluster labels contain duplicate dates after sorting.")
    if frame["cluster_label"].isna().any():
        raise ValueError("Cluster labels contain missing cluster values.")

    frame["cluster_label"] = frame["cluster_label"].astype(int)
    observed_labels = set(frame["cluster_label"].unique().tolist())
    unsupported_labels = sorted(observed_labels - {0, 1, 2})
    if unsupported_labels:
        raise ValueError(
            "Cluster labels contain unsupported ordered values: "
            + ", ".join(str(label) for label in unsupported_labels)
        )
    return frame


def load_optional_comparison_dates(input_path: str | Path) -> pd.DataFrame | None:
    path = Path(input_path)
    if not path.exists():
        return None

    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        return None

    frame = frame[["date"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)
    return frame


def build_kmeans_signal_frame(
    asset_return_frame: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    comparison_dates: pd.DataFrame | None,
) -> pd.DataFrame:
    merged = asset_return_frame.merge(
        cluster_labels,
        on="date",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["signal_state"] = merged["cluster_label"].shift(1)

    return_columns = [f"{asset.lower()}_return" for asset in TRADED_ASSETS]
    merged = merged.dropna(subset=[*return_columns, "signal_state"]).reset_index(drop=True)
    merged["signal_state"] = merged["signal_state"].astype(int)
    merged["cluster_state_name"] = merged["cluster_label"].map(STATE_LABELS)
    merged["signal_state_name"] = merged["signal_state"].map(STATE_LABELS)

    if comparison_dates is not None:
        merged = merged.merge(comparison_dates, on="date", how="inner", validate="one_to_one")

    if merged.empty:
        raise ValueError("K-means backtest alignment produced no usable rows.")
    if merged["signal_state_name"].isna().any():
        raise ValueError("K-means signal frame contains unknown ordered cluster labels.")

    return merged


def load_optional_strategy_returns(input_path: str | Path) -> pd.DataFrame | None:
    path = Path(input_path)
    if not path.exists():
        return None

    frame = pd.read_parquet(path)
    required_columns = ["date", "strategy_return"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        return None

    frame = frame[required_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)
    return frame


def build_model_comparison_metrics(
    kmeans_strategy_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    benchmark_names: list[str],
    comparison_strategy_input: str | Path,
) -> pd.DataFrame | None:
    walkforward_strategy = load_optional_strategy_returns(comparison_strategy_input)
    if walkforward_strategy is None:
        return None

    overlap = kmeans_strategy_frame[["date", "strategy_return"]].merge(
        walkforward_strategy,
        on="date",
        how="inner",
        suffixes=("_kmeans", "_walkforward_hmm"),
        validate="one_to_one",
    )
    if len(overlap) < 2:
        return None

    benchmark_overlap = benchmark_frame.merge(
        overlap[["date"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )

    rows = [
        {
            "portfolio_name": "kmeans_regime_strategy",
            **calculate_performance_metrics(overlap["strategy_return_kmeans"]),
        },
        {
            "portfolio_name": "walkforward_hmm_regime_strategy",
            **calculate_performance_metrics(overlap["strategy_return_walkforward_hmm"]),
        },
    ]
    for benchmark_name in benchmark_names:
        rows.append(
            {
                "portfolio_name": benchmark_name,
                **calculate_performance_metrics(benchmark_overlap[f"{benchmark_name}_return"]),
            }
        )

    return pd.DataFrame(rows)


def build_notes_text(
    summary: KMeansBacktestRunSummary,
    metrics_frame: pd.DataFrame,
    comparison_metrics_frame: pd.DataFrame | None,
) -> str:
    strategy_metrics = metrics_frame.loc[
        metrics_frame["portfolio_name"] == "kmeans_regime_strategy"
    ].iloc[0]
    best_sharpe_row = metrics_frame.sort_values("sharpe_ratio", ascending=False).iloc[0]
    highest_return_row = metrics_frame.sort_values("cumulative_return", ascending=False).iloc[0]

    comparison_section: list[str] = []
    if comparison_metrics_frame is not None:
        comparison_section.extend(
            [
                "## Direct Comparison With The Walk-Forward HMM",
                "",
                build_metrics_markdown(comparison_metrics_frame),
                "",
            ]
        )
        kmeans_comparison = comparison_metrics_frame.loc[
            comparison_metrics_frame["portfolio_name"] == "kmeans_regime_strategy"
        ].iloc[0]
        walkforward_comparison = comparison_metrics_frame.loc[
            comparison_metrics_frame["portfolio_name"] == "walkforward_hmm_regime_strategy"
        ].iloc[0]

        comparison_section.extend(
            [
                (
                    "- On the exact shared window, the K-means strategy cumulative return was "
                    f"{format_percentage(kmeans_comparison['cumulative_return'])} versus "
                    f"{format_percentage(walkforward_comparison['cumulative_return'])} for "
                    "the walk-forward HMM."
                ),
                (
                    "- On the exact shared window, the K-means strategy Sharpe ratio was "
                    f"{format_decimal(kmeans_comparison['sharpe_ratio'])} versus "
                    f"{format_decimal(walkforward_comparison['sharpe_ratio'])} for the "
                    "walk-forward HMM."
                ),
                "",
            ]
        )

    return "\n".join(
        [
            "# K-means Backtest Notes",
            "",
            "- Strategy type: long-only regime allocation driven by ordered PCA plus K-means cluster labels.",
            (
                "- Signal timing: the ordered cluster label from trading day `t-1` is applied to "
                "returns on day `t` to avoid using same-day feature information."
            ),
            (
                "- Methodology note: the cluster labels now come from the walk-forward clustering baseline, "
                "where scaler, PCA, K-means, and label ordering are all fit on the training window only."
            ),
            (
                "- Comparison window: by default the backtest is restricted to the dates where the "
                "walk-forward HMM produces a tradable signal."
            ),
            f"- Approved interpretation source: `{summary.clustering_notes_path}`",
            (
                f"- Backtest sample: `{summary.start_date}` to `{summary.end_date}` "
                f"({summary.row_count} rows)"
            ),
            "",
            "## Regime Mapping",
            "",
            (
                "The ordered cluster labels run from weakest to strongest in-sample `SPY` return, "
                "so we reuse the same defensive-to-aggressive allocation map:"
            ),
            "",
            build_allocation_table(),
            "",
            "## Static Benchmarks",
            "",
            build_benchmark_table(),
            "",
            "## Performance Metrics",
            "",
            build_metrics_markdown(metrics_frame),
            "",
            *comparison_section,
            "## Short Interpretation",
            "",
            (
                "- The K-means strategy cumulative return was "
                f"{format_percentage(strategy_metrics['cumulative_return'])} with annualized "
                f"volatility {format_percentage(strategy_metrics['annualized_volatility'])}."
            ),
            (
                f"- The highest Sharpe ratio in the K-means sample table was "
                f"`{best_sharpe_row['portfolio_name']}` at {format_decimal(best_sharpe_row['sharpe_ratio'])}."
            ),
            (
                f"- The highest cumulative return in the K-means sample table was "
                f"`{highest_return_row['portfolio_name']}` at "
                f"{format_percentage(highest_return_row['cumulative_return'])}."
            ),
            "",
            "## Validation Notes",
            "",
            f"- Input artifact dates aligned successfully with `{summary.cluster_labels_path}`.",
            "- The K-means strategy uses a one-day lagged signal because the walk-forward cluster label for date `t` is assigned from same-day features `x_t`.",
            "- The walk-forward HMM strategy uses a predictive same-day signal because it trades on `P(z_t | x_1:t-1)`.",
            "- Static benchmarks are computed on the same daily return rows as the K-means strategy.",
        ]
    )


def run_kmeans_backtest_pipeline(args: argparse.Namespace) -> KMeansBacktestRunSummary:
    validate_allocation_map(REGIME_ALLOCATIONS, expected_keys=[0, 1, 2], name="regime")
    validate_allocation_map(
        BENCHMARK_ALLOCATIONS,
        expected_keys=list(BENCHMARK_ALLOCATIONS),
        name="benchmark",
    )

    clustering_notes_path = require_existing_file(
        args.clustering_notes_input, "Clustering baseline notes"
    )
    market_data = load_market_data(args.market_data_input)
    cluster_labels = load_cluster_labels(args.cluster_labels_input)
    comparison_dates = load_optional_comparison_dates(args.comparison_dates_input)

    asset_return_frame = build_asset_return_frame(market_data)
    signal_frame = build_kmeans_signal_frame(
        asset_return_frame=asset_return_frame,
        cluster_labels=cluster_labels,
        comparison_dates=comparison_dates,
    )
    strategy_frame = run_strategy_backtest(signal_frame, REGIME_ALLOCATIONS)
    benchmark_frame = run_benchmark_backtests(signal_frame, BENCHMARK_ALLOCATIONS)

    benchmark_names = list(BENCHMARK_ALLOCATIONS.keys())
    metrics_frame = build_metrics_table(
        strategy_frame,
        benchmark_frame,
        benchmark_names,
        strategy_name="kmeans_regime_strategy",
    )
    comparison_metrics_frame = build_model_comparison_metrics(
        kmeans_strategy_frame=strategy_frame,
        benchmark_frame=benchmark_frame,
        benchmark_names=benchmark_names,
        comparison_strategy_input=args.comparison_strategy_input,
    )

    summary = KMeansBacktestRunSummary(
        market_data_path=str(Path(args.market_data_input)),
        cluster_labels_path=str(Path(args.cluster_labels_input)),
        clustering_notes_path=str(clustering_notes_path),
        strategy_returns_path=str(Path(args.strategy_output)),
        benchmark_returns_path=str(Path(args.benchmarks_output)),
        metrics_path=str(Path(args.metrics_output)),
        comparison_metrics_path=str(Path(args.comparison_metrics_output)),
        notes_path=str(Path(args.notes_output)),
        row_count=int(len(strategy_frame)),
        start_date=str(strategy_frame["date"].min().date()),
        end_date=str(strategy_frame["date"].max().date()),
        traded_assets=list(TRADED_ASSETS.keys()),
        benchmark_names=benchmark_names,
        regime_state_map=STATE_LABELS,
    )

    ensure_parent_directory(args.strategy_output)
    ensure_parent_directory(args.benchmarks_output)
    ensure_parent_directory(args.metrics_output)
    ensure_parent_directory(args.comparison_metrics_output)
    ensure_parent_directory(args.notes_output)

    strategy_frame.to_parquet(args.strategy_output, index=False)
    benchmark_frame.to_parquet(args.benchmarks_output, index=False)
    metrics_frame.to_csv(args.metrics_output, index=False)
    if comparison_metrics_frame is not None:
        comparison_metrics_frame.to_csv(args.comparison_metrics_output, index=False)
    Path(args.notes_output).write_text(
        build_notes_text(summary, metrics_frame, comparison_metrics_frame) + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_kmeans_backtest_pipeline(args)
    print("K-means backtest completed.")
    print(f"Strategy returns: {summary.strategy_returns_path}")
    print(f"Benchmark returns: {summary.benchmark_returns_path}")
    print(f"Metrics table: {summary.metrics_path}")
    print(f"Comparison metrics: {summary.comparison_metrics_path}")
    print(f"Notes: {summary.notes_path}")
    print(f"Backtest rows: {summary.row_count}")
    print(f"Sample window: {summary.start_date} to {summary.end_date}")
    print(f"Configuration: {asdict(summary)}")


if __name__ == "__main__":
    main()
