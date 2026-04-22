"""Backtest the walk-forward HMM regime signal without same-day leakage."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.portfolio.backtest_common import (
    BENCHMARK_ALLOCATIONS,
    REGIME_ALLOCATIONS,
    STATE_LABELS,
    TRADED_ASSETS,
    build_allocation_table,
    build_asset_return_frame,
    build_benchmark_table,
    build_markdown_table,
    build_metrics_markdown,
    build_metrics_table,
    ensure_parent_directory,
    format_decimal,
    format_percentage,
    load_market_data,
    run_benchmark_backtests,
    run_strategy_backtest,
    require_existing_file,
    validate_allocation_map,
)


@dataclass(frozen=True)
class WalkforwardBacktestRunSummary:
    market_data_path: str
    hmm_labels_path: str
    hmm_notes_path: str
    strategy_returns_path: str
    benchmark_returns_path: str
    metrics_path: str
    notes_path: str
    row_count: int
    start_date: str
    end_date: str
    traded_assets: list[str]
    benchmark_names: list[str]
    regime_state_map: dict[int, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the walk-forward HMM regime backtest using real-time predictive signals."
    )
    parser.add_argument(
        "--market-data-input",
        default="data/processed/aligned_market_data.parquet",
        help="Path to the approved aligned market data parquet file.",
    )
    parser.add_argument(
        "--hmm-labels-input",
        default="results/regimes/hmm_walkforward_state_labels.parquet",
        help="Path to the walk-forward HMM state labels parquet file.",
    )
    parser.add_argument(
        "--hmm-notes-input",
        default="docs/hmm_walkforward_notes.md",
        help="Path to the walk-forward HMM notes.",
    )
    parser.add_argument(
        "--strategy-output",
        default="results/backtests/walkforward_strategy_returns.parquet",
        help="Output parquet path for daily walk-forward strategy returns and weights.",
    )
    parser.add_argument(
        "--benchmarks-output",
        default="results/backtests/walkforward_benchmark_returns.parquet",
        help="Output parquet path for daily benchmark returns on the walk-forward sample.",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/backtests/walkforward_strategy_metrics.csv",
        help="Output CSV path for walk-forward strategy and benchmark metrics.",
    )
    parser.add_argument(
        "--notes-output",
        default="docs/backtest_walkforward_notes.md",
        help="Output Markdown path for concise walk-forward backtest notes.",
    )
    return parser


def load_walkforward_labels(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Walk-forward HMM labels not found: {path}")

    frame = pd.read_parquet(path)
    required_columns = ["date", "hmm_state"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Walk-forward HMM labels are missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = frame[required_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Walk-forward HMM labels contain duplicate dates after sorting.")
    if frame["hmm_state"].isna().any():
        raise ValueError("Walk-forward HMM labels contain missing states.")

    frame["hmm_state"] = frame["hmm_state"].astype(int)
    unsupported_states = sorted(set(frame["hmm_state"].tolist()) - set(STATE_LABELS))
    if unsupported_states:
        raise ValueError(
            "Walk-forward labels contain unsupported states: "
            + ", ".join(str(state) for state in unsupported_states)
        )
    return frame


def build_real_time_signal_frame(
    asset_return_frame: pd.DataFrame,
    walkforward_labels: pd.DataFrame,
) -> pd.DataFrame:
    merged = asset_return_frame.merge(
        walkforward_labels,
        on="date",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values("date").reset_index(drop=True)

    return_columns = [f"{asset.lower()}_return" for asset in TRADED_ASSETS]
    merged = merged.dropna(subset=return_columns).reset_index(drop=True)
    merged["signal_state"] = merged["hmm_state"].astype(int)
    merged["hmm_state_name"] = merged["hmm_state"].map(STATE_LABELS)
    merged["signal_state_name"] = merged["signal_state"].map(STATE_LABELS)

    if merged.empty:
        raise ValueError("Walk-forward backtest alignment produced no usable rows.")
    if merged["signal_state_name"].isna().any():
        raise ValueError("Walk-forward signal frame contains unknown state labels.")

    return merged


def build_notes_text(
    summary: WalkforwardBacktestRunSummary,
    metrics_frame: pd.DataFrame,
) -> str:
    strategy_metrics = metrics_frame.loc[
        metrics_frame["portfolio_name"] == "walkforward_hmm_regime_strategy"
    ].iloc[0]
    best_sharpe_row = metrics_frame.sort_values("sharpe_ratio", ascending=False).iloc[0]
    highest_return_row = metrics_frame.sort_values("cumulative_return", ascending=False).iloc[0]

    return "\n".join(
        [
            "# Walk-Forward Backtest Notes",
            "",
            "- Strategy type: long-only regime allocation driven by the walk-forward HMM signal.",
            (
                "- Signal timing: the exported walk-forward HMM label on date `t` is already "
                "a real-time predictive signal formed before the return on date `t` is realized, "
                "so no additional one-day lag is applied in this backtest."
            ),
            (
                "- Real-time methodology: the tradable signal comes from forward-recursion-based "
                "one-step-ahead predictive probabilities, not from a Viterbi path."
            ),
            "- Transaction costs: omitted in this first walk-forward pass.",
            f"- Approved interpretation source: `{summary.hmm_notes_path}`",
            (
                f"- Backtest sample: `{summary.start_date}` to `{summary.end_date}` "
                f"({summary.row_count} rows)"
            ),
            "",
            "## Regime Mapping",
            "",
            "The walk-forward states remain ordered from weakest to strongest risk tone, so the portfolio map is:",
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
            "## Short Interpretation",
            "",
            (
                "- The walk-forward HMM strategy cumulative return was "
                f"{format_percentage(strategy_metrics['cumulative_return'])} with annualized "
                f"volatility {format_percentage(strategy_metrics['annualized_volatility'])}."
            ),
            (
                f"- The highest Sharpe ratio on the walk-forward sample was "
                f"`{best_sharpe_row['portfolio_name']}` at {format_decimal(best_sharpe_row['sharpe_ratio'])}."
            ),
            (
                f"- The highest cumulative return on the walk-forward sample was "
                f"`{highest_return_row['portfolio_name']}` at "
                f"{format_percentage(highest_return_row['cumulative_return'])}."
            ),
            (
                "- The walk-forward strategy exports one daily row with the predictive HMM state, "
                "the realized daily portfolio return, and the per-asset weights in "
                f"`{summary.strategy_returns_path}`."
            ),
            "",
            "## Validation Notes",
            "",
            f"- Input artifact dates aligned successfully with `{summary.hmm_labels_path}`.",
            "- The signal is non-empty and was applied directly to the same tradable date because the labels are already predictive.",
            "- Benchmarks use the same daily return window as the walk-forward regime strategy.",
        ]
    )


def run_walkforward_backtest_pipeline(args: argparse.Namespace) -> WalkforwardBacktestRunSummary:
    validate_allocation_map(REGIME_ALLOCATIONS, expected_keys=[0, 1, 2], name="regime")
    validate_allocation_map(
        BENCHMARK_ALLOCATIONS,
        expected_keys=list(BENCHMARK_ALLOCATIONS),
        name="benchmark",
    )

    hmm_notes_path = require_existing_file(args.hmm_notes_input, "Walk-forward HMM notes")
    market_data = load_market_data(args.market_data_input)
    walkforward_labels = load_walkforward_labels(args.hmm_labels_input)

    asset_return_frame = build_asset_return_frame(market_data)
    signal_frame = build_real_time_signal_frame(asset_return_frame, walkforward_labels)
    strategy_frame = run_strategy_backtest(signal_frame, REGIME_ALLOCATIONS)
    benchmark_frame = run_benchmark_backtests(signal_frame, BENCHMARK_ALLOCATIONS)

    benchmark_names = list(BENCHMARK_ALLOCATIONS.keys())
    metrics_frame = build_metrics_table(
        strategy_frame,
        benchmark_frame,
        benchmark_names,
        strategy_name="walkforward_hmm_regime_strategy",
    )

    summary = WalkforwardBacktestRunSummary(
        market_data_path=str(Path(args.market_data_input)),
        hmm_labels_path=str(Path(args.hmm_labels_input)),
        hmm_notes_path=str(hmm_notes_path),
        strategy_returns_path=str(Path(args.strategy_output)),
        benchmark_returns_path=str(Path(args.benchmarks_output)),
        metrics_path=str(Path(args.metrics_output)),
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
    ensure_parent_directory(args.notes_output)

    strategy_frame.to_parquet(args.strategy_output, index=False)
    benchmark_frame.to_parquet(args.benchmarks_output, index=False)
    metrics_frame.to_csv(args.metrics_output, index=False)
    Path(args.notes_output).write_text(
        build_notes_text(summary, metrics_frame) + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_walkforward_backtest_pipeline(args)
    print("Walk-forward backtest completed.")
    print(f"Strategy returns: {summary.strategy_returns_path}")
    print(f"Benchmark returns: {summary.benchmark_returns_path}")
    print(f"Metrics table: {summary.metrics_path}")
    print(f"Notes: {summary.notes_path}")
    print(f"Backtest rows: {summary.row_count}")
    print(f"Sample window: {summary.start_date} to {summary.end_date}")
    print(f"Configuration: {asdict(summary)}")


if __name__ == "__main__":
    main()
