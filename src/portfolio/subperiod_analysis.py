"""Shared-window subperiod comparison for the HMM and K-means strategies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import calculate_performance_metrics

PERIOD_SPECS = [
    ("2013-2016", "2013-02-05", "2016-12-31"),
    ("2017-2019", "2017-01-01", "2019-12-31"),
    ("2020-2022", "2020-01-01", "2022-12-31"),
    ("2023-2026", "2023-01-01", None),
]


@dataclass(frozen=True)
class SubperiodRunSummary:
    output_path: str
    notes_path: str
    row_count: int
    period_count: int
    start_date: str
    end_date: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a coarse shared-window subperiod comparison for HMM and K-means."
    )
    parser.add_argument(
        "--walkforward-input",
        default="results/backtests/walkforward_strategy_returns.parquet",
        help="Path to the walk-forward HMM strategy returns parquet file.",
    )
    parser.add_argument(
        "--kmeans-input",
        default="results/backtests/kmeans_strategy_returns.parquet",
        help="Path to the K-means strategy returns parquet file.",
    )
    parser.add_argument(
        "--benchmarks-input",
        default="results/backtests/kmeans_benchmark_returns.parquet",
        help="Path to the benchmark returns parquet file on the exact shared window.",
    )
    parser.add_argument(
        "--output",
        default="results/backtests/shared_window_subperiod_metrics.csv",
        help="Output CSV path for subperiod metrics.",
    )
    parser.add_argument(
        "--notes-output",
        default="docs/subperiod_analysis.md",
        help="Output Markdown path for concise subperiod notes.",
    )
    return parser


def require_existing_file(path_like: str | Path, description: str) -> Path:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_returns_frame(
    input_path: str | Path,
    required_columns: list[str],
    description: str,
) -> pd.DataFrame:
    path = require_existing_file(input_path, description)
    frame = pd.read_parquet(path)

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"{description} is missing required columns: " + ", ".join(missing_columns)
        )

    frame = frame[required_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError(f"{description} contains duplicate dates after sorting.")

    return frame


def build_shared_window_frame(
    walkforward_input: str | Path,
    kmeans_input: str | Path,
    benchmarks_input: str | Path,
) -> pd.DataFrame:
    walkforward = load_returns_frame(
        walkforward_input,
        required_columns=["date", "strategy_return"],
        description="Walk-forward HMM strategy returns",
    ).rename(columns={"strategy_return": "walkforward_hmm_return"})
    kmeans = load_returns_frame(
        kmeans_input,
        required_columns=["date", "strategy_return"],
        description="K-means strategy returns",
    ).rename(columns={"strategy_return": "kmeans_return"})
    benchmarks = load_returns_frame(
        benchmarks_input,
        required_columns=[
            "date",
            "equal_weight_4_asset_return",
            "fixed_60_40_stock_bond_return",
        ],
        description="Shared-window benchmark returns",
    )

    merged = (
        kmeans.merge(walkforward, on="date", how="inner", validate="one_to_one")
        .merge(benchmarks, on="date", how="inner", validate="one_to_one")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(merged) < 2:
        raise ValueError("Shared-window merge did not produce enough observations.")

    return merged


def build_metrics_rows(shared_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    return_columns = {
        "walkforward_hmm_regime_strategy": "walkforward_hmm_return",
        "kmeans_regime_strategy": "kmeans_return",
        "equal_weight_4_asset": "equal_weight_4_asset_return",
        "fixed_60_40_stock_bond": "fixed_60_40_stock_bond_return",
    }

    for period_name, start_date, end_date in PERIOD_SPECS:
        mask = shared_frame["date"] >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= shared_frame["date"] <= pd.Timestamp(end_date)
        period_frame = shared_frame.loc[mask].copy()
        if len(period_frame) < 2:
            continue

        for portfolio_name, return_column in return_columns.items():
            metrics = calculate_performance_metrics(period_frame[return_column])
            rows.append(
                {
                    "period_name": period_name,
                    "start_date": period_frame["date"].min().date().isoformat(),
                    "end_date": period_frame["date"].max().date().isoformat(),
                    "observations": int(len(period_frame)),
                    "portfolio_name": portfolio_name,
                    **metrics,
                }
            )

    if not rows:
        raise ValueError("No usable subperiod blocks were generated.")

    return pd.DataFrame(rows)


def build_summary_lines(metrics_frame: pd.DataFrame) -> list[str]:
    hmm = metrics_frame.loc[
        metrics_frame["portfolio_name"] == "walkforward_hmm_regime_strategy"
    ].set_index("period_name")
    kmeans = metrics_frame.loc[
        metrics_frame["portfolio_name"] == "kmeans_regime_strategy"
    ].set_index("period_name")
    comparison = hmm.join(
        kmeans[
            [
                "cumulative_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
            ]
        ],
        how="inner",
        lsuffix="_hmm",
        rsuffix="_kmeans",
    )

    hmm_better_return = int(
        (comparison["cumulative_return_hmm"] > comparison["cumulative_return_kmeans"]).sum()
    )
    hmm_better_sharpe = int(
        (comparison["sharpe_ratio_hmm"] > comparison["sharpe_ratio_kmeans"]).sum()
    )

    best_period_row = comparison.assign(
        sharpe_gap=lambda frame: frame["sharpe_ratio_hmm"] - frame["sharpe_ratio_kmeans"]
    )["sharpe_gap"].sort_values(ascending=False)
    worst_period_row = comparison.assign(
        sharpe_gap=lambda frame: frame["sharpe_ratio_hmm"] - frame["sharpe_ratio_kmeans"]
    )["sharpe_gap"].sort_values(ascending=True)

    best_period = best_period_row.index[0]
    worst_period = worst_period_row.index[0]

    return [
        (
            f"- Across {len(comparison)} coarse contiguous subperiods, the walk-forward HMM "
            f"beats K-means in cumulative return in {hmm_better_return} period(s) and in Sharpe "
            f"in {hmm_better_sharpe} period(s)."
        ),
        (
            f"- The strongest HMM-vs-K-means Sharpe gap appears in `{best_period}`, while the "
            f"weakest relative block appears in `{worst_period}`."
        ),
        "- These blocks were fixed before inspecting the metrics and are meant as a coarse stability check, not as an optimized market-timing study.",
    ]


def format_percentage(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def format_decimal(value: float) -> str:
    return f"{float(value):.3f}"


def build_markdown_table(display_frame: pd.DataFrame) -> str:
    headers = list(display_frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_notes(metrics_frame: pd.DataFrame) -> str:
    display_frame = metrics_frame[
        [
            "period_name",
            "start_date",
            "end_date",
            "observations",
            "portfolio_name",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]
    ].copy()
    for column in [
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
    ]:
        display_frame[column] = display_frame[column].map(format_percentage)
    display_frame["sharpe_ratio"] = display_frame["sharpe_ratio"].map(format_decimal)

    return "\n".join(
        [
            "# Shared-Window Subperiod Analysis",
            "",
            "- This note splits the exact shared HMM-versus-K-means trading window into four coarse contiguous calendar blocks.",
            "- The goal is to check whether the HMM edge over K-means is broad or concentrated in one episode.",
            *build_summary_lines(metrics_frame),
            "",
            "## Subperiod Metrics",
            "",
            build_markdown_table(display_frame),
        ]
    )


def ensure_parent_directory(path_like: str | Path) -> None:
    Path(path_like).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def run_subperiod_analysis(args: argparse.Namespace) -> SubperiodRunSummary:
    shared_frame = build_shared_window_frame(
        walkforward_input=args.walkforward_input,
        kmeans_input=args.kmeans_input,
        benchmarks_input=args.benchmarks_input,
    )
    metrics_frame = build_metrics_rows(shared_frame)

    ensure_parent_directory(args.output)
    metrics_frame.to_csv(args.output, index=False)
    ensure_parent_directory(args.notes_output)
    Path(args.notes_output).write_text(build_notes(metrics_frame) + "\n", encoding="utf-8")

    return SubperiodRunSummary(
        output_path=str(Path(args.output)),
        notes_path=str(Path(args.notes_output)),
        row_count=int(len(metrics_frame)),
        period_count=int(metrics_frame["period_name"].nunique()),
        start_date=shared_frame["date"].min().date().isoformat(),
        end_date=shared_frame["date"].max().date().isoformat(),
    )


def main() -> None:
    args = build_parser().parse_args()
    summary = run_subperiod_analysis(args)
    print(summary.output_path)
    print(summary.notes_path)
    print(summary)


if __name__ == "__main__":
    main()
