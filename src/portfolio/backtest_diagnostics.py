"""Turnover and transaction-cost diagnostics for strategy backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import calculate_performance_metrics


DEFAULT_COST_BPS = [0, 10, 25]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build turnover and simple transaction-cost diagnostics for strategy backtests."
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
        "--turnover-output",
        default="results/backtests/strategy_turnover_summary.csv",
        help="Output CSV path for turnover diagnostics.",
    )
    parser.add_argument(
        "--cost-output",
        default="results/backtests/strategy_cost_sensitivity.csv",
        help="Output CSV path for transaction-cost sensitivity diagnostics.",
    )
    parser.add_argument(
        "--notes-output",
        default="docs/backtest_diagnostics.md",
        help="Output Markdown path for concise diagnostics notes.",
    )
    return parser


def load_strategy_frame(input_path: str | Path, strategy_name: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"{strategy_name} strategy returns not found: {path}")

    frame = pd.read_parquet(path)
    required_columns = ["date", "strategy_return"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"{strategy_name} strategy frame is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError(f"{strategy_name} strategy frame contains duplicate dates.")

    weight_columns = [column for column in frame.columns if column.startswith("weight_")]
    if not weight_columns:
        raise ValueError(f"{strategy_name} strategy frame is missing allocation weight columns.")
    if frame[weight_columns].isna().any().any():
        raise ValueError(f"{strategy_name} strategy frame contains missing allocation weights.")

    return frame


def compute_turnover(frame: pd.DataFrame) -> pd.Series:
    weight_columns = [column for column in frame.columns if column.startswith("weight_")]
    turnover = frame[weight_columns].diff().abs().sum(axis=1).fillna(0.0) / 2.0
    return turnover.astype(float)


def count_state_switches(frame: pd.DataFrame) -> int:
    if "signal_state" not in frame.columns:
        return 0
    switches = frame["signal_state"].astype(float).diff().fillna(0.0) != 0.0
    return int(switches.sum())


def build_turnover_summary(strategy_name: str, frame: pd.DataFrame) -> dict[str, object]:
    turnover = compute_turnover(frame)
    return {
        "strategy_name": strategy_name,
        "observations": int(len(frame)),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
        "avg_daily_turnover": float(turnover.mean()),
        "median_daily_turnover": float(turnover.median()),
        "max_daily_turnover": float(turnover.max()),
        "days_with_turnover": int((turnover > 0.0).sum()),
        "state_switches": count_state_switches(frame),
    }


def build_cost_sensitivity_rows(
    strategy_name: str,
    frame: pd.DataFrame,
    cost_bps_values: list[int],
) -> list[dict[str, object]]:
    turnover = compute_turnover(frame)
    rows: list[dict[str, object]] = []
    for cost_bps in cost_bps_values:
        cost_rate = float(cost_bps) / 10_000.0
        net_returns = frame["strategy_return"].astype(float) - turnover * cost_rate
        metrics = calculate_performance_metrics(net_returns)
        rows.append(
            {
                "strategy_name": strategy_name,
                "cost_bps": int(cost_bps),
                **metrics,
            }
        )
    return rows


def format_percentage(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def format_decimal(value: float) -> str:
    return f"{float(value):.3f}"


def build_markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        formatted_row: list[str] = []
        for value in row:
            if isinstance(value, float):
                formatted_row.append(f"{value:.6f}")
            else:
                formatted_row.append(str(value))
        lines.append("| " + " | ".join(formatted_row) + " |")
    return "\n".join(lines)


def build_notes(
    turnover_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
) -> str:
    turnover_display = turnover_summary.copy()
    for column in ["avg_daily_turnover", "median_daily_turnover", "max_daily_turnover"]:
        turnover_display[column] = turnover_display[column].map(format_percentage)

    cost_display = cost_sensitivity[
        [
            "strategy_name",
            "cost_bps",
            "observations",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]
    ].copy()
    for column in ["cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown"]:
        cost_display[column] = cost_display[column].map(format_percentage)
    cost_display["sharpe_ratio"] = cost_display["sharpe_ratio"].map(format_decimal)

    return "\n".join(
        [
            "# Backtest Diagnostics",
            "",
            "- Turnover is computed as one-half the sum of absolute daily weight changes.",
            "- Transaction-cost sensitivity applies a simple linear cost model: `net_return_t = gross_return_t - turnover_t * cost_rate`.",
            "- Cost rates shown below are one-way trading costs in basis points per unit of portfolio turnover.",
            "",
            "## Turnover Summary",
            "",
            build_markdown_table(turnover_display),
            "",
            "## Transaction-Cost Sensitivity",
            "",
            build_markdown_table(cost_display),
        ]
    )


def ensure_parent_directory(path_like: str | Path) -> None:
    Path(path_like).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    frames = {
        "walkforward_hmm_regime_strategy": load_strategy_frame(
            args.walkforward_input,
            "walk-forward HMM",
        ),
        "kmeans_regime_strategy": load_strategy_frame(
            args.kmeans_input,
            "K-means",
        ),
    }

    turnover_summary = pd.DataFrame(
        [build_turnover_summary(name, frame) for name, frame in frames.items()]
    )
    cost_rows: list[dict[str, object]] = []
    for strategy_name, frame in frames.items():
        cost_rows.extend(
            build_cost_sensitivity_rows(
                strategy_name=strategy_name,
                frame=frame,
                cost_bps_values=DEFAULT_COST_BPS,
            )
        )
    cost_sensitivity = pd.DataFrame(cost_rows)

    ensure_parent_directory(args.turnover_output)
    turnover_summary.to_csv(args.turnover_output, index=False)
    ensure_parent_directory(args.cost_output)
    cost_sensitivity.to_csv(args.cost_output, index=False)
    ensure_parent_directory(args.notes_output)
    Path(args.notes_output).write_text(
        build_notes(turnover_summary, cost_sensitivity) + "\n",
        encoding="utf-8",
    )

    print(Path(args.turnover_output))
    print(Path(args.cost_output))
    print(Path(args.notes_output))


if __name__ == "__main__":
    main()
