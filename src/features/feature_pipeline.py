"""Phase B Step 7 feature engineering pipeline."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EXPECTED_INPUT_COLUMNS = [
    "date",
    "spy_adj_close",
    "qqq_adj_close",
    "tlt_adj_close",
    "ief_adj_close",
    "gld_adj_close",
    "uso_adj_close",
    "vix_close",
    "dgs2_yield_pct",
    "dgs10_yield_pct",
]

PRICE_COLUMNS = {
    "spy_adj_close": "spy",
    "qqq_adj_close": "qqq",
    "tlt_adj_close": "tlt",
    "ief_adj_close": "ief",
    "gld_adj_close": "gld",
    "uso_adj_close": "uso",
}

VOL_WINDOW_DAYS = 20
SHORT_HORIZON_DAYS = 5
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class FeatureBuildSummary:
    output_path: str
    row_count: int
    start_date: str
    end_date: str
    columns: list[str]
    missing_value_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Phase B Step 7 modeling feature table."
    )
    parser.add_argument(
        "--input",
        default="data/processed/aligned_market_data.parquet",
        help="Path to the approved aligned market dataset parquet file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/model_features.parquet",
        help="Output path for the engineered feature table.",
    )
    return parser


def load_aligned_data(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Aligned dataset not found: {path}")

    frame = pd.read_parquet(path)
    missing_columns = [column for column in EXPECTED_INPUT_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Aligned dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = frame[EXPECTED_INPUT_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Aligned dataset contains invalid dates.")

    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)

    if frame[EXPECTED_INPUT_COLUMNS[1:]].isna().any().any():
        raise ValueError("Aligned dataset contains missing values in required columns.")

    return frame


def build_feature_table(aligned_data: pd.DataFrame) -> pd.DataFrame:
    feature_table = pd.DataFrame({"date": aligned_data["date"]})
    daily_returns = aligned_data[list(PRICE_COLUMNS)].pct_change()

    for input_column, prefix in PRICE_COLUMNS.items():
        feature_table[f"{prefix}_ret_1d"] = daily_returns[input_column]
        feature_table[f"{prefix}_vol_20d_ann"] = (
            daily_returns[input_column]
            .rolling(window=VOL_WINDOW_DAYS, min_periods=VOL_WINDOW_DAYS)
            .std()
            * math.sqrt(TRADING_DAYS_PER_YEAR)
        )

    feature_table["vix_close"] = aligned_data["vix_close"]
    feature_table["vix_change_5d"] = aligned_data["vix_close"].diff(SHORT_HORIZON_DAYS)
    feature_table["yield_curve_slope_pct"] = (
        aligned_data["dgs10_yield_pct"] - aligned_data["dgs2_yield_pct"]
    )

    feature_table["spy_minus_tlt_ret_5d"] = (
        aligned_data["spy_adj_close"].pct_change(SHORT_HORIZON_DAYS)
        - aligned_data["tlt_adj_close"].pct_change(SHORT_HORIZON_DAYS)
    )
    feature_table["qqq_minus_spy_ret_5d"] = (
        aligned_data["qqq_adj_close"].pct_change(SHORT_HORIZON_DAYS)
        - aligned_data["spy_adj_close"].pct_change(SHORT_HORIZON_DAYS)
    )
    feature_table["gld_minus_uso_ret_5d"] = (
        aligned_data["gld_adj_close"].pct_change(SHORT_HORIZON_DAYS)
        - aligned_data["uso_adj_close"].pct_change(SHORT_HORIZON_DAYS)
    )

    feature_table = feature_table.dropna().reset_index(drop=True)
    if feature_table.empty:
        raise ValueError("Feature table is empty after trimming rolling-window rows.")

    return feature_table


def save_feature_table(feature_table: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(path, index=False)
    return path


def run_feature_pipeline(
    input_path: str | Path = "data/processed/aligned_market_data.parquet",
    output_path: str | Path = "data/processed/model_features.parquet",
) -> FeatureBuildSummary:
    aligned_data = load_aligned_data(input_path)
    feature_table = build_feature_table(aligned_data)
    saved_path = save_feature_table(feature_table, output_path)

    missing_value_count = int(feature_table.isna().sum().sum())
    return FeatureBuildSummary(
        output_path=str(saved_path),
        row_count=int(len(feature_table)),
        start_date=feature_table["date"].min().date().isoformat(),
        end_date=feature_table["date"].max().date().isoformat(),
        columns=list(feature_table.columns),
        missing_value_count=missing_value_count,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_feature_pipeline(input_path=args.input, output_path=args.output)

    print(f"Saved feature table to {summary.output_path}")
    print(f"Rows: {summary.row_count}")
    print(f"Date range: {summary.start_date} to {summary.end_date}")
    print(f"Columns ({len(summary.columns)}): {', '.join(summary.columns)}")
    print(f"Missing values: {summary.missing_value_count}")


if __name__ == "__main__":
    main()
