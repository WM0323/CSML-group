"""Phase B Step 5 market data download, cleaning, and alignment pipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FRED_API_KEY_ENV = "FRED_API_KEY"


@dataclass(frozen=True)
class PipelineSummary:
    output_path: str
    row_count: int
    start_date: str
    end_date: str
    columns: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download, clean, and align market data for Phase B Step 5."
    )
    parser.add_argument(
        "--config",
        default="config/market_data_sources.json",
        help="Path to the Step 5 source configuration JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional override for the aligned dataset output path.",
    )
    parser.add_argument(
        "--no-save-raw",
        action="store_true",
        help="Skip saving the cleaned per-series raw extracts under data/raw/.",
    )
    return parser


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'yfinance'. Install it in the Python environment used "
            "to run scripts/01_download_data.py."
        ) from exc

    return yf


def _build_fred_client(api_key_env: str):
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'fredapi'. Install it in the Python environment used "
            "to run scripts/01_download_data.py."
        ) from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing {api_key_env} environment variable required by fredapi."
        )

    return Fred(api_key=api_key)


def _safe_series_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _normalize_dates(values: Any) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_localize(None).normalize()


def _resolve_price_column(history: pd.DataFrame, price_field: str) -> str:
    field_candidates = {
        "adjclose": ["Adj Close", "AdjClose"],
        "close": ["Close"],
    }
    if price_field not in field_candidates:
        raise ValueError(f"Unsupported Yahoo price_field: {price_field}")

    for candidate in field_candidates[price_field]:
        if candidate in history.columns:
            return candidate

    if price_field == "adjclose" and "Close" in history.columns:
        return "Close"

    raise RuntimeError(
        f"yfinance returned no usable {price_field} field. Available columns: "
        f"{list(history.columns)}"
    )


def fetch_yahoo_history(
    symbol: str, column_name: str, start_date: str, end_date: str, price_field: str
) -> pd.DataFrame:
    yf = _import_yfinance()
    ticker = yf.Ticker(symbol)
    end_exclusive = (pd.Timestamp(end_date) + timedelta(days=1)).date().isoformat()

    try:
        history = ticker.history(
            start=start_date,
            end=end_exclusive,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
    except Exception as exc:
        raise RuntimeError(f"yfinance failed for {symbol}: {exc}") from exc

    if history.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}.")

    history = history.reset_index()
    if "Date" in history.columns:
        date_column = "Date"
    elif "Datetime" in history.columns:
        date_column = "Datetime"
    else:
        date_column = str(history.columns[0])

    value_column = _resolve_price_column(history, price_field)
    frame = history[[date_column, value_column]].rename(
        columns={date_column: "date", value_column: column_name}
    )
    frame["date"] = _normalize_dates(frame["date"])
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.dropna(subset=[column_name]).sort_values("date")
    frame = frame.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return frame


def fetch_fred_history(
    fred_client: Any, series_id: str, column_name: str, start_date: str, end_date: str
) -> pd.DataFrame:
    try:
        series = fred_client.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
    except Exception as exc:
        raise RuntimeError(f"fredapi failed for {series_id}: {exc}") from exc

    if series is None or series.empty:
        raise RuntimeError(f"fredapi returned no rows for {series_id}.")

    frame = series.rename(column_name).rename_axis("date").reset_index()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.sort_values("date")
    frame = frame.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return frame


def save_raw_extracts(
    raw_dir: Path,
    yahoo_frames: dict[str, pd.DataFrame],
    fred_frames: dict[str, pd.DataFrame],
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for symbol, frame in yahoo_frames.items():
        frame.to_csv(raw_dir / f"yahoo_{_safe_series_name(symbol)}.csv", index=False)
    for series_id, frame in fred_frames.items():
        frame.to_csv(raw_dir / f"fred_{_safe_series_name(series_id)}.csv", index=False)


def align_frames(
    config: dict[str, Any],
    yahoo_frames: dict[str, pd.DataFrame],
    fred_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    yahoo_specs = config["yahoo_symbols"]
    fred_specs = config["fred_series"]
    anchor_symbol = config["anchor_symbol"]

    anchor_spec = next(
        (spec for spec in yahoo_specs if spec["symbol"] == anchor_symbol),
        None,
    )
    if anchor_spec is None:
        raise RuntimeError(f"Anchor symbol {anchor_symbol} is missing from config.")

    aligned = yahoo_frames[anchor_symbol][["date"]].copy()

    for spec in yahoo_specs:
        aligned = aligned.merge(
            yahoo_frames[spec["symbol"]][["date", spec["column"]]],
            on="date",
            how="left",
        )

    for spec in fred_specs:
        aligned = aligned.merge(
            fred_frames[spec["series_id"]][["date", spec["column"]]],
            on="date",
            how="left",
        )

    market_columns = [spec["column"] for spec in yahoo_specs]
    macro_columns = [spec["column"] for spec in fred_specs]

    aligned = aligned.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    aligned[macro_columns] = aligned[macro_columns].ffill()
    aligned = aligned.dropna(subset=market_columns + macro_columns).reset_index(drop=True)

    return aligned


def validate_aligned_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RuntimeError("The aligned dataset is empty after cleaning.")
    if frame["date"].duplicated().any():
        raise RuntimeError("The aligned dataset contains duplicate dates.")
    if not frame["date"].is_monotonic_increasing:
        raise RuntimeError("The aligned dataset is not sorted by date.")
    data_columns = [col for col in frame.columns if col != "date"]
    missing_summary = frame[data_columns].isna().sum()
    remaining_missing = missing_summary[missing_summary > 0]
    if not remaining_missing.empty:
        details = remaining_missing.to_dict()
        raise RuntimeError(f"Unexpected missing values remain after cleaning: {details}")


def write_output(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        frame.to_csv(output_path, index=False)
        return output_path

    if output_path.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet export requires pyarrow or fastparquet. "
                "Install one of them, or rerun with --output data/processed/aligned_market_data.csv."
            ) from exc
        return output_path

    raise RuntimeError(
        f"Unsupported output extension for {output_path}. Use .parquet or .csv."
    )


def run_pipeline(
    config_path: str | Path,
    output_path: str | Path | None = None,
    save_raw: bool = True,
) -> PipelineSummary:
    config = load_config(config_path)

    start_date = config["start_date"]
    end_date = config.get("end_date") or date.today().isoformat()
    raw_dir = Path(config.get("raw_dir", "data/raw"))
    output_path = output_path or config["output_path"]
    fred_api_key_env = config.get("fred_api_key_env", DEFAULT_FRED_API_KEY_ENV)

    # Fail fast on missing runtime requirements before any partial download work begins.
    _import_yfinance()
    fred_client = _build_fred_client(fred_api_key_env)

    yahoo_frames: dict[str, pd.DataFrame] = {}
    fred_frames: dict[str, pd.DataFrame] = {}

    for spec in config["yahoo_symbols"]:
        print(f"Fetching Yahoo Finance series via yfinance: {spec['symbol']}")
        yahoo_frames[spec["symbol"]] = fetch_yahoo_history(
            symbol=spec["symbol"],
            column_name=spec["column"],
            start_date=start_date,
            end_date=end_date,
            price_field=spec["price_field"],
        )

    for spec in config["fred_series"]:
        print(f"Fetching FRED series via fredapi: {spec['series_id']}")
        fred_frames[spec["series_id"]] = fetch_fred_history(
            fred_client=fred_client,
            series_id=spec["series_id"],
            column_name=spec["column"],
            start_date=start_date,
            end_date=end_date,
        )

    if save_raw:
        save_raw_extracts(raw_dir=raw_dir, yahoo_frames=yahoo_frames, fred_frames=fred_frames)

    aligned = align_frames(config=config, yahoo_frames=yahoo_frames, fred_frames=fred_frames)
    validate_aligned_frame(aligned)
    written_path = write_output(aligned, output_path)

    return PipelineSummary(
        output_path=str(written_path),
        row_count=len(aligned),
        start_date=aligned["date"].min().date().isoformat(),
        end_date=aligned["date"].max().date().isoformat(),
        columns=aligned.columns.tolist(),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_pipeline(
        config_path=args.config,
        output_path=args.output,
        save_raw=not args.no_save_raw,
    )
    print(f"Output path: {summary.output_path}")
    print(f"Rows: {summary.row_count}")
    print(f"Date range: {summary.start_date} -> {summary.end_date}")
    print(f"Columns: {', '.join(summary.columns)}")


if __name__ == "__main__":
    main()
