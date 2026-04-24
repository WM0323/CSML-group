"""Backtest the walk-forward HMM regime signal with allocation-scheme variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_performance_metrics
from src.portfolio.allocation_schemes import (
    AssetBounds,
    BenchmarkRelativeTiltConfig,
    ConfidenceGateConfig,
    DEFAULT_ASSET_BOUNDS,
    DEFAULT_CONFIDENCE_GATE,
    DEFAULT_OPTIMIZER_CONFIG,
    DEFAULT_TILT_CONFIG,
    OptimizerConfig,
    OptimizedTemplateBlockSummary,
    build_blockwise_optimized_weight_frame,
    build_hard_label_weight_frame,
    build_probability_weighted_weight_frame,
    build_tilt_weight_frame,
    serialize_asset_bounds,
    serialize_confidence_gate,
    serialize_optimizer_config,
    serialize_tilt_config,
)
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
    require_existing_file,
    run_benchmark_backtests,
    run_strategy_backtest_from_weights,
    validate_allocation_map,
)


HMM_PROBABILITY_COLUMNS = [f"state_{state}_probability" for state in sorted(STATE_LABELS)]
PRIMARY_VARIANT_CHOICES = [
    "hard_label_fixed_map",
    "probability_weighted_fixed_templates",
    "benchmark_relative_tilt",
    "benchmark_relative_tilt_confidence_gate",
    "training_window_regime_optimizer",
]


@dataclass(frozen=True)
class WalkforwardBacktestRunSummary:
    market_data_path: str
    hmm_labels_path: str
    hmm_probabilities_path: str
    hmm_metadata_path: str
    hmm_notes_path: str
    strategy_returns_path: str
    strategy_variant_returns_path: str
    benchmark_returns_path: str
    metrics_path: str
    sweep_metrics_path: str
    metadata_path: str
    notes_path: str
    primary_variant_name: str
    row_count: int
    start_date: str
    end_date: str
    traded_assets: list[str]
    benchmark_names: list[str]
    regime_state_map: dict[int, str]
    variant_names: list[str]


@dataclass(frozen=True)
class AllocationVariantSpec:
    name: str
    description: str
    uses_probabilities: bool
    uses_confidence_gate: bool
    uses_optimizer: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run walk-forward HMM allocation backtests using hard labels, predictive "
            "probabilities, benchmark-relative tilts, and an optional training-window optimizer."
        )
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
        "--hmm-probabilities-input",
        default="results/regimes/hmm_walkforward_state_probabilities.parquet",
        help="Path to the walk-forward HMM predictive probability parquet file.",
    )
    parser.add_argument(
        "--hmm-metadata-input",
        default="results/models/hmm_walkforward_metadata.json",
        help="Path to the walk-forward HMM metadata JSON file.",
    )
    parser.add_argument(
        "--hmm-notes-input",
        default="docs/hmm_walkforward_notes.md",
        help="Path to the walk-forward HMM notes.",
    )
    parser.add_argument(
        "--primary-variant",
        default="benchmark_relative_tilt_confidence_gate",
        choices=PRIMARY_VARIANT_CHOICES,
        help=(
            "Variant exported to the legacy walkforward_strategy_returns.parquet path. "
            "The full sweep is always written separately."
        ),
    )
    parser.add_argument(
        "--strategy-output",
        default="results/backtests/walkforward_strategy_returns.parquet",
        help="Output parquet path for the primary walk-forward strategy returns and weights.",
    )
    parser.add_argument(
        "--strategy-variants-output",
        default="results/backtests/walkforward_strategy_variant_returns.parquet",
        help="Output parquet path for all allocation-variant daily returns and weights.",
    )
    parser.add_argument(
        "--benchmarks-output",
        default="results/backtests/walkforward_benchmark_returns.parquet",
        help="Output parquet path for daily benchmark returns on the walk-forward sample.",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/backtests/walkforward_strategy_metrics.csv",
        help="Output CSV path for the primary strategy and benchmark metrics.",
    )
    parser.add_argument(
        "--sweep-metrics-output",
        default="results/backtests/walkforward_allocation_sweep_metrics.csv",
        help="Output CSV path for the allocation sensitivity sweep metrics.",
    )
    parser.add_argument(
        "--metadata-output",
        default="results/backtests/walkforward_allocation_sweep_metadata.json",
        help="Output JSON path for allocation-scheme metadata and optimizer notes.",
    )
    parser.add_argument(
        "--notes-output",
        default="docs/backtest_walkforward_notes.md",
        help="Output Markdown path for concise walk-forward backtest notes.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_GATE.threshold,
        help="Confidence gate threshold applied to max state probability.",
    )
    parser.add_argument(
        "--confidence-scale",
        type=float,
        default=DEFAULT_CONFIDENCE_GATE.scale,
        help=(
            "Confidence gate scale. A value of 0.25 means tilt strength ramps from zero "
            "at the threshold to full strength 0.25 probability points above it."
        ),
    )
    parser.add_argument(
        "--spy-floor",
        type=float,
        default=DEFAULT_ASSET_BOUNDS.floors["SPY"],
        help="Minimum SPY portfolio weight for the benchmark-relative tilt designs.",
    )
    parser.add_argument(
        "--spy-cap",
        type=float,
        default=DEFAULT_ASSET_BOUNDS.caps["SPY"],
        help="Maximum SPY portfolio weight for the benchmark-relative tilt designs.",
    )
    parser.add_argument(
        "--tlt-floor",
        type=float,
        default=DEFAULT_ASSET_BOUNDS.floors["TLT"],
        help="Minimum TLT portfolio weight for the benchmark-relative tilt designs.",
    )
    parser.add_argument(
        "--tlt-cap",
        type=float,
        default=DEFAULT_ASSET_BOUNDS.caps["TLT"],
        help="Maximum TLT portfolio weight for the benchmark-relative tilt designs.",
    )
    parser.add_argument(
        "--ief-floor",
        type=float,
        default=DEFAULT_ASSET_BOUNDS.floors["IEF"],
        help="Minimum IEF portfolio weight for the benchmark-relative tilt designs.",
    )
    parser.add_argument(
        "--ief-cap",
        type=float,
        default=DEFAULT_ASSET_BOUNDS.caps["IEF"],
        help="Maximum IEF portfolio weight for the benchmark-relative tilt designs.",
    )
    parser.add_argument(
        "--gld-floor",
        type=float,
        default=DEFAULT_ASSET_BOUNDS.floors["GLD"],
        help="Minimum GLD portfolio weight for the benchmark-relative tilt designs.",
    )
    parser.add_argument(
        "--gld-cap",
        type=float,
        default=DEFAULT_ASSET_BOUNDS.caps["GLD"],
        help="Maximum GLD portfolio weight for the benchmark-relative tilt designs.",
    )
    parser.add_argument(
        "--optimizer-method",
        choices=["none", "min_variance", "mean_variance"],
        default="none",
        help=(
            "Optional training-window optimizer used to generate regime-conditional "
            "allocation templates at each HMM refit block."
        ),
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


def load_walkforward_probabilities(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Walk-forward HMM probabilities not found: {path}")

    frame = pd.read_parquet(path)
    required_columns = ["date", *HMM_PROBABILITY_COLUMNS]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Walk-forward HMM probabilities are missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = frame[required_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Walk-forward HMM probabilities contain duplicate dates after sorting.")

    probability_values = frame[HMM_PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if np.isnan(probability_values).any():
        raise ValueError("Walk-forward HMM probabilities contain missing values.")
    if not np.allclose(probability_values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Walk-forward HMM probabilities do not sum to 1.0 within tolerance.")
    if (probability_values < -1e-9).any():
        raise ValueError("Walk-forward HMM probabilities contain negative values.")
    return frame


def load_walkforward_metadata(input_path: str | Path) -> dict[str, object]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Walk-forward HMM metadata not found: {path}")

    metadata = json.loads(path.read_text(encoding="utf-8"))
    refit_blocks = metadata.get("refit_blocks")
    if not isinstance(refit_blocks, list) or not refit_blocks:
        raise ValueError("Walk-forward HMM metadata is missing refit_blocks.")

    required_keys = {"refit_date", "training_end_date", "block_end_date"}
    for block in refit_blocks:
        if not required_keys.issubset(block):
            raise ValueError("Each refit block must include refit_date, training_end_date, and block_end_date.")
    return metadata


def build_real_time_signal_frame(
    asset_return_frame: pd.DataFrame,
    walkforward_labels: pd.DataFrame,
    walkforward_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        asset_return_frame.merge(
            walkforward_labels,
            on="date",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            walkforward_probabilities,
            on="date",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return_columns = [f"{asset.lower()}_return" for asset in TRADED_ASSETS]
    merged = merged.dropna(subset=return_columns).reset_index(drop=True)
    merged["signal_state"] = merged["hmm_state"].astype(int)
    merged["hmm_state_name"] = merged["hmm_state"].map(STATE_LABELS)
    merged["signal_state_name"] = merged["signal_state"].map(STATE_LABELS)
    merged["signal_confidence"] = merged[HMM_PROBABILITY_COLUMNS].max(axis=1)

    if merged.empty:
        raise ValueError("Walk-forward backtest alignment produced no usable rows.")
    if merged["signal_state_name"].isna().any():
        raise ValueError("Walk-forward signal frame contains unknown state labels.")

    argmax_states = merged[HMM_PROBABILITY_COLUMNS].to_numpy(dtype=float).argmax(axis=1)
    if not np.array_equal(argmax_states.astype(int), merged["hmm_state"].to_numpy(dtype=int)):
        raise ValueError("Walk-forward labels do not match the predictive probability argmax.")

    return merged


def build_asset_bounds_from_args(args: argparse.Namespace) -> AssetBounds:
    return AssetBounds(
        floors={
            "SPY": float(args.spy_floor),
            "TLT": float(args.tlt_floor),
            "IEF": float(args.ief_floor),
            "GLD": float(args.gld_floor),
        },
        caps={
            "SPY": float(args.spy_cap),
            "TLT": float(args.tlt_cap),
            "IEF": float(args.ief_cap),
            "GLD": float(args.gld_cap),
        },
    )


def build_variant_specs(
    include_optimizer: bool,
) -> dict[str, AllocationVariantSpec]:
    specs = {
        "hard_label_fixed_map": AllocationVariantSpec(
            name="hard_label_fixed_map",
            description="Legacy hard-label strategy using the existing hand-picked regime map.",
            uses_probabilities=False,
            uses_confidence_gate=False,
            uses_optimizer=False,
        ),
        "probability_weighted_fixed_templates": AllocationVariantSpec(
            name="probability_weighted_fixed_templates",
            description=(
                "Probability-weighted version of the legacy hand-picked templates using the "
                "predictive HMM state probabilities directly."
            ),
            uses_probabilities=True,
            uses_confidence_gate=False,
            uses_optimizer=False,
        ),
        "benchmark_relative_tilt": AllocationVariantSpec(
            name="benchmark_relative_tilt",
            description=(
                "Probability-weighted benchmark-relative tilt around the fixed 60/40 stock-bond base."
            ),
            uses_probabilities=True,
            uses_confidence_gate=False,
            uses_optimizer=False,
        ),
        "benchmark_relative_tilt_confidence_gate": AllocationVariantSpec(
            name="benchmark_relative_tilt_confidence_gate",
            description=(
                "Benchmark-relative tilt with a confidence gate that shrinks low-conviction "
                "signals back toward the fixed 60/40 base."
            ),
            uses_probabilities=True,
            uses_confidence_gate=True,
            uses_optimizer=False,
        ),
    }
    if include_optimizer:
        specs["training_window_regime_optimizer"] = AllocationVariantSpec(
            name="training_window_regime_optimizer",
            description=(
                "Blockwise probability-weighted regime templates optimized on training-window "
                "historical returns only, with regularized long-only constraints and benchmark-relative fallbacks."
            ),
            uses_probabilities=True,
            uses_confidence_gate=False,
            uses_optimizer=True,
        )
    return specs


def annotate_strategy_frame(
    strategy_frame: pd.DataFrame,
    variant_spec: AllocationVariantSpec,
    is_primary_strategy: bool,
) -> pd.DataFrame:
    annotated = strategy_frame.copy()
    annotated.insert(1, "allocation_variant", variant_spec.name)
    annotated.insert(2, "allocation_description", variant_spec.description)
    annotated.insert(3, "is_primary_strategy", bool(is_primary_strategy))
    return annotated


def build_primary_portfolio_name(variant_name: str) -> str:
    return f"walkforward_hmm_{variant_name}"


def build_sweep_metrics_frame(
    strategy_frames: dict[str, pd.DataFrame],
    benchmark_frame: pd.DataFrame,
    benchmark_names: list[str],
    variant_specs: dict[str, AllocationVariantSpec],
    primary_variant_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, strategy_frame in strategy_frames.items():
        metrics = calculate_performance_metrics(strategy_frame["strategy_return"])
        variant_spec = variant_specs[variant_name]
        rows.append(
            {
                "portfolio_name": variant_name,
                "portfolio_type": "strategy",
                "description": variant_spec.description,
                "is_primary_strategy": variant_name == primary_variant_name,
                "uses_probabilities": variant_spec.uses_probabilities,
                "uses_confidence_gate": variant_spec.uses_confidence_gate,
                "uses_optimizer": variant_spec.uses_optimizer,
                **metrics,
            }
        )

    for benchmark_name in benchmark_names:
        metrics = calculate_performance_metrics(benchmark_frame[f"{benchmark_name}_return"])
        rows.append(
            {
                "portfolio_name": benchmark_name,
                "portfolio_type": "benchmark",
                "description": "Static benchmark portfolio.",
                "is_primary_strategy": False,
                "uses_probabilities": False,
                "uses_confidence_gate": False,
                "uses_optimizer": False,
                **metrics,
            }
        )

    sweep_frame = pd.DataFrame(rows)
    baseline = sweep_frame.loc[sweep_frame["portfolio_name"] == "hard_label_fixed_map"].iloc[0]
    equal_weight = sweep_frame.loc[
        sweep_frame["portfolio_name"] == "equal_weight_4_asset"
    ].iloc[0]
    fixed_sixty_forty = sweep_frame.loc[
        sweep_frame["portfolio_name"] == "fixed_60_40_stock_bond"
    ].iloc[0]

    strategy_mask = sweep_frame["portfolio_type"] == "strategy"
    sweep_frame["delta_cumulative_return_vs_hard_label"] = np.nan
    sweep_frame["delta_sharpe_ratio_vs_hard_label"] = np.nan
    sweep_frame["delta_cumulative_return_vs_equal_weight"] = np.nan
    sweep_frame["delta_sharpe_ratio_vs_equal_weight"] = np.nan
    sweep_frame["delta_cumulative_return_vs_fixed_60_40"] = np.nan
    sweep_frame["delta_sharpe_ratio_vs_fixed_60_40"] = np.nan

    sweep_frame.loc[strategy_mask, "delta_cumulative_return_vs_hard_label"] = (
        sweep_frame.loc[strategy_mask, "cumulative_return"] - float(baseline["cumulative_return"])
    )
    sweep_frame.loc[strategy_mask, "delta_sharpe_ratio_vs_hard_label"] = (
        sweep_frame.loc[strategy_mask, "sharpe_ratio"] - float(baseline["sharpe_ratio"])
    )
    sweep_frame.loc[strategy_mask, "delta_cumulative_return_vs_equal_weight"] = (
        sweep_frame.loc[strategy_mask, "cumulative_return"] - float(equal_weight["cumulative_return"])
    )
    sweep_frame.loc[strategy_mask, "delta_sharpe_ratio_vs_equal_weight"] = (
        sweep_frame.loc[strategy_mask, "sharpe_ratio"] - float(equal_weight["sharpe_ratio"])
    )
    sweep_frame.loc[strategy_mask, "delta_cumulative_return_vs_fixed_60_40"] = (
        sweep_frame.loc[strategy_mask, "cumulative_return"]
        - float(fixed_sixty_forty["cumulative_return"])
    )
    sweep_frame.loc[strategy_mask, "delta_sharpe_ratio_vs_fixed_60_40"] = (
        sweep_frame.loc[strategy_mask, "sharpe_ratio"] - float(fixed_sixty_forty["sharpe_ratio"])
    )

    return sweep_frame


def build_variant_returns_frame(
    strategy_frames: dict[str, pd.DataFrame],
    variant_specs: dict[str, AllocationVariantSpec],
    primary_variant_name: str,
) -> pd.DataFrame:
    variant_frames = [
        annotate_strategy_frame(
            strategy_frame=strategy_frame,
            variant_spec=variant_specs[variant_name],
            is_primary_strategy=(variant_name == primary_variant_name),
        )
        for variant_name, strategy_frame in strategy_frames.items()
    ]
    return pd.concat(variant_frames, ignore_index=True)


def build_variant_summary_table(variant_specs: dict[str, AllocationVariantSpec]) -> str:
    rows = [
        [
            variant_spec.name,
            "yes" if variant_spec.uses_probabilities else "no",
            "yes" if variant_spec.uses_confidence_gate else "no",
            "yes" if variant_spec.uses_optimizer else "no",
            variant_spec.description,
        ]
        for variant_spec in variant_specs.values()
    ]
    return build_markdown_table(
        headers=["variant", "uses_probabilities", "confidence_gate", "optimizer", "description"],
        rows=rows,
    )


def build_asset_bounds_table(asset_bounds: AssetBounds) -> str:
    rows = [
        [
            record["asset"],
            format_percentage(record["floor"]),
            format_percentage(record["cap"]),
        ]
        for record in asset_bounds.as_rows()
    ]
    return build_markdown_table(headers=["asset", "floor", "cap"], rows=rows)


def build_sweep_metrics_markdown(sweep_metrics_frame: pd.DataFrame) -> str:
    display_frame = sweep_metrics_frame[
        [
            "portfolio_name",
            "portfolio_type",
            "observations",
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

    rows = [
        [str(value) for value in row]
        for row in display_frame.itertuples(index=False, name=None)
    ]
    return build_markdown_table(headers=list(display_frame.columns), rows=rows)


def build_notes_text(
    summary: WalkforwardBacktestRunSummary,
    primary_metrics_frame: pd.DataFrame,
    sweep_metrics_frame: pd.DataFrame,
    variant_specs: dict[str, AllocationVariantSpec],
    benchmark_relative_template_map: dict[int, dict[str, float]],
    asset_bounds: AssetBounds,
    confidence_gate: ConfidenceGateConfig,
    optimizer_config: OptimizerConfig | None,
    optimizer_block_summaries: list[OptimizedTemplateBlockSummary],
) -> str:
    primary_portfolio_name = build_primary_portfolio_name(summary.primary_variant_name)
    primary_metrics = primary_metrics_frame.loc[
        primary_metrics_frame["portfolio_name"] == primary_portfolio_name
    ].iloc[0]
    strategy_only = sweep_metrics_frame.loc[
        sweep_metrics_frame["portfolio_type"] == "strategy"
    ].copy()
    best_strategy_by_sharpe = strategy_only.sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_strategy_by_return = strategy_only.sort_values(
        "cumulative_return",
        ascending=False,
    ).iloc[0]
    baseline = strategy_only.loc[
        strategy_only["portfolio_name"] == "hard_label_fixed_map"
    ].iloc[0]
    equal_weight = sweep_metrics_frame.loc[
        sweep_metrics_frame["portfolio_name"] == "equal_weight_4_asset"
    ].iloc[0]
    fixed_sixty_forty = sweep_metrics_frame.loc[
        sweep_metrics_frame["portfolio_name"] == "fixed_60_40_stock_bond"
    ].iloc[0]

    optimizer_lines: list[str] = []
    if optimizer_config is not None and optimizer_block_summaries:
        optimized_blocks = sum(
            1 for block in optimizer_block_summaries if len(block.optimized_states) > 0
        )
        any_fallback_blocks = sum(
            1 for block in optimizer_block_summaries if len(block.fallback_states) > 0
        )
        optimizer_lines.extend(
            [
                "## Training-Window Optimizer",
                "",
                (
                    "- Optional optimizer status: enabled. Each HMM refit block uses only returns "
                    "and predictive probabilities observed up to that block's training end date."
                ),
                (
                    f"- Optimizer method: `{optimizer_config.method}` with covariance shrinkage "
                    f"{optimizer_config.covariance_shrinkage:.2f}, ridge penalty "
                    f"{optimizer_config.ridge_penalty:.1e}, and template blend "
                    f"{optimizer_config.template_blend:.2f}."
                ),
                (
                    f"- Blocks with at least one optimized regime template: {optimized_blocks} "
                    f"of {len(optimizer_block_summaries)}."
                ),
                (
                    f"- Blocks requiring at least one benchmark-relative fallback template: "
                    f"{any_fallback_blocks} of {len(optimizer_block_summaries)}."
                ),
                "",
            ]
        )

    return "\n".join(
        [
            "# Walk-Forward Backtest Notes",
            "",
            "- Strategy family: long-only HMM-driven asset allocation on the same walk-forward signal dates as the existing project.",
            (
                "- Signal timing: the exported walk-forward HMM probability vector on date `t` is "
                "already the real-time predictive signal `P(z_t | x_1:t-1)`, so it is applied "
                "directly to returns on date `t` without any extra lag."
            ),
            (
                "- Fairness rule: the primary exported strategy is chosen ex ante by configuration "
                f"(`{summary.primary_variant_name}`), while the full allocation sweep below is "
                "reported only as sensitivity analysis and not used to re-select the signal itself."
            ),
            (
                "- No look-ahead: the HMM signal remains unchanged, the probability-weighted and "
                "tilt variants use only same-day predictive probabilities, and the optional optimizer "
                "uses only historical rows available before each refit block begins."
            ),
            "- Transaction costs: omitted in this pass to preserve the existing evaluation logic.",
            f"- Approved interpretation source: `{summary.hmm_notes_path}`",
            (
                f"- Backtest sample: `{summary.start_date}` to `{summary.end_date}` "
                f"({summary.row_count} rows)"
            ),
            "",
            "## Allocation Variants",
            "",
            build_variant_summary_table(variant_specs),
            "",
            "## Legacy Hard-Label Templates",
            "",
            build_allocation_table(REGIME_ALLOCATIONS),
            "",
            "## Benchmark-Relative Tilt Templates",
            "",
            build_allocation_table(benchmark_relative_template_map),
            "",
            "## Tilt Asset Bounds",
            "",
            build_asset_bounds_table(asset_bounds),
            "",
            "## Static Benchmarks",
            "",
            build_benchmark_table(),
            "",
            "## Allocation Sweep Metrics",
            "",
            build_sweep_metrics_markdown(sweep_metrics_frame),
            "",
            "## Primary Strategy Metrics",
            "",
            build_metrics_markdown(primary_metrics_frame),
            "",
            "## Short Interpretation",
            "",
            (
                f"- The configured primary export `{summary.primary_variant_name}` achieved "
                f"cumulative return {format_percentage(primary_metrics['cumulative_return'])}, "
                f"annualized volatility {format_percentage(primary_metrics['annualized_volatility'])}, "
                f"and Sharpe ratio {format_decimal(primary_metrics['sharpe_ratio'])}."
            ),
            (
                f"- Relative to the current hard-label map, the primary export changed cumulative "
                f"return by {format_percentage(primary_metrics['cumulative_return'] - baseline['cumulative_return'])} "
                f"and Sharpe by {format_decimal(primary_metrics['sharpe_ratio'] - baseline['sharpe_ratio'])}."
            ),
            (
                f"- The highest-Sharpe dynamic variant in the sweep was "
                f"`{best_strategy_by_sharpe['portfolio_name']}` at "
                f"{format_decimal(best_strategy_by_sharpe['sharpe_ratio'])}."
            ),
            (
                f"- The highest cumulative-return dynamic variant in the sweep was "
                f"`{best_strategy_by_return['portfolio_name']}` at "
                f"{format_percentage(best_strategy_by_return['cumulative_return'])}."
            ),
            (
                f"- Against static benchmarks, the best dynamic Sharpe was "
                f"{format_decimal(best_strategy_by_sharpe['sharpe_ratio'])} versus "
                f"{format_decimal(equal_weight['sharpe_ratio'])} for equal-weight and "
                f"{format_decimal(fixed_sixty_forty['sharpe_ratio'])} for fixed 60/40."
            ),
            (
                f"- Against static benchmarks, the best dynamic cumulative return was "
                f"{format_percentage(best_strategy_by_return['cumulative_return'])} versus "
                f"{format_percentage(equal_weight['cumulative_return'])} for equal-weight and "
                f"{format_percentage(fixed_sixty_forty['cumulative_return'])} for fixed 60/40."
            ),
            "",
            "## Confidence Gate",
            "",
            (
                f"- Confidence definition: `max(state probabilities)` on each tradable date."
            ),
            (
                f"- Gate parameters: threshold `{confidence_gate.threshold:.2f}` and scale "
                f"`{confidence_gate.scale:.2f}`. Below the threshold, the portfolio stays close to the "
                "base 60/40 allocation; above it, regime tilts increase linearly."
            ),
            "",
            *optimizer_lines,
            "## Validation Notes",
            "",
            (
                f"- Input artifact dates aligned successfully across `{summary.hmm_labels_path}`, "
                f"`{summary.hmm_probabilities_path}`, and the market return frame."
            ),
            "- The hard-label variant remains available unchanged for direct comparison.",
            "- The probability-weighted variants use the exported predictive probabilities directly instead of only the argmax state.",
            "- All benchmark-relative tilt weights stay long-only, respect configured per-asset caps and floors, and sum to 1.0.",
        ]
    )


def run_walkforward_backtest_pipeline(args: argparse.Namespace) -> WalkforwardBacktestRunSummary:
    validate_allocation_map(REGIME_ALLOCATIONS, expected_keys=[0, 1, 2], name="regime")
    validate_allocation_map(
        BENCHMARK_ALLOCATIONS,
        expected_keys=list(BENCHMARK_ALLOCATIONS),
        name="benchmark",
    )

    asset_bounds = build_asset_bounds_from_args(args)
    asset_bounds.validate()
    confidence_gate = ConfidenceGateConfig(
        threshold=float(args.confidence_threshold),
        scale=float(args.confidence_scale),
    )
    confidence_gate.validate()
    tilt_config = BenchmarkRelativeTiltConfig(
        base_allocation=dict(DEFAULT_TILT_CONFIG.base_allocation),
        risk_off_delta=dict(DEFAULT_TILT_CONFIG.risk_off_delta),
        risk_on_delta=dict(DEFAULT_TILT_CONFIG.risk_on_delta),
    )
    optimizer_config = None
    if args.optimizer_method != "none":
        optimizer_config = OptimizerConfig(
            method=str(args.optimizer_method),
            covariance_shrinkage=DEFAULT_OPTIMIZER_CONFIG.covariance_shrinkage,
            ridge_penalty=DEFAULT_OPTIMIZER_CONFIG.ridge_penalty,
            mean_shrinkage=DEFAULT_OPTIMIZER_CONFIG.mean_shrinkage,
            risk_aversion=DEFAULT_OPTIMIZER_CONFIG.risk_aversion,
            template_blend=DEFAULT_OPTIMIZER_CONFIG.template_blend,
            min_effective_observations=DEFAULT_OPTIMIZER_CONFIG.min_effective_observations,
            max_iterations=DEFAULT_OPTIMIZER_CONFIG.max_iterations,
            tolerance=DEFAULT_OPTIMIZER_CONFIG.tolerance,
        )
        optimizer_config.validate()

    variant_specs = build_variant_specs(include_optimizer=optimizer_config is not None)
    if args.primary_variant not in variant_specs:
        raise ValueError(
            f"Primary variant `{args.primary_variant}` is unavailable under the current configuration."
        )

    hmm_notes_path = require_existing_file(args.hmm_notes_input, "Walk-forward HMM notes")
    market_data = load_market_data(args.market_data_input)
    walkforward_labels = load_walkforward_labels(args.hmm_labels_input)
    walkforward_probabilities = load_walkforward_probabilities(args.hmm_probabilities_input)
    walkforward_metadata = load_walkforward_metadata(args.hmm_metadata_input)
    refit_blocks = list(walkforward_metadata["refit_blocks"])

    asset_return_frame = build_asset_return_frame(market_data)
    signal_frame = build_real_time_signal_frame(
        asset_return_frame=asset_return_frame,
        walkforward_labels=walkforward_labels,
        walkforward_probabilities=walkforward_probabilities,
    )
    benchmark_frame = run_benchmark_backtests(signal_frame, BENCHMARK_ALLOCATIONS)
    benchmark_names = list(BENCHMARK_ALLOCATIONS.keys())

    strategy_frames: dict[str, pd.DataFrame] = {}

    hard_label_frame = run_strategy_backtest_from_weights(
        signal_frame=signal_frame,
        weight_frame=build_hard_label_weight_frame(signal_frame, REGIME_ALLOCATIONS),
        name="hard_label_fixed_map",
    )
    strategy_frames["hard_label_fixed_map"] = hard_label_frame

    probability_weighted_fixed_frame = run_strategy_backtest_from_weights(
        signal_frame=signal_frame,
        weight_frame=build_probability_weighted_weight_frame(
            signal_frame,
            REGIME_ALLOCATIONS,
            asset_bounds=None,
        ),
        name="probability_weighted_fixed_templates",
    )
    strategy_frames["probability_weighted_fixed_templates"] = probability_weighted_fixed_frame

    tilt_weight_frame, benchmark_relative_template_map = build_tilt_weight_frame(
        signal_frame=signal_frame,
        tilt_config=tilt_config,
        asset_bounds=asset_bounds,
        confidence_gate=None,
    )
    benchmark_relative_tilt_frame = run_strategy_backtest_from_weights(
        signal_frame=signal_frame,
        weight_frame=tilt_weight_frame,
        name="benchmark_relative_tilt",
    )
    strategy_frames["benchmark_relative_tilt"] = benchmark_relative_tilt_frame

    gated_tilt_weight_frame, _ = build_tilt_weight_frame(
        signal_frame=signal_frame,
        tilt_config=tilt_config,
        asset_bounds=asset_bounds,
        confidence_gate=confidence_gate,
    )
    benchmark_relative_tilt_confidence_frame = run_strategy_backtest_from_weights(
        signal_frame=signal_frame,
        weight_frame=gated_tilt_weight_frame,
        name="benchmark_relative_tilt_confidence_gate",
    )
    strategy_frames["benchmark_relative_tilt_confidence_gate"] = (
        benchmark_relative_tilt_confidence_frame
    )

    optimizer_block_summaries: list[OptimizedTemplateBlockSummary] = []
    if optimizer_config is not None:
        optimizer_weight_frame, optimizer_block_summaries = build_blockwise_optimized_weight_frame(
            signal_frame=signal_frame,
            refit_blocks=refit_blocks,
            fallback_template_map=benchmark_relative_template_map,
            asset_bounds=asset_bounds,
            optimizer_config=optimizer_config,
        )
        optimizer_frame = run_strategy_backtest_from_weights(
            signal_frame=signal_frame,
            weight_frame=optimizer_weight_frame,
            name="training_window_regime_optimizer",
        )
        strategy_frames["training_window_regime_optimizer"] = optimizer_frame

    primary_strategy_frame = annotate_strategy_frame(
        strategy_frame=strategy_frames[args.primary_variant],
        variant_spec=variant_specs[args.primary_variant],
        is_primary_strategy=True,
    )
    primary_metrics_frame = build_metrics_table(
        strategy_frame=strategy_frames[args.primary_variant],
        benchmark_frame=benchmark_frame,
        benchmark_names=benchmark_names,
        strategy_name=build_primary_portfolio_name(args.primary_variant),
    )
    sweep_metrics_frame = build_sweep_metrics_frame(
        strategy_frames=strategy_frames,
        benchmark_frame=benchmark_frame,
        benchmark_names=benchmark_names,
        variant_specs=variant_specs,
        primary_variant_name=args.primary_variant,
    )
    variant_returns_frame = build_variant_returns_frame(
        strategy_frames=strategy_frames,
        variant_specs=variant_specs,
        primary_variant_name=args.primary_variant,
    )

    summary = WalkforwardBacktestRunSummary(
        market_data_path=str(Path(args.market_data_input)),
        hmm_labels_path=str(Path(args.hmm_labels_input)),
        hmm_probabilities_path=str(Path(args.hmm_probabilities_input)),
        hmm_metadata_path=str(Path(args.hmm_metadata_input)),
        hmm_notes_path=str(hmm_notes_path),
        strategy_returns_path=str(Path(args.strategy_output)),
        strategy_variant_returns_path=str(Path(args.strategy_variants_output)),
        benchmark_returns_path=str(Path(args.benchmarks_output)),
        metrics_path=str(Path(args.metrics_output)),
        sweep_metrics_path=str(Path(args.sweep_metrics_output)),
        metadata_path=str(Path(args.metadata_output)),
        notes_path=str(Path(args.notes_output)),
        primary_variant_name=str(args.primary_variant),
        row_count=int(len(primary_strategy_frame)),
        start_date=str(primary_strategy_frame["date"].min().date()),
        end_date=str(primary_strategy_frame["date"].max().date()),
        traded_assets=list(TRADED_ASSETS.keys()),
        benchmark_names=benchmark_names,
        regime_state_map=STATE_LABELS,
        variant_names=list(strategy_frames.keys()),
    )

    metadata_payload = {
        "market_data_path": summary.market_data_path,
        "hmm_labels_path": summary.hmm_labels_path,
        "hmm_probabilities_path": summary.hmm_probabilities_path,
        "hmm_metadata_path": summary.hmm_metadata_path,
        "primary_variant_name": summary.primary_variant_name,
        "variant_specs": [asdict(variant_spec) for variant_spec in variant_specs.values()],
        "legacy_regime_allocations": REGIME_ALLOCATIONS,
        "benchmark_relative_tilt_config": serialize_tilt_config(tilt_config),
        "benchmark_relative_templates": benchmark_relative_template_map,
        "asset_bounds": serialize_asset_bounds(asset_bounds),
        "confidence_gate": serialize_confidence_gate(confidence_gate),
        "optimizer_config": serialize_optimizer_config(optimizer_config),
        "optimizer_block_summaries": [asdict(block) for block in optimizer_block_summaries],
        "notes": (
            "The primary variant is configured ex ante. The sweep metrics are reported for "
            "comparison only and should not be used to claim a full-sample optimized allocation."
        ),
    }

    ensure_parent_directory(args.strategy_output)
    ensure_parent_directory(args.strategy_variants_output)
    ensure_parent_directory(args.benchmarks_output)
    ensure_parent_directory(args.metrics_output)
    ensure_parent_directory(args.sweep_metrics_output)
    ensure_parent_directory(args.metadata_output)
    ensure_parent_directory(args.notes_output)

    primary_strategy_frame.to_parquet(args.strategy_output, index=False)
    variant_returns_frame.to_parquet(args.strategy_variants_output, index=False)
    benchmark_frame.to_parquet(args.benchmarks_output, index=False)
    primary_metrics_frame.to_csv(args.metrics_output, index=False)
    sweep_metrics_frame.to_csv(args.sweep_metrics_output, index=False)
    Path(args.metadata_output).write_text(
        json.dumps(metadata_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.notes_output).write_text(
        build_notes_text(
            summary=summary,
            primary_metrics_frame=primary_metrics_frame,
            sweep_metrics_frame=sweep_metrics_frame,
            variant_specs=variant_specs,
            benchmark_relative_template_map=benchmark_relative_template_map,
            asset_bounds=asset_bounds,
            confidence_gate=confidence_gate,
            optimizer_config=optimizer_config,
            optimizer_block_summaries=optimizer_block_summaries,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_walkforward_backtest_pipeline(args)
    print("Walk-forward backtest completed.")
    print(f"Primary strategy returns: {summary.strategy_returns_path}")
    print(f"All variant returns: {summary.strategy_variant_returns_path}")
    print(f"Benchmark returns: {summary.benchmark_returns_path}")
    print(f"Primary metrics table: {summary.metrics_path}")
    print(f"Allocation sweep metrics: {summary.sweep_metrics_path}")
    print(f"Allocation metadata: {summary.metadata_path}")
    print(f"Notes: {summary.notes_path}")
    print(f"Backtest rows: {summary.row_count}")
    print(f"Sample window: {summary.start_date} to {summary.end_date}")
    print(f"Primary variant: {summary.primary_variant_name}")
    print(f"Configuration: {asdict(summary)}")


if __name__ == "__main__":
    main()
