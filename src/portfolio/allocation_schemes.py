"""Allocation builders for walk-forward HMM portfolio backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.portfolio.backtest_common import (
    BENCHMARK_ALLOCATIONS,
    REGIME_ALLOCATIONS,
    RETURN_COLUMN_BY_ASSET,
    STATE_LABELS,
    TRADED_ASSETS,
    WEIGHT_COLUMN_BY_ASSET,
)


ASSET_SEQUENCE = tuple(TRADED_ASSETS.keys())
STATE_SEQUENCE = tuple(sorted(STATE_LABELS))
PROBABILITY_COLUMN_BY_STATE = {
    state: f"state_{state}_probability" for state in STATE_SEQUENCE
}


@dataclass(frozen=True)
class AssetBounds:
    floors: dict[str, float]
    caps: dict[str, float]

    def validate(self) -> None:
        missing_floor_assets = [asset for asset in ASSET_SEQUENCE if asset not in self.floors]
        missing_cap_assets = [asset for asset in ASSET_SEQUENCE if asset not in self.caps]
        if missing_floor_assets:
            raise ValueError(
                "Asset bounds are missing floor values for: "
                + ", ".join(missing_floor_assets)
            )
        if missing_cap_assets:
            raise ValueError(
                "Asset bounds are missing cap values for: " + ", ".join(missing_cap_assets)
            )

        extra_floor_assets = [asset for asset in self.floors if asset not in ASSET_SEQUENCE]
        extra_cap_assets = [asset for asset in self.caps if asset not in ASSET_SEQUENCE]
        if extra_floor_assets:
            raise ValueError(
                "Asset bounds include unsupported floor assets: "
                + ", ".join(extra_floor_assets)
            )
        if extra_cap_assets:
            raise ValueError(
                "Asset bounds include unsupported cap assets: " + ", ".join(extra_cap_assets)
            )

        for asset in ASSET_SEQUENCE:
            floor = float(self.floors[asset])
            cap = float(self.caps[asset])
            if floor < 0.0:
                raise ValueError(f"Asset floor for {asset} cannot be negative.")
            if cap <= 0.0:
                raise ValueError(f"Asset cap for {asset} must be positive.")
            if floor > cap:
                raise ValueError(f"Asset floor for {asset} exceeds its cap.")

        lower_sum = float(sum(float(self.floors[asset]) for asset in ASSET_SEQUENCE))
        upper_sum = float(sum(float(self.caps[asset]) for asset in ASSET_SEQUENCE))
        if lower_sum > 1.0 + 1e-9:
            raise ValueError("Asset bounds floors sum to more than 1.0.")
        if upper_sum < 1.0 - 1e-9:
            raise ValueError("Asset bounds caps sum to less than 1.0.")

    def lower_vector(self) -> np.ndarray:
        self.validate()
        return np.asarray([float(self.floors[asset]) for asset in ASSET_SEQUENCE], dtype=float)

    def upper_vector(self) -> np.ndarray:
        self.validate()
        return np.asarray([float(self.caps[asset]) for asset in ASSET_SEQUENCE], dtype=float)

    def as_rows(self) -> list[dict[str, float]]:
        return [
            {
                "asset": asset,
                "floor": float(self.floors[asset]),
                "cap": float(self.caps[asset]),
            }
            for asset in ASSET_SEQUENCE
        ]


@dataclass(frozen=True)
class BenchmarkRelativeTiltConfig:
    base_allocation: dict[str, float] = field(
        default_factory=lambda: dict(BENCHMARK_ALLOCATIONS["fixed_60_40_stock_bond"])
    )
    risk_off_delta: dict[str, float] = field(
        default_factory=lambda: {
            "SPY": -0.15,
            "TLT": 0.05,
            "IEF": 0.00,
            "GLD": 0.10,
        }
    )
    risk_on_delta: dict[str, float] = field(
        default_factory=lambda: {
            "SPY": 0.15,
            "TLT": -0.10,
            "IEF": -0.05,
            "GLD": 0.00,
        }
    )

    def validate(self) -> None:
        validate_allocation_dict(self.base_allocation, name="benchmark_relative_base_allocation")
        validate_delta_dict(self.risk_off_delta, name="benchmark_relative_risk_off_delta")
        validate_delta_dict(self.risk_on_delta, name="benchmark_relative_risk_on_delta")


@dataclass(frozen=True)
class ConfidenceGateConfig:
    threshold: float = 0.70
    scale: float = 0.30

    def validate(self) -> None:
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("Confidence gate threshold must lie between 0 and 1.")
        if float(self.scale) <= 0.0:
            raise ValueError("Confidence gate scale must be positive.")


@dataclass(frozen=True)
class OptimizerConfig:
    method: str = "mean_variance"
    covariance_shrinkage: float = 0.35
    ridge_penalty: float = 1e-4
    mean_shrinkage: float = 0.50
    risk_aversion: float = 3.0
    template_blend: float = 0.50
    min_effective_observations: float = 60.0
    max_iterations: int = 250
    tolerance: float = 1e-8

    def validate(self) -> None:
        if self.method not in {"min_variance", "mean_variance"}:
            raise ValueError("Optimizer method must be 'min_variance' or 'mean_variance'.")
        if not 0.0 <= float(self.covariance_shrinkage) <= 1.0:
            raise ValueError("Covariance shrinkage must lie between 0 and 1.")
        if float(self.ridge_penalty) < 0.0:
            raise ValueError("Ridge penalty cannot be negative.")
        if not 0.0 <= float(self.mean_shrinkage) <= 1.0:
            raise ValueError("Mean shrinkage must lie between 0 and 1.")
        if float(self.risk_aversion) <= 0.0:
            raise ValueError("Risk aversion must be positive.")
        if not 0.0 <= float(self.template_blend) <= 1.0:
            raise ValueError("Template blend must lie between 0 and 1.")
        if float(self.min_effective_observations) <= 0.0:
            raise ValueError("Minimum effective observations must be positive.")
        if int(self.max_iterations) <= 0:
            raise ValueError("Max iterations must be positive.")
        if float(self.tolerance) <= 0.0:
            raise ValueError("Optimizer tolerance must be positive.")


@dataclass(frozen=True)
class RegimeStatisticEstimate:
    effective_observations: float
    mean_returns: dict[str, float]
    covariance: list[list[float]]


@dataclass(frozen=True)
class OptimizedTemplateBlockSummary:
    block_index: int
    refit_date: str
    training_end_date: str
    block_end_date: str
    historical_rows: int
    optimized_states: list[int]
    fallback_states: list[int]
    effective_observations: dict[str, float]
    regime_templates: dict[str, dict[str, float]]
    optimizer_method: str


DEFAULT_ASSET_BOUNDS = AssetBounds(
    floors={"SPY": 0.20, "TLT": 0.05, "IEF": 0.05, "GLD": 0.00},
    caps={"SPY": 0.85, "TLT": 0.45, "IEF": 0.35, "GLD": 0.20},
)
DEFAULT_TILT_CONFIG = BenchmarkRelativeTiltConfig()
DEFAULT_CONFIDENCE_GATE = ConfidenceGateConfig()
DEFAULT_OPTIMIZER_CONFIG = OptimizerConfig()


def validate_allocation_dict(allocation: dict[str, float], name: str) -> None:
    missing_assets = [asset for asset in ASSET_SEQUENCE if asset not in allocation]
    extra_assets = [asset for asset in allocation if asset not in ASSET_SEQUENCE]
    if missing_assets:
        raise ValueError(f"{name} is missing assets: " + ", ".join(missing_assets))
    if extra_assets:
        raise ValueError(f"{name} includes unsupported assets: " + ", ".join(extra_assets))

    total_weight = float(sum(float(allocation[asset]) for asset in ASSET_SEQUENCE))
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"{name} sums to {total_weight:.6f}, not 1.0.")

    negative_assets = [asset for asset, weight in allocation.items() if float(weight) < 0.0]
    if negative_assets:
        raise ValueError(f"{name} has negative weights for: " + ", ".join(negative_assets))


def validate_delta_dict(delta: dict[str, float], name: str) -> None:
    missing_assets = [asset for asset in ASSET_SEQUENCE if asset not in delta]
    extra_assets = [asset for asset in delta if asset not in ASSET_SEQUENCE]
    if missing_assets:
        raise ValueError(f"{name} is missing assets: " + ", ".join(missing_assets))
    if extra_assets:
        raise ValueError(f"{name} includes unsupported assets: " + ", ".join(extra_assets))
    total_delta = float(sum(float(delta[asset]) for asset in ASSET_SEQUENCE))
    if not np.isclose(total_delta, 0.0):
        raise ValueError(f"{name} sums to {total_delta:.6f}, not 0.0.")


def allocation_to_vector(allocation: dict[str, float]) -> np.ndarray:
    validate_allocation_dict(allocation, name="allocation")
    return np.asarray([float(allocation[asset]) for asset in ASSET_SEQUENCE], dtype=float)


def asset_dict_to_vector(values: dict[str, float], name: str) -> np.ndarray:
    missing_assets = [asset for asset in ASSET_SEQUENCE if asset not in values]
    extra_assets = [asset for asset in values if asset not in ASSET_SEQUENCE]
    if missing_assets:
        raise ValueError(f"{name} is missing assets: " + ", ".join(missing_assets))
    if extra_assets:
        raise ValueError(f"{name} includes unsupported assets: " + ", ".join(extra_assets))
    return np.asarray([float(values[asset]) for asset in ASSET_SEQUENCE], dtype=float)


def vector_to_allocation(weight_vector: np.ndarray) -> dict[str, float]:
    array = np.asarray(weight_vector, dtype=float)
    if array.shape != (len(ASSET_SEQUENCE),):
        raise ValueError("Weight vector has the wrong shape.")
    return {
        asset: float(array[index]) for index, asset in enumerate(ASSET_SEQUENCE)
    }


def build_weight_frame(
    weight_matrix: np.ndarray,
    extra_columns: dict[str, np.ndarray | list[object] | pd.Series] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        weight_matrix,
        columns=[WEIGHT_COLUMN_BY_ASSET[asset] for asset in ASSET_SEQUENCE],
    )
    for column_name, values in (extra_columns or {}).items():
        frame[column_name] = values
    return frame


def project_to_bounded_simplex(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_sum: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> np.ndarray:
    candidate = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if candidate.shape != lower.shape or candidate.shape != upper.shape:
        raise ValueError("Bounded simplex projection shapes do not match.")
    if float(lower.sum()) > target_sum + tolerance:
        raise ValueError("Lower bounds make the simplex infeasible.")
    if float(upper.sum()) < target_sum - tolerance:
        raise ValueError("Upper bounds make the simplex infeasible.")

    left = float(np.min(candidate - upper))
    right = float(np.max(candidate - lower))
    projected = np.clip(candidate, lower, upper)

    for _ in range(max_iterations):
        midpoint = 0.5 * (left + right)
        projected = np.clip(candidate - midpoint, lower, upper)
        total = float(projected.sum())
        if abs(total - target_sum) <= tolerance:
            break
        if total > target_sum:
            left = midpoint
        else:
            right = midpoint

    projected = np.clip(projected, lower, upper)
    total = float(projected.sum())
    if not np.isclose(total, target_sum, atol=1e-8):
        # Fall back to a final numerically stable correction along the free coordinates.
        residual = target_sum - total
        if abs(residual) > 1e-8:
            if residual > 0.0:
                slack = upper - projected
            else:
                slack = projected - lower
            free_mask = slack > 1e-12
            if free_mask.any():
                projected[free_mask] = (
                    projected[free_mask]
                    + residual * slack[free_mask] / float(slack[free_mask].sum())
                )
        projected = np.clip(projected, lower, upper)

    normalized_total = float(projected.sum())
    if not np.isclose(normalized_total, target_sum, atol=1e-6):
        raise ValueError("Unable to project weights onto the bounded simplex.")
    return projected


def enforce_asset_bounds(weight_matrix: np.ndarray, asset_bounds: AssetBounds) -> np.ndarray:
    lower = asset_bounds.lower_vector()
    upper = asset_bounds.upper_vector()
    bounded_rows = [
        project_to_bounded_simplex(weight_row, lower=lower, upper=upper)
        for weight_row in np.asarray(weight_matrix, dtype=float)
    ]
    return np.asarray(bounded_rows, dtype=float)


def build_allocation_matrix(
    allocation_map: dict[int, dict[str, float]],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for state in STATE_SEQUENCE:
        if state not in allocation_map:
            raise ValueError(f"Allocation map is missing state {state}.")
        rows.append(allocation_to_vector(allocation_map[state]))
    return np.vstack(rows)


def compute_confidence_scale(
    probabilities: np.ndarray,
    confidence_gate: ConfidenceGateConfig,
) -> np.ndarray:
    confidence_gate.validate()
    max_probabilities = np.asarray(probabilities, dtype=float).max(axis=1)
    return np.clip(
        (max_probabilities - float(confidence_gate.threshold)) / float(confidence_gate.scale),
        0.0,
        1.0,
    )


def build_benchmark_relative_template_map(
    tilt_config: BenchmarkRelativeTiltConfig = DEFAULT_TILT_CONFIG,
    asset_bounds: AssetBounds = DEFAULT_ASSET_BOUNDS,
) -> dict[int, dict[str, float]]:
    tilt_config.validate()
    base_vector = allocation_to_vector(tilt_config.base_allocation)
    risk_off = project_to_bounded_simplex(
        base_vector + allocation_to_vector_from_delta(tilt_config.risk_off_delta),
        lower=asset_bounds.lower_vector(),
        upper=asset_bounds.upper_vector(),
    )
    transition = project_to_bounded_simplex(
        base_vector,
        lower=asset_bounds.lower_vector(),
        upper=asset_bounds.upper_vector(),
    )
    risk_on = project_to_bounded_simplex(
        base_vector + allocation_to_vector_from_delta(tilt_config.risk_on_delta),
        lower=asset_bounds.lower_vector(),
        upper=asset_bounds.upper_vector(),
    )
    return {
        0: vector_to_allocation(risk_off),
        1: vector_to_allocation(transition),
        2: vector_to_allocation(risk_on),
    }


def allocation_to_vector_from_delta(delta: dict[str, float]) -> np.ndarray:
    validate_delta_dict(delta, name="allocation_delta")
    return np.asarray([float(delta[asset]) for asset in ASSET_SEQUENCE], dtype=float)


def build_hard_label_weight_frame(
    signal_frame: pd.DataFrame,
    allocation_map: dict[int, dict[str, float]] = REGIME_ALLOCATIONS,
) -> pd.DataFrame:
    signal_states = signal_frame["signal_state"].astype(int).to_numpy()
    allocation_matrix = build_allocation_matrix(allocation_map)
    weight_matrix = allocation_matrix[signal_states]
    return build_weight_frame(weight_matrix)


def build_probability_weighted_weight_frame(
    signal_frame: pd.DataFrame,
    allocation_map: dict[int, dict[str, float]],
    asset_bounds: AssetBounds | None = None,
) -> pd.DataFrame:
    probability_matrix = signal_frame[
        [PROBABILITY_COLUMN_BY_STATE[state] for state in STATE_SEQUENCE]
    ].to_numpy(dtype=float)
    allocation_matrix = build_allocation_matrix(allocation_map)
    weight_matrix = probability_matrix @ allocation_matrix
    if asset_bounds is not None:
        weight_matrix = enforce_asset_bounds(weight_matrix, asset_bounds)
    return build_weight_frame(weight_matrix)


def build_tilt_weight_frame(
    signal_frame: pd.DataFrame,
    tilt_config: BenchmarkRelativeTiltConfig = DEFAULT_TILT_CONFIG,
    asset_bounds: AssetBounds = DEFAULT_ASSET_BOUNDS,
    confidence_gate: ConfidenceGateConfig | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    template_map = build_benchmark_relative_template_map(
        tilt_config=tilt_config,
        asset_bounds=asset_bounds,
    )
    probability_matrix = signal_frame[
        [PROBABILITY_COLUMN_BY_STATE[state] for state in STATE_SEQUENCE]
    ].to_numpy(dtype=float)
    template_matrix = build_allocation_matrix(template_map)
    weight_matrix = probability_matrix @ template_matrix

    extra_columns: dict[str, np.ndarray | list[object] | pd.Series] = {}
    if confidence_gate is not None:
        base_vector = allocation_to_vector(tilt_config.base_allocation)
        gate_scale = compute_confidence_scale(probability_matrix, confidence_gate)
        weight_matrix = base_vector + gate_scale[:, np.newaxis] * (weight_matrix - base_vector)
        extra_columns["signal_confidence"] = probability_matrix.max(axis=1)
        extra_columns["confidence_gate_scale"] = gate_scale
    else:
        extra_columns["signal_confidence"] = probability_matrix.max(axis=1)
        extra_columns["confidence_gate_scale"] = np.ones(len(signal_frame), dtype=float)

    weight_matrix = enforce_asset_bounds(weight_matrix, asset_bounds)
    return build_weight_frame(weight_matrix, extra_columns=extra_columns), template_map


def estimate_regime_statistics(
    historical_frame: pd.DataFrame,
    state: int,
    optimizer_config: OptimizerConfig,
) -> RegimeStatisticEstimate | None:
    optimizer_config.validate()
    if historical_frame.empty:
        return None

    return_columns = [RETURN_COLUMN_BY_ASSET[asset] for asset in ASSET_SEQUENCE]
    returns_matrix = historical_frame[return_columns].to_numpy(dtype=float)
    raw_weights = historical_frame[PROBABILITY_COLUMN_BY_STATE[state]].to_numpy(dtype=float)
    total_weight = float(raw_weights.sum())
    if total_weight <= 0.0:
        return None

    normalized_weights = raw_weights / total_weight
    effective_observations = float(1.0 / np.sum(np.square(normalized_weights)))
    if effective_observations < float(optimizer_config.min_effective_observations):
        return None

    mean_vector = normalized_weights @ returns_matrix
    centered = returns_matrix - mean_vector
    covariance = (centered * normalized_weights[:, np.newaxis]).T @ centered

    finite_sample_denom = 1.0 - float(np.sum(np.square(normalized_weights)))
    if finite_sample_denom > 1e-8:
        covariance = covariance / finite_sample_denom

    covariance = 0.5 * (covariance + covariance.T)
    return RegimeStatisticEstimate(
        effective_observations=effective_observations,
        mean_returns=vector_to_allocation(mean_vector),
        covariance=covariance.tolist(),
    )


def regularize_covariance(
    covariance: np.ndarray,
    optimizer_config: OptimizerConfig,
) -> np.ndarray:
    diagonal = np.diag(np.diag(covariance))
    regularized = (
        (1.0 - float(optimizer_config.covariance_shrinkage)) * covariance
        + float(optimizer_config.covariance_shrinkage) * diagonal
    )
    regularized = regularized + float(optimizer_config.ridge_penalty) * np.eye(
        regularized.shape[0]
    )
    return 0.5 * (regularized + regularized.T)


def solve_long_only_portfolio(
    mean_vector: np.ndarray,
    covariance: np.ndarray,
    asset_bounds: AssetBounds,
    optimizer_config: OptimizerConfig,
    initial_weights: np.ndarray,
) -> np.ndarray:
    optimizer_config.validate()
    lower = asset_bounds.lower_vector()
    upper = asset_bounds.upper_vector()
    covariance = regularize_covariance(np.asarray(covariance, dtype=float), optimizer_config)
    mean_vector = np.asarray(mean_vector, dtype=float)

    if optimizer_config.method == "min_variance":
        effective_mean = np.zeros_like(mean_vector)
    else:
        effective_mean = (1.0 - float(optimizer_config.mean_shrinkage)) * mean_vector

    lipschitz = float(np.linalg.norm(covariance, ord=2))
    step_size = 1.0 / max(lipschitz, 1e-8)
    weights = project_to_bounded_simplex(initial_weights, lower=lower, upper=upper)

    for _ in range(int(optimizer_config.max_iterations)):
        gradient = covariance @ weights - float(optimizer_config.risk_aversion) * effective_mean
        updated = project_to_bounded_simplex(
            weights - step_size * gradient,
            lower=lower,
            upper=upper,
        )
        if float(np.linalg.norm(updated - weights, ord=1)) <= float(optimizer_config.tolerance):
            weights = updated
            break
        weights = updated

    return project_to_bounded_simplex(weights, lower=lower, upper=upper)


def build_blockwise_optimized_weight_frame(
    signal_frame: pd.DataFrame,
    refit_blocks: list[dict[str, object]],
    fallback_template_map: dict[int, dict[str, float]],
    asset_bounds: AssetBounds = DEFAULT_ASSET_BOUNDS,
    optimizer_config: OptimizerConfig = DEFAULT_OPTIMIZER_CONFIG,
) -> tuple[pd.DataFrame, list[OptimizedTemplateBlockSummary]]:
    optimizer_config.validate()
    fallback_matrix = build_allocation_matrix(fallback_template_map)

    weight_rows: list[np.ndarray] = []
    block_indices: list[int] = []
    optimized_flags: list[bool] = []
    block_summaries: list[OptimizedTemplateBlockSummary] = []

    for block_index, block in enumerate(refit_blocks):
        refit_date = pd.Timestamp(block["refit_date"]).normalize()
        training_end_date = pd.Timestamp(block["training_end_date"]).normalize()
        block_end_date = pd.Timestamp(block["block_end_date"]).normalize()

        block_mask = (signal_frame["date"] >= refit_date) & (signal_frame["date"] <= block_end_date)
        block_frame = signal_frame.loc[block_mask].copy()
        if block_frame.empty:
            continue

        historical_frame = signal_frame.loc[
            signal_frame["date"] <= training_end_date
        ].copy()

        template_rows: list[np.ndarray] = []
        optimized_states: list[int] = []
        fallback_states: list[int] = []
        effective_observations: dict[str, float] = {}
        template_map: dict[int, dict[str, float]] = {}

        for state_position, state in enumerate(STATE_SEQUENCE):
            estimate = estimate_regime_statistics(
                historical_frame=historical_frame,
                state=state,
                optimizer_config=optimizer_config,
            )
            fallback_weights = fallback_matrix[state_position]
            if estimate is None:
                template_vector = fallback_weights
                fallback_states.append(state)
                effective_observations[str(state)] = 0.0
            else:
                try:
                    optimized_vector = solve_long_only_portfolio(
                        mean_vector=asset_dict_to_vector(
                            estimate.mean_returns,
                            name="regime_mean_returns",
                        ),
                        covariance=np.asarray(estimate.covariance, dtype=float),
                        asset_bounds=asset_bounds,
                        optimizer_config=optimizer_config,
                        initial_weights=fallback_weights,
                    )
                    template_vector = project_to_bounded_simplex(
                        float(optimizer_config.template_blend) * optimized_vector
                        + (1.0 - float(optimizer_config.template_blend)) * fallback_weights,
                        lower=asset_bounds.lower_vector(),
                        upper=asset_bounds.upper_vector(),
                    )
                    optimized_states.append(state)
                    effective_observations[str(state)] = float(
                        estimate.effective_observations
                    )
                except (np.linalg.LinAlgError, ValueError):
                    template_vector = fallback_weights
                    fallback_states.append(state)
                    effective_observations[str(state)] = float(
                        estimate.effective_observations
                    )

            template_rows.append(template_vector)
            template_map[state] = vector_to_allocation(template_vector)

        template_matrix = np.vstack(template_rows)
        probability_matrix = block_frame[
            [PROBABILITY_COLUMN_BY_STATE[state] for state in STATE_SEQUENCE]
        ].to_numpy(dtype=float)
        block_weights = probability_matrix @ template_matrix
        block_weights = enforce_asset_bounds(block_weights, asset_bounds)

        weight_rows.extend(block_weights)
        block_indices.extend([block_index] * len(block_frame))
        optimized_flags.extend([len(optimized_states) > 0] * len(block_frame))
        block_summaries.append(
            OptimizedTemplateBlockSummary(
                block_index=block_index,
                refit_date=refit_date.date().isoformat(),
                training_end_date=training_end_date.date().isoformat(),
                block_end_date=block_end_date.date().isoformat(),
                historical_rows=int(len(historical_frame)),
                optimized_states=optimized_states,
                fallback_states=fallback_states,
                effective_observations=effective_observations,
                regime_templates={str(state): template_map[state] for state in STATE_SEQUENCE},
                optimizer_method=optimizer_config.method,
            )
        )

    if not weight_rows:
        raise ValueError("Blockwise optimizer produced no allocation rows.")

    weight_matrix = np.asarray(weight_rows, dtype=float)
    if len(weight_matrix) != len(signal_frame):
        raise ValueError("Blockwise optimizer output rows do not align with the signal frame.")

    weight_frame = build_weight_frame(
        weight_matrix,
        extra_columns={
            "refit_block_index": np.asarray(block_indices, dtype=int),
            "optimizer_applied": np.asarray(optimized_flags, dtype=bool),
            "signal_confidence": signal_frame[
                [PROBABILITY_COLUMN_BY_STATE[state] for state in STATE_SEQUENCE]
            ]
            .to_numpy(dtype=float)
            .max(axis=1),
        },
    )
    return weight_frame, block_summaries


def serialize_asset_bounds(asset_bounds: AssetBounds) -> dict[str, object]:
    return {"floors": dict(asset_bounds.floors), "caps": dict(asset_bounds.caps)}


def serialize_tilt_config(tilt_config: BenchmarkRelativeTiltConfig) -> dict[str, object]:
    return asdict(tilt_config)


def serialize_confidence_gate(
    confidence_gate: ConfidenceGateConfig | None,
) -> dict[str, float] | None:
    if confidence_gate is None:
        return None
    return asdict(confidence_gate)


def serialize_optimizer_config(
    optimizer_config: OptimizerConfig | None,
) -> dict[str, object] | None:
    if optimizer_config is None:
        return None
    return asdict(optimizer_config)
