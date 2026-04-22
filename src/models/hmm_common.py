"""Shared utilities for the project HMM pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


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

DEFAULT_RANDOM_SEEDS = [7, 11, 19, 23, 31]


@dataclass(frozen=True)
class FitAttempt:
    seed: int
    log_likelihood: float
    converged: bool
    iterations: int
    states_observed: int


def parse_random_seeds(raw_value: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one random seed is required.")
    return seeds


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
        raise ValueError("Feature table contains missing values in approved HMM inputs.")

    if len(frame) < 30:
        raise ValueError("Feature table is too short for a stable 3-state HMM pipeline.")

    return frame


def fit_hmm_with_restarts(
    standardized_values: np.ndarray,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    tol: float,
    random_seeds: list[int],
) -> tuple[GaussianHMM, FitAttempt, list[FitAttempt]]:
    attempts: list[FitAttempt] = []
    best_model: GaussianHMM | None = None
    best_attempt: FitAttempt | None = None
    best_priority: tuple[int, int, float] | None = None

    for seed in random_seeds:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            min_covar=1e-4,
            random_state=seed,
            implementation="log",
        )
        model.fit(standardized_values)
        posterior_weights = model.predict_proba(standardized_values)
        state_masses = np.asarray(posterior_weights.sum(axis=0), dtype=float)
        log_likelihood = float(model.score(standardized_values))
        attempt = FitAttempt(
            seed=seed,
            log_likelihood=log_likelihood,
            converged=bool(model.monitor_.converged),
            iterations=int(model.monitor_.iter),
            states_observed=int(np.count_nonzero(state_masses > 1e-6)),
        )
        attempts.append(attempt)

        priority = (
            1 if attempt.converged else 0,
            attempt.states_observed,
            attempt.log_likelihood,
        )
        if best_priority is None or priority > best_priority:
            best_model = model
            best_attempt = attempt
            best_priority = priority

    if best_model is None or best_attempt is None:
        raise RuntimeError("No HMM fit attempts completed.")

    return best_model, best_attempt, attempts


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
