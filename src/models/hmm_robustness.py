"""Compact robustness checks for the walk-forward HMM signal."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.hmm_common import (
    APPROVED_FEATURE_COLUMNS,
    fit_hmm_with_restarts,
    load_feature_table,
)
from src.models.hmm_walkforward import (
    build_filtered_posterior_block,
    build_posterior_weighted_state_order,
    build_predictive_signal_block,
    build_refit_schedule,
    validate_scaler,
)


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROBUSTNESS_SPECS = [
    {
        "check_name": "baseline",
        "n_components": 3,
        "min_training_days": 756,
        "refit_frequency": "monthly",
        "covariance_type": "diag",
    },
    {
        "check_name": "states_2",
        "n_components": 2,
        "min_training_days": 756,
        "refit_frequency": "monthly",
        "covariance_type": "diag",
    },
    {
        "check_name": "states_4",
        "n_components": 4,
        "min_training_days": 756,
        "refit_frequency": "monthly",
        "covariance_type": "diag",
    },
    {
        "check_name": "warmup_504",
        "n_components": 3,
        "min_training_days": 504,
        "refit_frequency": "monthly",
        "covariance_type": "diag",
    },
    {
        "check_name": "quarterly_refit",
        "n_components": 3,
        "min_training_days": 756,
        "refit_frequency": "quarterly",
        "covariance_type": "diag",
    },
    {
        "check_name": "tied_covariance",
        "n_components": 3,
        "min_training_days": 756,
        "refit_frequency": "monthly",
        "covariance_type": "tied",
    },
]
ROBUSTNESS_RANDOM_SEEDS = [7, 19, 31]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run compact walk-forward HMM robustness checks."
    )
    parser.add_argument(
        "--input",
        default="data/processed/model_features.parquet",
        help="Path to the approved modeling feature table parquet file.",
    )
    parser.add_argument(
        "--output",
        default="results/models/hmm_robustness_summary.csv",
        help="Output CSV path for the robustness summary.",
    )
    parser.add_argument(
        "--notes-output",
        default="docs/hmm_robustness_notes.md",
        help="Output Markdown path for concise robustness notes.",
    )
    return parser


def calculate_state_persistence(assignments: np.ndarray) -> float:
    if assignments.size < 2:
        return float("nan")
    return float(np.mean(assignments[1:] == assignments[:-1]))


def ensure_parent_directory(path_like: str | Path) -> None:
    Path(path_like).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def run_single_spec(
    feature_table: pd.DataFrame,
    *,
    check_name: str,
    n_components: int,
    min_training_days: int,
    refit_frequency: str,
    covariance_type: str,
) -> dict[str, object]:
    refit_schedule = build_refit_schedule(feature_table, min_training_days, refit_frequency)

    ordered_states_blocks: list[np.ndarray] = []
    max_probability_blocks: list[np.ndarray] = []

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

        model, _, _ = fit_hmm_with_restarts(
            standardized_values=standardized_training,
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=200,
            tol=1e-4,
            random_seeds=ROBUSTNESS_RANDOM_SEEDS,
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
        ordered_raw_states, _ = build_posterior_weighted_state_order(
            training_frame=training_frame,
            training_posteriors=training_filtered_posteriors,
        )

        signal_emission_log_likelihoods = np.asarray(
            model._compute_log_likelihood(standardized_signal),
            dtype=float,
        )
        predictive_probabilities_raw, _ = build_predictive_signal_block(
            emission_log_likelihoods=signal_emission_log_likelihoods,
            last_training_filtered=np.asarray(training_filtered_posteriors[-1], dtype=float),
            transition_matrix=np.asarray(model.transmat_, dtype=float),
        )
        predictive_probabilities_ordered = predictive_probabilities_raw[:, ordered_raw_states]
        predictive_probabilities_ordered = (
            predictive_probabilities_ordered
            / predictive_probabilities_ordered.sum(axis=1, keepdims=True)
        )

        ordered_states_blocks.append(
            np.argmax(predictive_probabilities_ordered, axis=1).astype(int)
        )
        max_probability_blocks.append(
            np.max(predictive_probabilities_ordered, axis=1).astype(float)
        )

    ordered_states = np.concatenate(ordered_states_blocks)
    max_probabilities = np.concatenate(max_probability_blocks)
    start_date = feature_table["date"].iloc[min_training_days].date().isoformat()
    end_date = feature_table["date"].iloc[-1].date().isoformat()

    return {
        "check_name": check_name,
        "n_components": int(n_components),
        "min_training_days": int(min_training_days),
        "refit_frequency": refit_frequency,
        "covariance_type": covariance_type,
        "row_count": int(ordered_states.size),
        "start_date": start_date,
        "end_date": end_date,
        "refit_count": int(len(refit_schedule)),
        "state_persistence": calculate_state_persistence(ordered_states),
        "avg_max_predictive_probability": float(max_probabilities.mean()),
        "pct_max_probability_below_0_60": float(np.mean(max_probabilities < 0.60)),
    }


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


def build_notes(summary_frame: pd.DataFrame) -> str:
    display_frame = summary_frame.copy()
    for column in ["state_persistence", "avg_max_predictive_probability", "pct_max_probability_below_0_60"]:
        display_frame[column] = display_frame[column].map(lambda value: f"{100.0 * float(value):.2f}%")

    return "\n".join(
        [
            "# HMM Robustness Notes",
            "",
            "- This is a compact one-factor-at-a-time robustness grid around the active walk-forward HMM specification.",
            "- The summary focuses on signal stability, not on adding new trading rules.",
            "- `state_persistence` is the same-as-previous-day rate of the ordered predictive state label sequence.",
            "- `avg_max_predictive_probability` measures how concentrated the predictive distribution is on average.",
            "",
            build_markdown_table(display_frame),
        ]
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    feature_table = load_feature_table(args.input)

    rows = [
        run_single_spec(feature_table, **spec)
        for spec in ROBUSTNESS_SPECS
    ]
    summary_frame = pd.DataFrame(rows)

    ensure_parent_directory(args.output)
    summary_frame.to_csv(args.output, index=False)
    ensure_parent_directory(args.notes_output)
    Path(args.notes_output).write_text(build_notes(summary_frame) + "\n", encoding="utf-8")

    print(Path(args.output))
    print(Path(args.notes_output))


if __name__ == "__main__":
    main()
