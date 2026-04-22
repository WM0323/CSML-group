"""Performance metric helpers for daily return series."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def clean_return_series(returns: pd.Series) -> pd.Series:
    """Return a finite float series with missing values removed."""
    cleaned = pd.Series(returns, copy=False).astype(float)
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        raise ValueError("Return series is empty after dropping missing values.")
    return cleaned


def build_nav_series(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Convert simple returns into a cumulative net asset value series."""
    cleaned = clean_return_series(returns)
    return float(start_value) * (1.0 + cleaned).cumprod()


def calculate_performance_metrics(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Calculate first-pass performance metrics for a daily strategy."""
    cleaned = clean_return_series(returns)
    nav = build_nav_series(cleaned)

    observations = int(cleaned.size)
    terminal_value = float(nav.iloc[-1])
    cumulative_return = terminal_value - 1.0

    annualized_return = np.nan
    if terminal_value > 0.0:
        annualized_return = terminal_value ** (periods_per_year / observations) - 1.0

    annualized_volatility = float(cleaned.std(ddof=0)) * math.sqrt(periods_per_year)
    sharpe_ratio = np.nan
    if not np.isclose(annualized_volatility, 0.0):
        sharpe_ratio = (
            float(cleaned.mean()) / float(cleaned.std(ddof=0)) * math.sqrt(periods_per_year)
        )

    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "observations": observations,
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
    }
