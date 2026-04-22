#!/usr/bin/env python3
"""Assemble the final report markdown deliverable from the updated project draft."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from shutil import copy2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_PATH = PROJECT_ROOT / "docs" / "report_draft.md"
OUTPUT_DIR = PROJECT_ROOT / "deliverables" / "final_report"
OUTPUT_PATH = OUTPUT_DIR / "ELEN4904_final_report.md"
ASSETS_DIR = OUTPUT_DIR / "assets"
FIGURE_NAMES = [
    "figure_01_regime_persistence_comparison.png",
    "figure_02_shared_window_nav_comparison.png",
    "figure_03_walkforward_state_probabilities.png",
    "figure_04_shared_window_strategy_metrics.png",
]


def load_report_body() -> str:
    text = DRAFT_PATH.read_text(encoding="utf-8").strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def build_document() -> str:
    today = date.today().isoformat()
    body = load_report_body()
    return f"""# Cross-Asset Regime Detection for Dynamic Portfolio Allocation

**Course:** ELEN4904: Statistical Learning with Applications in Quant Trading  
**Team:** Weiman Sun, Tianyi Chen, Ruomeng Ma, Xintong Li  
**Assembly Date:** {today}

## Executive Summary

This report studies cross-asset regime detection for dynamic portfolio allocation using a reproducible daily pipeline. The active model is a walk-forward three-state Hidden Markov Model (HMM) that uses only past data and forward-recursion predictive signals. The baseline model is a fair walk-forward PCA plus K-means pipeline on the same approved feature table.

The main findings are:

- the walk-forward HMM remains more stable and more interpretable than the clustering baseline,
- on the full shared sample, the walk-forward HMM strategy materially outperforms the fair walk-forward K-means strategy,
- that HMM trading edge is not uniform across coarse subperiods,
- neither dynamic strategy beats the simple static benchmark portfolios on that same window.

{body}
"""


def copy_report_assets() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = PROJECT_ROOT / "results" / "figures"
    for figure_name in FIGURE_NAMES:
        source_path = source_dir / figure_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing required figure: {source_path}")
        copy2(source_path, ASSETS_DIR / figure_name)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_report_assets()
    OUTPUT_PATH.write_text(build_document() + "\n", encoding="utf-8")
    print(OUTPUT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
