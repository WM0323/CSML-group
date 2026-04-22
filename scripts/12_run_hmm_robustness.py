#!/usr/bin/env python3
"""Entry point for compact walk-forward HMM robustness checks."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.hmm_robustness import main


if __name__ == "__main__":
    main()
