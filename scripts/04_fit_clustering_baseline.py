"""Entry point for Phase C Step 10 clustering baseline modeling."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.clustering_baseline import main


if __name__ == "__main__":
    main()
