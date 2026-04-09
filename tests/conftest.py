import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENRICHED = ROOT / "enriched_monitoring_report.xlsx"


@pytest.fixture
def enriched_path() -> Path:
    if not ENRICHED.is_file():
        pytest.skip(f"Missing {ENRICHED}; run enrich_monitoring.py first")
    return ENRICHED
