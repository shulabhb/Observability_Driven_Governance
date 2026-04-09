"""Step 4–5: full engine produces decision JSON shape."""

from pathlib import Path

import pandas as pd

from governance_engine.data_bridge import load_enriched_post
from governance_engine.engine import decision_to_jsonable, run_governance_for_row
from governance_engine.expectations import load_expectations


def test_engine_on_each_live_row(enriched_path: Path) -> None:
    df = load_enriched_post(enriched_path)
    cfg = load_expectations()
    for _, row in df.iterrows():
        d = run_governance_for_row(row, cfg)
        assert d.model_id
        assert 0 <= d.risk_score <= 100
        assert 0 <= d.breach_component_score <= 100
        assert 0 <= d.observability_stress_score <= 100
        assert d.risk_level in ("low", "medium", "high", "critical")
        assert d.risk_traffic_light in ("green", "yellow", "orange", "red")
        assert isinstance(d.issue_categories, list)
        assert d.recommended_action_id
        assert d.recommended_action_text
        assert isinstance(d.human_review_required, bool)
        assert d.human_review_priority in ("none", "P1", "P2", "P3")
        assert len(d.dimension_results) == 4
        j = decision_to_jsonable(d)
        assert "risk_score" in j
