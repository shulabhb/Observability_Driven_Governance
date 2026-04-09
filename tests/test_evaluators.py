"""Step 3: evaluators return structured dimension results."""

import pandas as pd

from governance_engine.evaluators import run_all_evaluators
from governance_engine.expectations import load_expectations, merged_bands_for_archetype


def _row(**kwargs) -> pd.Series:
    base = {
        "telemetry_archetype": "tabular_ml",
        "accuracy_pct": 90.0,
        "error_pct": 3.0,
        "drift_index": 0.05,
        "escalation_pct": 5.0,
        "expected_latency_p95_ms": 200.0,
        "observed_latency_p95_ms": 210.0,
        "output_consistency_score": 0.9,
        "audit_coverage_pct": 97.0,
        "decision_trace_completeness": 0.92,
        "minimum_required_audit_coverage": 0.95,
        "policy_violation_rate": 0.0005,
        "security_anomaly_count": 0,
        "compliance_incidents": 0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_evaluators_clean_row_low_severity() -> None:
    cfg = load_expectations()
    bands = merged_bands_for_archetype(cfg, "tabular_ml")
    results = run_all_evaluators(_row(), bands, "tabular_ml")
    assert len(results) == 4
    assert results[0].dimension == "reliability"
    assert all(r.severity == 0.0 for r in results if r.dimension != "narrative_assurance")
    nar = next(r for r in results if r.dimension == "narrative_assurance")
    assert nar.skipped


def test_evaluators_reliability_breach() -> None:
    cfg = load_expectations()
    bands = merged_bands_for_archetype(cfg, "tabular_ml")
    results = run_all_evaluators(
        _row(accuracy_pct=60.0, error_pct=15.0),
        bands,
        "tabular_ml",
    )
    rel = next(r for r in results if r.dimension == "reliability")
    assert rel.severity > 0
    assert rel.breaches
