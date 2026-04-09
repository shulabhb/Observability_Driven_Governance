"""Observability stress index (telemetry-derived)."""

import pandas as pd

from governance_engine.stress import compute_observability_stress, traffic_light


def test_traffic_light_mapping() -> None:
    assert traffic_light("low") == "green"
    assert traffic_light("medium") == "yellow"
    assert traffic_light("high") == "orange"
    assert traffic_light("critical") == "red"


def test_stress_increases_with_bad_telemetry() -> None:
    calm = pd.Series(
        {
            "drift_index": 0.02,
            "error_pct": 2.0,
            "escalation_pct": 3.0,
            "audit_coverage_pct": 99.0,
            "compliance_incidents": 0,
            "anomaly_frequency": 0,
            "policy_violation_rate": 0.0008,
            "security_anomaly_count": 0,
            "manual_review_burden": 0.03,
        }
    )
    hot = calm.copy()
    hot["drift_index"] = 0.2
    hot["error_pct"] = 9.0
    hot["escalation_pct"] = 28.0
    hot["audit_coverage_pct"] = 94.0
    hot["compliance_incidents"] = 2
    hot["anomaly_frequency"] = 8
    assert compute_observability_stress(hot) > compute_observability_stress(calm)
