"""Archetype-specific dashboard metric lists."""

import pandas as pd

from governance_engine.telemetry_display import (
    format_snapshot_value,
    humanize_column_name,
    narrative_metric_keys_for_arch,
    primary_metrics_for_row,
    reliability_metric_keys_for_arch,
)


def test_tabular_primary_omits_drift_in_reliability_tab() -> None:
    keys = [c for c, _ in reliability_metric_keys_for_arch("tabular_ml")]
    assert "drift_index" not in keys


def test_time_series_reliability_includes_drift() -> None:
    keys = [c for c, _ in reliability_metric_keys_for_arch("time_series")]
    assert "drift_index" in keys


def test_primary_metrics_nonempty_for_synthetic_row() -> None:
    row = pd.Series(
        {
            "telemetry_archetype": "tabular_ml",
            "accuracy_pct": 90.0,
            "error_pct": 3.0,
            "false_positive_rate": 1.2,
            "false_negative_rate": 1.5,
            "escalation_pct": 5.0,
            "audit_coverage_pct": 97.0,
        }
    )
    pm = primary_metrics_for_row(row)
    assert len(pm) >= 4


def test_narrative_keys_llm_rag() -> None:
    assert "context_retrieval_hit_rate" in narrative_metric_keys_for_arch("llm_rag")


def test_humanize_column_name_known_and_fallback() -> None:
    assert humanize_column_name("model_type") == "Model type"
    assert " %" in humanize_column_name("foo_pct") or humanize_column_name("foo_pct").endswith("%")


def test_format_snapshot_value_numeric_and_missing() -> None:
    assert format_snapshot_value("error_pct", 3.14159) == "3.14"
    assert format_snapshot_value("x", float("nan")) == "—"
    assert format_snapshot_value("security_anomaly_count", 12) == "12"
