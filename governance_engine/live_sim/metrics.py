"""Which columns to simulate per model (archetype + governance-critical core)."""

from __future__ import annotations

import pandas as pd

from governance_engine.telemetry_display import PRIMARY_BY_ARCHETYPE

# Always try to simulate these when present — stress + evaluators consume them.
CORE_METRICS: tuple[str, ...] = (
    "accuracy_pct",
    "error_pct",
    "drift_index",
    "escalation_pct",
    "audit_coverage_pct",
    "compliance_incidents",
    "hallucination_live_pct",
    "observed_latency_p95_ms",
    "policy_violation_rate",
    "security_anomaly_count",
    "anomaly_frequency",
    "manual_review_burden",
)


def _default_for_metric(metric: str) -> float:
    d = {
        "accuracy_pct": 90.0,
        "error_pct": 4.0,
        "drift_index": 0.08,
        "escalation_pct": 8.0,
        "audit_coverage_pct": 88.0,
        "compliance_incidents": 0.5,
        "hallucination_live_pct": 2.0,
        "observed_latency_p95_ms": 450.0,
        "policy_violation_rate": 0.001,
        "security_anomaly_count": 0.5,
        "anomaly_frequency": 3.0,
        "manual_review_burden": 0.05,
        "document_extraction_fidelity": 0.94,
        "ocr_confidence_score": 0.92,
        "field_validation_match_rate": 0.96,
        "image_authenticity_confidence": 0.95,
        "patch_embedding_drift_score": 0.06,
        "transcription_wer_proxy_pct": 5.0,
        "speaker_id_confidence": 0.95,
        "intent_stability_score": 0.92,
        "output_consistency_score": 0.9,
        "expected_hallucination_rate": 0.02,
        "observed_grounding_score": 0.88,
        "citation_coverage": 0.85,
        "unsupported_claim_rate": 0.03,
        "retrieval_failure_rate": 0.04,
        "context_retrieval_hit_rate": 0.92,
        "false_positive_rate": 0.02,
        "false_negative_rate": 0.015,
    }
    return float(d.get(metric, 0.5))


def bounds_for_metric(metric: str) -> tuple[float, float]:
    """Clip simulated values to plausible ranges."""
    b: dict[str, tuple[float, float]] = {
        "accuracy_pct": (0.0, 100.0),
        "error_pct": (0.0, 100.0),
        "drift_index": (0.0, 1.0),
        "escalation_pct": (0.0, 100.0),
        "audit_coverage_pct": (0.0, 100.0),
        "compliance_incidents": (0.0, 50.0),
        "hallucination_live_pct": (0.0, 100.0),
        "observed_latency_p95_ms": (50.0, 50000.0),
        "policy_violation_rate": (0.0, 0.05),
        "security_anomaly_count": (0.0, 100.0),
        "anomaly_frequency": (0.0, 100.0),
        "manual_review_burden": (0.0, 1.0),
        "document_extraction_fidelity": (0.0, 1.0),
        "ocr_confidence_score": (0.0, 1.0),
        "field_validation_match_rate": (0.0, 1.0),
        "image_authenticity_confidence": (0.0, 1.0),
        "patch_embedding_drift_score": (0.0, 1.0),
        "transcription_wer_proxy_pct": (0.0, 100.0),
        "speaker_id_confidence": (0.0, 1.0),
        "intent_stability_score": (0.0, 1.0),
        "output_consistency_score": (0.0, 1.0),
        "expected_hallucination_rate": (0.0, 1.0),
        "observed_grounding_score": (0.0, 1.0),
        "citation_coverage": (0.0, 1.0),
        "unsupported_claim_rate": (0.0, 1.0),
        "retrieval_failure_rate": (0.0, 1.0),
        "context_retrieval_hit_rate": (0.0, 1.0),
        "false_positive_rate": (0.0, 1.0),
        "false_negative_rate": (0.0, 1.0),
    }
    return b.get(metric, (0.0, 1.0e6))


def initial_value_for_row(row: pd.Series, metric: str) -> float:
    if metric in row.index and pd.notna(row.get(metric)):
        try:
            return float(row[metric])
        except (TypeError, ValueError):
            pass
    return _default_for_metric(metric)


def metrics_for_row(row: pd.Series) -> list[str]:
    """Columns to simulate for this model (union of core + archetype primary)."""
    arch = str(row.get("telemetry_archetype", "tabular_ml"))
    spec = PRIMARY_BY_ARCHETYPE.get(arch) or PRIMARY_BY_ARCHETYPE["tabular_ml"]
    primary = {col for col, _ in spec}
    out: set[str] = set(CORE_METRICS) | primary
    # keep only columns that exist in workbook schema
    out = {c for c in out if c in row.index}
    if not out:
        out = {c for c in CORE_METRICS if c in row.index}
    return sorted(out)


def clip_metric(metric: str, value: float) -> float:
    lo, hi = bounds_for_metric(metric)
    return max(lo, min(hi, value))
