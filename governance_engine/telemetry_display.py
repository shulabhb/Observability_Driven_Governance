"""
Archetype-aware metric labels for dashboards (which signals to emphasize).

Tabular / classical models: emphasize calibration-style metrics — not labeled as “drift”
for stakeholder copy; we use population_stability_index as alias for drift_index in UI.
"""

from __future__ import annotations

import numbers
import re

import pandas as pd

# (column_name, display_label)
PRIMARY_BY_ARCHETYPE: dict[str, list[tuple[str, str]]] = {
    "tabular_ml": [
        ("accuracy_pct", "Accuracy %"),
        ("error_pct", "Error rate %"),
        ("false_positive_rate", "False positive rate"),
        ("false_negative_rate", "False negative rate"),
        ("escalation_pct", "Escalation %"),
        ("audit_coverage_pct", "Audit coverage %"),
    ],
    "time_series": [
        ("accuracy_pct", "Forecast accuracy %"),
        ("error_pct", "Error rate %"),
        ("drift_index", "Series / residual drift"),
        ("output_consistency_score", "Output stability"),
        ("audit_coverage_pct", "Audit coverage %"),
    ],
    "vision_cnn": [
        ("accuracy_pct", "Accuracy %"),
        ("error_pct", "Error rate %"),
        ("image_authenticity_confidence", "Image authenticity score"),
        ("patch_embedding_drift_score", "Embedding drift"),
        ("audit_coverage_pct", "Audit coverage %"),
    ],
    "vision_document": [
        ("document_extraction_fidelity", "Extraction fidelity"),
        ("field_validation_match_rate", "Field match rate"),
        ("ocr_confidence_score", "OCR confidence"),
        ("audit_coverage_pct", "Audit coverage %"),
    ],
    "sequence_anomaly": [
        ("accuracy_pct", "Detection accuracy %"),
        ("error_pct", "Error rate %"),
        ("drift_index", "Behavior drift index"),
        ("anomaly_frequency", "Anomaly events"),
        ("audit_coverage_pct", "Audit coverage %"),
    ],
    "nlp_classifier": [
        ("accuracy_pct", "Accuracy %"),
        ("error_pct", "Error rate %"),
        ("intent_stability_score", "Intent stability"),
        ("output_consistency_score", "Output consistency"),
        ("audit_coverage_pct", "Audit coverage %"),
    ],
    "speech_asr": [
        ("accuracy_pct", "Pipeline accuracy %"),
        ("transcription_wer_proxy_pct", "WER proxy %"),
        ("speaker_id_confidence", "Speaker ID confidence"),
        ("error_pct", "Error rate %"),
        ("audit_coverage_pct", "Audit coverage %"),
    ],
    "reinforcement_learning": [
        ("accuracy_pct", "Policy score / accuracy %"),
        ("error_pct", "Error rate %"),
        ("drift_index", "Environment drift index"),
        ("escalation_pct", "Escalation %"),
        ("audit_coverage_pct", "Audit coverage %"),
    ],
    "hybrid_rules_ml": [
        ("accuracy_pct", "Accuracy %"),
        ("error_pct", "Error rate %"),
        ("false_positive_rate", "False positive rate"),
        ("escalation_pct", "Escalation %"),
        ("audit_coverage_pct", "Audit coverage %"),
    ],
    "llm_rag": [
        ("observed_grounding_score", "Grounding score"),
        ("context_retrieval_hit_rate", "Retrieval hit rate"),
        ("retrieval_failure_rate", "Retrieval failure rate"),
        ("citation_coverage", "Citation coverage"),
        ("unsupported_claim_rate", "Unsupported claim rate"),
        ("error_pct", "Error rate %"),
    ],
    "llm_text": [
        ("observed_grounding_score", "Grounding score"),
        ("expected_hallucination_rate", "Hallucination rate (expected)"),
        ("citation_coverage", "Citation coverage"),
        ("unsupported_claim_rate", "Unsupported claim rate"),
        ("error_pct", "Error rate %"),
    ],
    "agentic_system": [
        ("observed_grounding_score", "Grounding score"),
        ("escalation_pct", "Escalation %"),
        ("decision_trace_completeness", "Trace completeness"),
        ("unsupported_claim_rate", "Unsupported claim rate"),
        ("error_pct", "Error rate %"),
    ],
}


def primary_metrics_for_row(row: pd.Series) -> list[tuple[str, str, object]]:
    """Return (column, label, value) for metrics that exist and are non-null."""
    arch = str(row.get("telemetry_archetype", "tabular_ml"))
    spec = PRIMARY_BY_ARCHETYPE.get(arch) or PRIMARY_BY_ARCHETYPE["tabular_ml"]
    out: list[tuple[str, str, object]] = []
    for col, label in spec:
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v):
            continue
        out.append((col, label, v))
    return out


def format_metric_value(col: str, v: object) -> str:
    if isinstance(v, float):
        if "rate" in col or "score" in col or "fidelity" in col or "confidence" in col:
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"
        return f"{v:.2f}"
    return str(v)


# Human-readable column titles for workbook / dashboard tables
_COL_PRETTY: dict[str, str] = {
    "model": "Model",
    "model_type": "Model type",
    "use_case": "Use case",
    "telemetry_archetype": "Telemetry archetype",
    "updated_date": "Updated date",
    "active_users_mau": "Active users (MAU)",
    "accuracy_pct": "Accuracy %",
    "error_pct": "Error %",
    "drift_index": "Drift index",
    "hallucination_live_pct": "Hallucination (live) %",
    "escalation_pct": "Escalation %",
    "compliance_incidents": "Compliance incidents",
    "audit_coverage_pct": "Audit coverage %",
    "document_extraction_fidelity": "Document extraction fidelity",
    "ocr_confidence_score": "OCR confidence",
    "field_validation_match_rate": "Field validation match rate",
    "image_authenticity_confidence": "Image authenticity confidence",
    "patch_embedding_drift_score": "Patch embedding drift",
    "transcription_wer_proxy_pct": "WER proxy %",
    "speaker_id_confidence": "Speaker ID confidence",
    "intent_stability_score": "Intent stability",
    "expected_hallucination_rate": "Expected hallucination rate",
    "expected_grounding_score": "Expected grounding score",
    "observed_grounding_score": "Observed grounding score",
    "citation_coverage": "Citation coverage",
    "unsupported_claim_rate": "Unsupported claim rate",
    "retrieval_failure_rate": "Retrieval failure rate",
    "context_retrieval_hit_rate": "Context retrieval hit rate",
    "expected_accuracy": "Expected accuracy %",
    "expected_error_rate": "Expected error rate %",
    "expected_latency_p95_ms": "Expected latency p95 (ms)",
    "observed_latency_p95_ms": "Observed latency p95 (ms)",
    "false_positive_rate": "False positive rate",
    "false_negative_rate": "False negative rate",
    "output_consistency_score": "Output consistency",
    "expected_drift_score": "Expected drift score",
    "drift_velocity": "Drift velocity",
    "feature_distribution_shift": "Feature distribution shift",
    "anomaly_frequency": "Anomaly frequency",
    "confidence_stability_score": "Confidence stability",
    "business_function": "Business function",
    "deployment_scope": "Deployment scope",
    "regulatory_sensitivity": "Regulatory sensitivity",
    "model_owner": "Model owner",
    "control_owner": "Control owner",
    "review_frequency": "Review frequency",
    "last_validation_days_ago": "Last validation (days ago)",
    "minimum_required_audit_coverage": "Minimum required audit coverage %",
    "decision_trace_completeness": "Decision trace completeness",
    "previous_escalation_count": "Previous escalation count",
    "policy_violation_rate": "Policy violation rate",
    "security_anomaly_count": "Security anomaly count",
    "cost_per_interaction": "Cost per interaction ($)",
    "inference_cost": "Inference cost ($)",
    "roi_multiple": "ROI multiple",
    "expected_cost_per_1k_requests": "Expected cost / 1k requests",
    "observed_cost_per_1k_requests": "Observed cost / 1k requests",
    "manual_review_burden": "Manual review burden",
}


def humanize_column_name(col: str) -> str:
    """Turn workbook column names into short labels for UI tables."""
    if col in _COL_PRETTY:
        return _COL_PRETTY[col]
    s = col.replace("_", " ")
    s = " ".join(w.capitalize() for w in s.split())
    if s.endswith(" Pct"):
        s = s[:-4] + " %"
    s = re.sub(r" P95 Ms$", " p95 (ms)", s)
    return s


def format_snapshot_value(col: str, v: object) -> str:
    """Format a telemetry cell for profile / snapshot tables (dates, ints, floats)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except (ValueError, TypeError):
        pass
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if hasattr(v, "isoformat") and not isinstance(v, str):
        try:
            return str(v.isoformat())[:19].replace("T", " ")
        except (TypeError, ValueError):
            pass
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, numbers.Integral):
        return f"{int(v):,}"
    if isinstance(v, numbers.Real):
        return format_metric_value(col, float(v))
    return str(v).strip()


def narrative_metric_keys_for_arch(arch: str) -> list[str]:
    """Subset of columns for Narrative / modality panel."""
    m = {
        "llm_rag": [
            "expected_hallucination_rate",
            "observed_grounding_score",
            "citation_coverage",
            "unsupported_claim_rate",
            "retrieval_failure_rate",
            "context_retrieval_hit_rate",
        ],
        "llm_text": [
            "expected_hallucination_rate",
            "expected_grounding_score",
            "observed_grounding_score",
            "citation_coverage",
            "unsupported_claim_rate",
        ],
        "agentic_system": [
            "expected_hallucination_rate",
            "observed_grounding_score",
            "citation_coverage",
            "unsupported_claim_rate",
            "decision_trace_completeness",
        ],
        "vision_document": [
            "document_extraction_fidelity",
            "ocr_confidence_score",
            "field_validation_match_rate",
        ],
        "vision_cnn": ["image_authenticity_confidence", "patch_embedding_drift_score"],
        "speech_asr": ["transcription_wer_proxy_pct", "speaker_id_confidence"],
        "nlp_classifier": ["intent_stability_score", "output_consistency_score"],
    }
    return m.get(arch, [])


def reliability_metric_keys_for_arch(arch: str) -> list[tuple[str, str]]:
    """Reliability tab: archetype-filtered core KPIs."""
    base = [
        ("accuracy_pct", "Accuracy %"),
        ("error_pct", "Error %"),
        ("expected_accuracy", "Expected accuracy %"),
        ("expected_error_rate", "Expected error rate %"),
        ("false_positive_rate", "False positive rate"),
        ("false_negative_rate", "False negative rate"),
        ("expected_latency_p95_ms", "Expected latency p95 (ms)"),
        ("observed_latency_p95_ms", "Observed latency p95 (ms)"),
        ("output_consistency_score", "Output consistency"),
    ]
    arch = str(arch)
    if arch == "tabular_ml":
        return [b for b in base if b[0] != "drift_index"]
    if arch in ("llm_rag", "llm_text", "agentic_system"):
        return base + [("drift_index", "Drift index (telemetry)")]
    if arch == "time_series":
        return base + [("drift_index", "Series drift index")]
    return base + [("drift_index", "Drift index")]
