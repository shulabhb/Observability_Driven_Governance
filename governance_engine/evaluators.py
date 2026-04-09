"""Compare observed telemetry to expected bands per dimension."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from governance_engine.data_bridge import (
    NARRATIVE_ARCHETYPES,
    NLP_CLS_ARCHETYPES,
    SPEECH_ARCHETYPES,
    VISION_CNN_ARCHETYPES,
    VISION_DOC_ARCHETYPES,
)


@dataclass
class Breach:
    metric: str
    observed: Any
    expected: str
    detail: str


@dataclass
class DimensionResult:
    dimension: str
    severity: float  # 0..1 for this dimension
    breaches: list[Breach] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None


def _num(row: pd.Series, key: str) -> float | None:
    if key not in row.index:
        return None
    v = row[key]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def evaluate_reliability(row: pd.Series, bands: dict[str, Any]) -> DimensionResult:
    breaches: list[Breach] = []
    acc = _num(row, "accuracy_pct")
    err = _num(row, "error_pct")
    drift = _num(row, "drift_index")
    esc = _num(row, "escalation_pct")
    exp_lat = _num(row, "expected_latency_p95_ms")
    obs_lat = _num(row, "observed_latency_p95_ms")

    min_acc = float(bands.get("min_accuracy_pct", 80))
    max_err = float(bands.get("max_error_pct", 10))
    max_drift = float(bands.get("max_drift_index", 0.25))
    max_esc = float(bands.get("max_escalation_pct", 25))
    max_lat_ratio = float(bands.get("max_latency_ratio", 1.4))

    parts: list[float] = []

    if acc is not None and acc < min_acc:
        sev = min(1.0, (min_acc - acc) / max(min_acc, 1.0))
        parts.append(sev)
        breaches.append(
            Breach(
                "accuracy_pct",
                acc,
                f">= {min_acc}",
                "Observed accuracy below expected floor",
            )
        )
    if err is not None and err > max_err:
        sev = min(1.0, (err - max_err) / max(max_err, 0.1))
        parts.append(sev)
        breaches.append(
            Breach("error_pct", err, f"<= {max_err}", "Error rate above expected ceiling")
        )
    if drift is not None and drift > max_drift:
        sev = min(1.0, (drift - max_drift) / max(max_drift, 0.01))
        parts.append(sev)
        breaches.append(
            Breach("drift_index", drift, f"<= {max_drift}", "Drift above expected band")
        )
    if esc is not None and esc > max_esc:
        sev = min(1.0, (esc - max_esc) / max(max_esc, 1.0))
        parts.append(sev)
        breaches.append(
            Breach("escalation_pct", esc, f"<= {max_esc}", "Escalation rate elevated")
        )
    if (
        exp_lat is not None
        and obs_lat is not None
        and exp_lat > 0
        and obs_lat > exp_lat * max_lat_ratio
    ):
        ratio = obs_lat / exp_lat
        sev = min(1.0, (ratio - max_lat_ratio) / max_lat_ratio)
        parts.append(sev)
        breaches.append(
            Breach(
                "observed_latency_p95_ms",
                obs_lat,
                f"<= {max_lat_ratio} x expected",
                "Latency materially above expected p95",
            )
        )

    cons = _num(row, "output_consistency_score")
    min_cons = float(bands.get("min_output_consistency", 0.75))
    if cons is not None and cons < min_cons:
        sev = min(1.0, (min_cons - cons) / min_cons)
        parts.append(sev)
        breaches.append(
            Breach(
                "output_consistency_score",
                cons,
                f">= {min_cons}",
                "Output consistency below expected",
            )
        )

    severity = max(parts) if parts else 0.0
    return DimensionResult(dimension="reliability", severity=float(severity), breaches=breaches)


def evaluate_narrative_assurance(
    row: pd.Series, bands: dict[str, Any], archetype: str
) -> DimensionResult:
    if archetype not in NARRATIVE_ARCHETYPES | VISION_DOC_ARCHETYPES | VISION_CNN_ARCHETYPES | SPEECH_ARCHETYPES | NLP_CLS_ARCHETYPES:
        return DimensionResult(
            dimension="narrative_assurance",
            severity=0.0,
            skipped=True,
            skip_reason="Archetype has no narrative / modality assurance block in POC",
        )

    breaches: list[Breach] = []
    parts: list[float] = []

    if archetype in NARRATIVE_ARCHETYPES:
        g = _num(row, "observed_grounding_score")
        min_g = float(bands.get("min_grounding_score", 0.72))
        if g is not None and g < min_g:
            parts.append(min(1.0, (min_g - g) / min_g))
            breaches.append(
                Breach(
                    "observed_grounding_score",
                    g,
                    f">= {min_g}",
                    "Grounding below expected for generative system",
                )
            )
        uc = _num(row, "unsupported_claim_rate")
        max_uc = float(bands.get("max_unsupported_claim_rate", 0.15))
        if uc is not None and uc > max_uc:
            parts.append(min(1.0, (uc - max_uc) / max(max_uc, 0.01)))
            breaches.append(
                Breach(
                    "unsupported_claim_rate",
                    uc,
                    f"<= {max_uc}",
                    "Unsupported-claim rate above band",
                )
            )
        rf = _num(row, "retrieval_failure_rate")
        max_rf = float(bands.get("max_retrieval_failure_rate", 0.05))
        if rf is not None and rf > max_rf:
            parts.append(min(1.0, (rf - max_rf) / max(max_rf, 0.001)))
            breaches.append(
                Breach(
                    "retrieval_failure_rate",
                    rf,
                    f"<= {max_rf}",
                    "Retrieval failures elevated",
                )
            )
        hit = _num(row, "context_retrieval_hit_rate")
        min_hit = float(bands.get("min_context_retrieval_hit_rate", 0.0))
        if hit is not None and min_hit > 0 and hit < min_hit:
            parts.append(min(1.0, (min_hit - hit) / min_hit))
            breaches.append(
                Breach(
                    "context_retrieval_hit_rate",
                    hit,
                    f">= {min_hit}",
                    "Context retrieval hit rate below band (RAG)",
                )
            )

    if archetype in VISION_DOC_ARCHETYPES:
        fm = _num(row, "field_validation_match_rate")
        min_fm = float(bands.get("min_field_validation_match_rate", 0.72))
        if fm is not None and fm < min_fm:
            parts.append(min(1.0, (min_fm - fm) / min_fm))
            breaches.append(
                Breach(
                    "field_validation_match_rate",
                    fm,
                    f">= {min_fm}",
                    "Document field validation below expected",
                )
            )

    if archetype in VISION_CNN_ARCHETYPES:
        auth = _num(row, "image_authenticity_confidence")
        if auth is not None and auth < 0.75:
            parts.append(min(1.0, (0.75 - auth) / 0.75))
            breaches.append(
                Breach(
                    "image_authenticity_confidence",
                    auth,
                    ">= 0.75",
                    "Image authenticity signal weak",
                )
            )

    if archetype in SPEECH_ARCHETYPES:
        wer = _num(row, "transcription_wer_proxy_pct")
        if wer is not None and wer > 8.0:
            parts.append(min(1.0, (wer - 8.0) / 12.0))
            breaches.append(
                Breach(
                    "transcription_wer_proxy_pct",
                    wer,
                    "<= 8 (POC band)",
                    "Transcription error proxy elevated",
                )
            )

    if archetype in NLP_CLS_ARCHETYPES:
        st = _num(row, "intent_stability_score")
        if st is not None and st < 0.78:
            parts.append(min(1.0, (0.78 - st) / 0.78))
            breaches.append(
                Breach(
                    "intent_stability_score",
                    st,
                    ">= 0.78",
                    "Intent stability below band",
                )
            )

    if not parts and archetype in NARRATIVE_ARCHETYPES:
        # Generative row but no narrative columns populated (data gap) — light penalty
        if _num(row, "observed_grounding_score") is None:
            return DimensionResult(
                dimension="narrative_assurance",
                severity=0.15,
                skipped=False,
                breaches=[
                    Breach(
                        "observed_grounding_score",
                        None,
                        "present",
                        "Narrative metrics missing for generative archetype",
                    )
                ],
            )

    severity = max(parts) if parts else 0.0
    return DimensionResult(
        dimension="narrative_assurance", severity=float(severity), breaches=breaches
    )


def evaluate_compliance_security(row: pd.Series, bands: dict[str, Any]) -> DimensionResult:
    breaches: list[Breach] = []
    parts: list[float] = []

    pv = _num(row, "policy_violation_rate")
    max_pv = float(bands.get("max_policy_violation_rate", 0.01))
    if pv is not None and pv > max_pv:
        parts.append(min(1.0, (pv - max_pv) / max(max_pv, 1e-6)))
        breaches.append(
            Breach(
                "policy_violation_rate",
                pv,
                f"<= {max_pv}",
                "Policy violation rate above band",
            )
        )

    sec = row["security_anomaly_count"] if "security_anomaly_count" in row.index else None
    if sec is not None and not pd.isna(sec):
        sec_i = int(sec)
        max_sec = int(bands.get("max_security_anomaly_count", 5))
        if sec_i > max_sec:
            parts.append(min(1.0, (sec_i - max_sec) / max(max_sec, 1)))
            breaches.append(
                Breach(
                    "security_anomaly_count",
                    sec_i,
                    f"<= {max_sec}",
                    "Security anomaly count elevated",
                )
            )

    ci = row["compliance_incidents"] if "compliance_incidents" in row.index else None
    if ci is not None and not pd.isna(ci):
        ci_i = int(ci)
        max_ci = int(bands.get("max_compliance_incidents", 2))
        if ci_i > max_ci:
            parts.append(min(1.0, (ci_i - max_ci) / max(max_ci, 1)))
            breaches.append(
                Breach(
                    "compliance_incidents",
                    ci_i,
                    f"<= {max_ci}",
                    "Compliance incidents above band",
                )
            )

    severity = max(parts) if parts else 0.0
    return DimensionResult(
        dimension="compliance_security", severity=float(severity), breaches=breaches
    )


def evaluate_auditability(row: pd.Series, bands: dict[str, Any]) -> DimensionResult:
    breaches: list[Breach] = []
    parts: list[float] = []

    ac = _num(row, "audit_coverage_pct")
    min_ac = float(bands.get("min_audit_coverage_pct", 90))
    if ac is not None and ac < min_ac:
        parts.append(min(1.0, (min_ac - ac) / min_ac))
        breaches.append(
            Breach(
                "audit_coverage_pct",
                ac,
                f">= {min_ac}",
                "Audit coverage below expected",
            )
        )

    dt = _num(row, "decision_trace_completeness")
    min_dt = float(bands.get("min_decision_trace_completeness", 0.72))
    if dt is not None and dt < min_dt:
        parts.append(min(1.0, (min_dt - dt) / min_dt))
        breaches.append(
            Breach(
                "decision_trace_completeness",
                dt,
                f">= {min_dt}",
                "Decision trace completeness below band",
            )
        )

    mac = _num(row, "minimum_required_audit_coverage")
    if ac is not None and mac is not None and ac / 100.0 < mac - 0.02:
        parts.append(0.6)
        breaches.append(
            Breach(
                "audit_coverage_pct",
                ac,
                f">= {mac * 100:.0f}% policy",
                "Observed audit coverage below policy minimum",
            )
        )

    severity = max(parts) if parts else 0.0
    return DimensionResult(dimension="auditability", severity=float(severity), breaches=breaches)


def run_all_evaluators(
    row: pd.Series, bands: dict[str, Any], archetype: str
) -> list[DimensionResult]:
    arch = str(archetype)
    return [
        evaluate_reliability(row, bands),
        evaluate_narrative_assurance(row, bands, arch),
        evaluate_compliance_security(row, bands),
        evaluate_auditability(row, bands),
    ]
