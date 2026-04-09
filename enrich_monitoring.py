#!/usr/bin/env python3
"""
Enrich AI governance monitoring spreadsheets with archetype-aware telemetry.

Stress shaping uses observable KPIs only (drift, error, escalation, audit, accuracy,
compliance incidents) — not exported risk labels. Risk columns from the source sheet
are omitted from the enriched workbook for downstream risk prediction.

Deterministic: numpy RNG seeded per (model_id, field_group).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd

SOURCE_XLSX = Path(__file__).resolve().parent / "Monitoring Report.xlsx"
OUT_XLSX = Path(__file__).resolve().parent / "enriched_monitoring_report.xlsx"
OUT_MD = Path(__file__).resolve().parent / "enrichment_summary.md"

GLOBAL_SEED = 20260409

Tier = Literal["green", "yellow", "red"]


def telemetry_snapshot_timestamp(model_id: str, phase: Literal["pre", "post"]) -> pd.Timestamp:
    """Deterministic snapshot date per model (pre-go-live vs post-go-live windows)."""
    h = int(
        hashlib.sha256(f"{GLOBAL_SEED}:{model_id}:snapshot_date:{phase}".encode()).hexdigest()[:8],
        16,
    )
    base = datetime(2026, 1, 6) if phase == "pre" else datetime(2026, 3, 2)
    offset_days = h % 52
    return pd.Timestamp(base + timedelta(days=offset_days))


TelemetryArchetype = Literal[
    "tabular_ml",
    "time_series",
    "vision_cnn",
    "vision_document",
    "sequence_anomaly",
    "nlp_classifier",
    "speech_asr",
    "reinforcement_learning",
    "hybrid_rules_ml",
    "llm_rag",
    "llm_text",
    "agentic_system",
]


def _row_rng(model_id: str, salt: str) -> np.random.Generator:
    h = hashlib.sha256(f"{GLOBAL_SEED}:{model_id}:{salt}".encode()).hexdigest()
    seed = int(h[:16], 16) % (2**32)
    return np.random.default_rng(seed)


def infer_telemetry_archetype(model_type: str, use_case: str) -> TelemetryArchetype:
    """Classify model for which telemetry blocks apply (order: most specific first)."""
    mt = model_type.strip().lower()
    uc = use_case.strip().lower()
    t = f"{mt} {uc}"
    if "speech" in t or "transcription" in uc:
        return "speech_asr"
    if "vision transformer" in mt:
        return "vision_document"
    if "cnn" in mt or "check image" in uc or ("image" in uc and "fraud" in uc):
        return "vision_cnn"
    if "rag" in t:
        return "llm_rag"
    if "agentic" in t or "multi-agent" in t:
        return "agentic_system"
    if "bert" in t or ("nlp" in t and "classif" in uc):
        return "nlp_classifier"
    if "gpt" in t or "llm" in t or "fine-tuned" in t:
        return "llm_text"
    if "transformer" in mt and "vision" not in mt:
        return "llm_text"
    if "reinforcement" in t:
        return "reinforcement_learning"
    if "hybrid" in t and "rules" in t:
        return "hybrid_rules_ml"
    if "arima" in t or "time series" in mt:
        return "time_series"
    if "lstm" in t:
        return "sequence_anomaly"
    if "tabular" in t or "automl" in t:
        return "tabular_ml"
    if any(x in mt for x in ("logistic", "xgboost", "random forest", "gradient")):
        return "tabular_ml"
    return "tabular_ml"


def infer_regulatory_sensitivity_from_context(use_case: str, model_type: str) -> Literal["low", "medium", "high"]:
    """Business context only — does not use L1/L2/L3 risk tier labels."""
    t = f"{use_case} {model_type}".lower()
    if any(
        x in t
        for x in (
            "sanction",
            "aml",
            "kyc",
            "credit risk",
            "credit memo",
            "fraud",
            "regulatory q&a",
            "pd model",
        )
    ):
        return "high"
    if any(
        x in t
        for x in (
            "treasury",
            "trading",
            "underwriting",
            "loan",
            "compliance",
            "memo",
            "cross-sell",
        )
    ):
        return "medium"
    return "low"


def compute_stress_tier_from_metrics(
    drift: float,
    error: float,
    escalation: float,
    audit: float,
    accuracy: float | None,
    compliance_incidents: float,
) -> Tier:
    """
    Map observable KPIs to an internal stress tier for synthetic telemetry shaping.
    Does not use risk_status or Risk Tier columns.
    """
    score = 0
    if drift >= 0.22:
        score += 3
    elif drift >= 0.15:
        score += 2
    elif drift >= 0.10:
        score += 1
    if error >= 12:
        score += 3
    elif error >= 9:
        score += 2
    elif error >= 6.5:
        score += 1
    if escalation >= 28:
        score += 3
    elif escalation >= 20:
        score += 2
    elif escalation >= 12:
        score += 1
    if audit <= 74:
        score += 3
    elif audit <= 80:
        score += 2
    elif audit <= 86:
        score += 1
    if accuracy is not None:
        if accuracy <= 72:
            score += 3
        elif accuracy <= 78:
            score += 2
        elif accuracy <= 82:
            score += 1
    if compliance_incidents >= 2:
        score += 2
    elif compliance_incidents >= 1:
        score += 1
    if score >= 7:
        return "red"
    if score >= 4:
        return "yellow"
    return "green"


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _pick_owner(model_id: str, kind: str) -> str:
    r = _row_rng(model_id, f"owner_{kind}")
    opts = (
        ("MRM — Enterprise Analytics", "MRM — Retail & Cards", "MRM — Markets & Treasury", "MRM — Operations")
        if kind == "model"
        else (
            "Controls — 2LoD Model Risk",
            "Controls — 1LoD Data & Analytics",
            "Controls — Operational Risk",
            "Controls — Compliance Testing",
        )
    )
    return str(opts[int(r.integers(0, len(opts)))])


@dataclass
class EnrichContext:
    model_id: str
    tier: Tier
    archetype: TelemetryArchetype
    use_case: str
    model_type: str
    accuracy_pre: float | None
    accuracy_post: float | None
    error_post: float
    drift_post: float
    escalation_post: float
    audit_post: float
    compliance_incidents: float
    cost_per_interaction: float
    inference_cost: float
    hallucination_live: float | None
    deployment_scope: str


def fills_llm_narrative(a: TelemetryArchetype) -> bool:
    return a in ("llm_rag", "llm_text", "agentic_system")


def fills_rag_metrics(a: TelemetryArchetype) -> bool:
    return a == "llm_rag"


def enrich_row(ctx: EnrichContext) -> dict[str, Any]:
    rng = _row_rng(ctx.model_id, "enrich")
    t = ctx.tier
    arch = ctx.archetype
    rs = infer_regulatory_sensitivity_from_context(ctx.use_case, ctx.model_type)

    out: dict[str, Any] = {}
    out["telemetry_archetype"] = arch

    # --- Business context (no exported risk tier / risk status)
    out["business_function"] = (
        "Retail credit & fraud"
        if "credit" in ctx.use_case.lower() or "fraud" in ctx.use_case.lower()
        else "Markets & treasury"
        if "treasury" in ctx.use_case.lower() or "trading" in ctx.use_case.lower()
        else "Compliance & financial crime"
        if any(
            x in ctx.use_case.lower()
            for x in ("aml", "sanction", "kyc", "compliance", "regulatory")
        )
        else "Operations & customer service"
        if "complaint" in ctx.use_case.lower() or "onboarding" in ctx.use_case.lower()
        else "Enterprise analytics"
    )
    out["deployment_scope"] = ctx.deployment_scope
    if rs == "high" or "customer-facing" in ctx.deployment_scope:
        out["customer_impact"] = "high" if rs == "high" else "medium-high"
    elif rs == "medium":
        out["customer_impact"] = "medium"
    else:
        out["customer_impact"] = "low-to-medium"
    out["regulatory_sensitivity"] = rs
    out["model_owner"] = _pick_owner(ctx.model_id, "model")
    out["control_owner"] = _pick_owner(ctx.model_id, "control")
    out["review_frequency"] = (
        "monthly"
        if rs == "high"
        else "quarterly"
        if rs == "medium" or ctx.deployment_scope == "customer-facing"
        else "semi-annual"
    )
    base_val = 38 + (8 if rs == "high" else 0)
    out["last_validation_days_ago"] = int(_clip(base_val + rng.normal(0, 8), 7, 120))

    # --- Reliability / performance
    exp_acc = ctx.accuracy_pre if ctx.accuracy_pre is not None else ctx.accuracy_post
    if exp_acc is None:
        exp_acc = 85.0
    if rs == "high":
        exp_acc = exp_acc + 1.0
    if rs == "low":
        exp_acc = exp_acc - 0.5

    if t == "green":
        exp_acc = exp_acc + rng.uniform(0, 1.0)
    elif t == "yellow":
        exp_acc = exp_acc - rng.uniform(0.3, 1.5)
    else:
        exp_acc = exp_acc - rng.uniform(1.0, 3.5)

    out["expected_accuracy"] = round(float(exp_acc), 2)
    out["expected_error_rate"] = round(_clip(100.0 - float(exp_acc), 0.5, 35.0), 2)

    obs_err = ctx.error_post
    out["false_positive_rate"] = round(
        obs_err * (0.35 if "fraud" in ctx.use_case.lower() else 0.45), 3
    )
    out["false_negative_rate"] = round(
        obs_err * (0.65 if "fraud" in ctx.use_case.lower() else 0.55), 3
    )
    if t == "red":
        out["false_negative_rate"] = round(out["false_negative_rate"] * 1.35, 3)
        out["false_positive_rate"] = round(out["false_positive_rate"] * 1.15, 3)
    elif t == "yellow":
        out["false_negative_rate"] = round(out["false_negative_rate"] * 1.12, 3)

    base_lat = 120 + ctx.inference_cost * 180 + rng.uniform(-15, 25)
    if fills_llm_narrative(arch):
        base_lat += 160
    if arch == "speech_asr":
        base_lat += 40
    if arch in ("vision_cnn", "vision_document"):
        base_lat += 90
    exp_lat = base_lat * (0.88 if t == "green" else 0.95 if t == "yellow" else 1.0)
    obs_lat = base_lat * (1.0 if t == "green" else 1.1 if t == "yellow" else 1.28)
    out["expected_latency_p95_ms"] = int(round(exp_lat))
    out["observed_latency_p95_ms"] = int(round(obs_lat))

    is_rag = fills_rag_metrics(arch)
    out["retrieval_failure_rate"] = (
        round(
            (0.008 if t == "green" else 0.022 if t == "yellow" else 0.055)
            + (0.012 if is_rag else 0),
            4,
        )
        if fills_llm_narrative(arch)
        else np.nan
    )

    cons = 0.92
    if t == "yellow":
        cons = 0.82
    if t == "red":
        cons = 0.68
    if arch in ("tabular_ml", "time_series", "hybrid_rules_ml"):
        cons += 0.02
    out["output_consistency_score"] = round(
        _clip(cons + rng.normal(0, 0.015), 0.45, 0.98), 3
    )

    # --- Drift / stability
    d0 = ctx.drift_post
    out["expected_drift_score"] = round(
        _clip(d0 * (0.65 if t == "green" else 0.85 if t == "yellow" else 1.05), 0.01, 0.5),
        4,
    )
    out["drift_velocity"] = round(
        _clip(
            (0.01 if t == "green" else 0.03 if t == "yellow" else 0.07)
            + abs(rng.normal(0, 0.004)),
            0.001,
            0.15,
        ),
        4,
    )
    out["feature_distribution_shift"] = round(
        _clip(d0 + (0.01 if t != "green" else -0.005), 0.01, 0.45), 4
    )
    out["concept_shift_flag"] = t != "green" and (
        t == "red" or (t == "yellow" and rng.random() > 0.35)
    )
    out["anomaly_frequency"] = int(
        round(
            _clip(
                (0 if t == "green" else 2 if t == "yellow" else 6)
                + rng.integers(0, 2)
                + ctx.compliance_incidents * 2,
                0,
                25,
            )
        )
    )
    cs = 0.9 if t == "green" else 0.74 if t == "yellow" else 0.55
    out["confidence_stability_score"] = round(_clip(cs + rng.normal(0, 0.02), 0.4, 0.98), 3)

    # --- Archetype-specific telemetry (mutually exclusive groups)
    for k in (
        "expected_hallucination_rate",
        "expected_grounding_score",
        "observed_grounding_score",
        "citation_coverage",
        "unsupported_claim_rate",
        "reviewer_override_rate",
        "context_retrieval_hit_rate",
        "document_extraction_fidelity",
        "ocr_confidence_score",
        "field_validation_match_rate",
        "image_authenticity_confidence",
        "patch_embedding_drift_score",
        "transcription_wer_proxy_pct",
        "speaker_id_confidence",
        "intent_stability_score",
    ):
        out[k] = np.nan

    hall_base = (
        ctx.hallucination_live / 100.0
        if ctx.hallucination_live is not None and not np.isnan(ctx.hallucination_live)
        else 0.04
    )

    if arch == "vision_document":
        out["document_extraction_fidelity"] = round(
            _clip(
                (0.93 if t == "green" else 0.82 if t == "yellow" else 0.64)
                + rng.normal(0, 0.012),
                0.35,
                0.99,
            ),
            3,
        )
        out["ocr_confidence_score"] = round(
            _clip(
                (0.91 if t == "green" else 0.79 if t == "yellow" else 0.58)
                + rng.normal(0, 0.015),
                0.35,
                0.99,
            ),
            3,
        )
        out["field_validation_match_rate"] = round(
            _clip(
                (0.9 if t == "green" else 0.76 if t == "yellow" else 0.55)
                + rng.normal(0, 0.01),
                0.3,
                0.99,
            ),
            3,
        )
    elif arch == "vision_cnn":
        out["image_authenticity_confidence"] = round(
            _clip(
                (0.94 if t == "green" else 0.84 if t == "yellow" else 0.62)
                + rng.normal(0, 0.015),
                0.35,
                0.99,
            ),
            3,
        )
        out["patch_embedding_drift_score"] = round(
            _clip(d0 * 1.1 + (0.02 if t != "green" else 0), 0.01, 0.45), 4
        )
    elif arch == "speech_asr":
        out["transcription_wer_proxy_pct"] = round(
            _clip(
                obs_err * (1.1 if t == "green" else 1.35 if t == "yellow" else 1.65)
                + rng.uniform(0.2, 0.8),
                0.5,
                25.0,
            ),
            2,
        )
        out["speaker_id_confidence"] = round(
            _clip(
                (0.88 if t == "green" else 0.74 if t == "yellow" else 0.55)
                + rng.normal(0, 0.02),
                0.3,
                0.99,
            ),
            3,
        )
    elif arch == "nlp_classifier":
        out["intent_stability_score"] = round(
            _clip(
                (0.91 if t == "green" else 0.8 if t == "yellow" else 0.62)
                + rng.normal(0, 0.015),
                0.35,
                0.99,
            ),
            3,
        )
    elif fills_llm_narrative(arch):
        exp_ground = 0.88 if t == "green" else 0.78 if t == "yellow" else 0.62
        obs_ground = exp_ground - (0.02 if t == "green" else 0.06 if t == "yellow" else 0.14)
        out["expected_hallucination_rate"] = round(
            _clip(hall_base * (0.9 if t == "green" else 1.15 if t == "yellow" else 1.45), 0.001, 0.35),
            4,
        )
        out["expected_grounding_score"] = round(exp_ground, 3)
        out["observed_grounding_score"] = round(
            _clip(obs_ground + rng.normal(0, 0.01), 0.35, 0.98), 3
        )
        out["citation_coverage"] = round(
            _clip(
                (0.82 if t == "green" else 0.68 if t == "yellow" else 0.5)
                + (0.05 if is_rag else 0),
                0.2,
                0.98,
            ),
            3,
        )
        out["unsupported_claim_rate"] = round(
            _clip(out["expected_hallucination_rate"] * 0.55, 0.0005, 0.2), 4
        )
        out["reviewer_override_rate"] = round(
            _clip(
                (0.02 if t == "green" else 0.06 if t == "yellow" else 0.14)
                + (0.03 if rs == "high" else 0),
                0.0,
                0.35,
            ),
            4,
        )
        out["context_retrieval_hit_rate"] = (
            np.nan
            if not is_rag
            else round(
                _clip(
                    0.94 - (0.03 if t == "yellow" else 0.08 if t == "red" else 0),
                    0.5,
                    0.99,
                ),
                3,
            )
        )
        if not is_rag:
            out["retrieval_failure_rate"] = np.nan

    # --- Governance
    min_audit = 0.95 if rs != "low" else 0.88
    out["minimum_required_audit_coverage"] = min_audit
    out["evidence_linked_flag"] = t == "green" or (t == "yellow" and rng.random() > 0.25)
    out["decision_trace_completeness"] = round(
        _clip(
            (0.93 if t == "green" else 0.78 if t == "yellow" else 0.58)
            + rng.normal(0, 0.015),
            0.35,
            0.99,
        ),
        3,
    )
    out["human_review_required"] = (
        rs == "high"
        or ctx.deployment_scope == "customer-facing"
        or arch in ("llm_rag", "agentic_system", "vision_document")
    )
    out["previous_escalation_count"] = int(
        _clip(
            (0 if t == "green" else 2 if t == "yellow" else 5)
            + rng.integers(0, 2)
            + (2 if arch == "agentic_system" else 0),
            0,
            20,
        )
    )
    out["unresolved_audit_findings"] = int(
        _clip(
            (0 if t == "green" else 1 if t == "yellow" else 3) + rng.integers(0, 2),
            0,
            12,
        )
    )
    out["audit_readiness_score"] = round(
        _clip(
            (ctx.audit_post / 100.0) * (1.0 if t == "green" else 0.9 if t == "yellow" else 0.72)
            + rng.normal(0, 0.02),
            0.35,
            0.99,
        ),
        3,
    )

    # --- Compliance / security
    out["policy_violation_rate"] = round(
        _clip(
            (0.0008 if t == "green" else 0.004 if t == "yellow" else 0.014)
            * (2.0 if rs == "high" else 1.0),
            0.0001,
            0.05,
        ),
        5,
    )
    out["compliance_incident_count_enriched"] = int(
        max(0, round(ctx.compliance_incidents + (0 if t == "green" else 1 if t == "yellow" else 3)))
    )
    out["prompt_injection_flag"] = fills_llm_narrative(arch) and t != "green" and rng.random() > 0.55
    sens = "medium"
    if rs == "high" or t == "red":
        sens = "high"
    elif t == "green" and rs == "low":
        sens = "low"
    out["sensitive_data_exposure_risk"] = sens
    if t == "green":
        sec_base = 0 if rng.random() < 0.88 else 1
    elif t == "yellow":
        sec_base = rng.integers(1, 4)
    else:
        sec_base = rng.integers(4, 8)
    out["security_anomaly_count"] = int(
        _clip(sec_base + (1 if t == "red" else 0) + int(ctx.compliance_incidents), 0, 15)
    )
    out["access_control_review_status"] = (
        "current"
        if t == "green"
        else "due_within_30d"
        if t == "yellow"
        else "overdue_escalated"
    )

    # --- Cost / operations
    c1k_exp = (ctx.cost_per_interaction + ctx.inference_cost) * 1000 * 0.95
    c1k_obs = (ctx.cost_per_interaction + ctx.inference_cost) * 1000 * (
        1.0 if t == "green" else 1.08 if t == "yellow" else 1.22
    )
    out["expected_cost_per_1k_requests"] = round(c1k_exp, 2)
    out["observed_cost_per_1k_requests"] = round(c1k_obs, 2)
    out["manual_review_burden"] = round(
        _clip(
            (0.04 if t == "green" else 0.11 if t == "yellow" else 0.24)
            + (0.05 if rs == "high" else 0)
            + rng.normal(0, 0.008),
            0.01,
            0.45,
        ),
        3,
    )
    out["remediation_cost_estimate_usd"] = int(
        round(
            (5000 if t == "green" else 28000 if t == "yellow" else 95000)
            * (1.4 if rs == "high" else 1.0)
            + rng.uniform(500, 4000)
        )
    )
    out["roi_pressure_flag"] = t == "red" or (t == "yellow" and ctx.cost_per_interaction > 0.45)

    return out


def _deployment_scope(model_type: str, use_case: str) -> str:
    t = f"{model_type} {use_case}".lower()
    if any(x in t for x in ("customer", "email", "rm productivity")):
        return "customer-facing"
    return "internal-only"


# Legacy risk labels from the source workbook (do not use as model features).
RISK_COLUMNS_DROP = frozenset({"risk_status", "risk_tier", "Risk Tier"})

# Enriched fields that duplicate what you intend to *predict* (risk score/level, issue,
# action, human review) or are strong label proxies — omitted from the exported sheet.
# Raw observational columns (e.g. compliance_incidents, audit_coverage_pct) are kept.
LABEL_PROXY_COLUMNS = frozenset(
    {
        "human_review_required",
        "audit_readiness_score",
        "sensitive_data_exposure_risk",
        "evidence_linked_flag",
        "concept_shift_flag",
        "prompt_injection_flag",
        "access_control_review_status",
        "roi_pressure_flag",
        "unresolved_audit_findings",
        "remediation_cost_estimate_usd",
        "reviewer_override_rate",
        "customer_impact",
        "compliance_incident_count_enriched",
    }
)


def load_pre_post(*, use_synthetic_fleet: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_synthetic_fleet:
        from synthetic_fleet import build_synthetic_pre_post

        return build_synthetic_pre_post()

    pre = pd.read_excel(SOURCE_XLSX, sheet_name="Pre-Go-Live Monitoring Report")
    post_raw = pd.read_excel(SOURCE_XLSX, sheet_name="Post-Go-Live Monitoring Report", header=None)
    cols = [
        "updated_date",
        "model",
        "model_type",
        "use_case",
        "risk_tier",
        "active_users_mau",
        "accuracy_pct",
        "error_pct",
        "hallucination_live_pct",
        "drift_index",
        "escalation_pct",
        "compliance_incidents",
        "audit_coverage_pct",
        "cost_per_interaction",
        "inference_cost",
        "roi_multiple",
        "risk_status",
    ]
    post = post_raw.iloc[2:].copy()
    post.columns = cols
    post = post.reset_index(drop=True)
    return pre, post


def strip_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c in RISK_COLUMNS_DROP]
    return df.drop(columns=drop, errors="ignore")


def strip_label_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that overlap intended ML outputs or encode prior risk judgments."""
    drop = [c for c in df.columns if c in LABEL_PROXY_COLUMNS]
    return df.drop(columns=drop, errors="ignore")


def order_columns(df: pd.DataFrame, is_pre: bool) -> pd.DataFrame:
    """Telemetry-first column order: identity → archetype → core KPIs → blocks."""
    if is_pre:
        id_cols = ["Model", "telemetry_snapshot_date", "Model Type", "Use Case", "telemetry_archetype"]
        core = [
            "Ative Users (MAU)",
            "Accuracy %",
            "Error %",
            "Drift Index",
            "Escalation %",
            "Audit Coverage %",
            "Hallucination % (Test)",
        ]
    else:
        id_cols = ["updated_date", "model", "model_type", "use_case", "telemetry_archetype"]
        core = [
            "active_users_mau",
            "accuracy_pct",
            "error_pct",
            "drift_index",
            "hallucination_live_pct",
            "escalation_pct",
            "compliance_incidents",
            "audit_coverage_pct",
        ]
    archetype_specific = [
        "document_extraction_fidelity",
        "ocr_confidence_score",
        "field_validation_match_rate",
        "image_authenticity_confidence",
        "patch_embedding_drift_score",
        "transcription_wer_proxy_pct",
        "speaker_id_confidence",
        "intent_stability_score",
        "expected_hallucination_rate",
        "expected_grounding_score",
        "observed_grounding_score",
        "citation_coverage",
        "unsupported_claim_rate",
        "retrieval_failure_rate",
        "context_retrieval_hit_rate",
    ]
    reliability = [
        "expected_accuracy",
        "expected_error_rate",
        "expected_latency_p95_ms",
        "observed_latency_p95_ms",
        "false_positive_rate",
        "false_negative_rate",
        "output_consistency_score",
    ]
    drift_b = [
        "expected_drift_score",
        "drift_velocity",
        "feature_distribution_shift",
        "anomaly_frequency",
        "confidence_stability_score",
    ]
    biz = [
        "business_function",
        "deployment_scope",
        "regulatory_sensitivity",
        "model_owner",
        "control_owner",
        "review_frequency",
        "last_validation_days_ago",
    ]
    gov = [
        "minimum_required_audit_coverage",
        "decision_trace_completeness",
        "previous_escalation_count",
    ]
    comp = [
        "policy_violation_rate",
        "security_anomaly_count",
    ]
    cost = [
        "Cost per Interaction ($)" if is_pre else "cost_per_interaction",
        "Inference Cost ($)" if is_pre else "inference_cost",
        "expected_cost_per_1k_requests",
        "observed_cost_per_1k_requests",
        "manual_review_burden",
    ]
    if not is_pre:
        cost = [
            "cost_per_interaction",
            "inference_cost",
            "roi_multiple",
            "expected_cost_per_1k_requests",
            "observed_cost_per_1k_requests",
            "manual_review_burden",
        ]
    pre_rest = [
        "Bias Gap %",
        "Red Team Pass %",
    ]
    order = id_cols + core + archetype_specific + reliability + drift_b + biz + gov + comp + cost
    if is_pre:
        order = order + pre_rest
    seen: list[str] = []
    for c in order:
        if c in df.columns and c not in seen:
            seen.append(c)
    rest = [c for c in df.columns if c not in seen]
    return df[seen + rest]


def build_enriched_pre(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    post_key = post.set_index("model")
    rows = []
    for _, r in pre.iterrows():
        mid = str(r["Model"])
        pr = post_key.loc[mid] if mid in post_key.index else None
        p_acc = float(r["Accuracy %"]) if pd.notna(r["Accuracy %"]) else None
        p_err = float(r["Error %"])
        p_drift = float(r["Drift Index"])
        p_esc = float(r["Escalation %"])
        p_audit = float(r["Audit Coverage %"])
        p_ci = (
            float(pr["compliance_incidents"])
            if pr is not None and pd.notna(pr["compliance_incidents"])
            else 0.0
        )
        p_cost = float(r["Cost per Interaction ($)"])
        p_inf = float(r["Inference Cost ($)"])
        hall = r["Hallucination % (Test)"]
        hall_v = float(hall) if pd.notna(hall) else None
        mt = str(r["Model Type"])
        uc = str(r["Use Case"])
        arch = infer_telemetry_archetype(mt, uc)
        tier = compute_stress_tier_from_metrics(
            p_drift, p_err, p_esc, p_audit, p_acc, p_ci
        )
        dep = _deployment_scope(mt, uc)
        ctx = EnrichContext(
            model_id=mid,
            tier=tier,
            archetype=arch,
            use_case=uc,
            model_type=mt,
            accuracy_pre=p_acc,
            accuracy_post=p_acc,
            error_post=p_err,
            drift_post=p_drift,
            escalation_post=p_esc,
            audit_post=p_audit,
            compliance_incidents=p_ci,
            cost_per_interaction=p_cost,
            inference_cost=p_inf,
            hallucination_live=hall_v,
            deployment_scope=dep,
        )
        ex = enrich_row(ctx)
        row_dict = {**r.to_dict(), **ex}
        rows.append(row_dict)
    df = pd.DataFrame(rows)
    df = strip_risk_columns(df)
    df = strip_label_proxy_columns(df)
    df["telemetry_snapshot_date"] = df["Model"].astype(str).map(
        lambda m: telemetry_snapshot_timestamp(m, "pre")
    )
    return order_columns(df, is_pre=True)


def build_enriched_post(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    pre_key = pre.set_index("Model")
    rows = []
    for _, r in post.iterrows():
        mid = str(r["model"])
        pre_row = pre_key.loc[mid] if mid in pre_key.index else None
        acc_pre = float(pre_row["Accuracy %"]) if pre_row is not None else None
        acc = float(r["accuracy_pct"]) if pd.notna(r["accuracy_pct"]) else None
        err = float(r["error_pct"])
        drift = float(r["drift_index"])
        esc = float(r["escalation_pct"])
        audit = float(r["audit_coverage_pct"])
        ci = float(r["compliance_incidents"])
        mt = str(r["model_type"])
        uc = str(r["use_case"])
        arch = infer_telemetry_archetype(mt, uc)
        tier = compute_stress_tier_from_metrics(drift, err, esc, audit, acc, ci)
        dep = _deployment_scope(mt, uc)
        ctx = EnrichContext(
            model_id=mid,
            tier=tier,
            archetype=arch,
            use_case=uc,
            model_type=mt,
            accuracy_pre=acc_pre,
            accuracy_post=acc,
            error_post=err,
            drift_post=drift,
            escalation_post=esc,
            audit_post=audit,
            compliance_incidents=ci,
            cost_per_interaction=float(r["cost_per_interaction"]),
            inference_cost=float(r["inference_cost"]),
            hallucination_live=(
                float(r["hallucination_live_pct"])
                if pd.notna(r["hallucination_live_pct"])
                else None
            ),
            deployment_scope=dep,
        )
        ex = enrich_row(ctx)
        rows.append({**r.to_dict(), **ex})
    df = pd.DataFrame(rows)
    df = strip_risk_columns(df)
    df = strip_label_proxy_columns(df)
    df["updated_date"] = df["model"].astype(str).map(
        lambda m: telemetry_snapshot_timestamp(m, "post")
    )
    return order_columns(df, is_pre=False)


def copy_dashboard(path_in: Path, writer: pd.ExcelWriter, name: str) -> None:
    df = pd.read_excel(path_in, sheet_name=name, header=None)
    df.to_excel(writer, sheet_name=name, index=False, header=False)


def profile_summary(
    pre: pd.DataFrame, post: pd.DataFrame, *, synthetic: bool = True
) -> str:
    lines = []
    src = "synthetic_fleet.py (50 models)" if synthetic else str(SOURCE_XLSX.name)
    lines.append(f"=== SOURCE: {src} ===\n")
    lines.append("--- Pre-Go-Live Monitoring Report ---")
    lines.append(f"Rows: {len(pre)}, Columns: {len(pre.columns)}")
    lines.append("\n--- Post-Go-Live (parsed) ---")
    lines.append(f"Rows: {len(post)}")
    lines.append("\n--- Enriched export omits (legacy risk labels) ---")
    lines.append(f"  {sorted(RISK_COLUMNS_DROP)}")
    lines.append("\n--- Enriched export omits (label proxies / prediction targets) ---")
    lines.append(f"  {sorted(LABEL_PROXY_COLUMNS)}")
    lines.append("\n--- Telemetry stress tier (internal only) from observables ---")
    lines.append("  drift, error, escalation, audit, accuracy, compliance_incidents")
    lines.append("\n--- Archetypes ---")
    lines.append("  " + ", ".join(get_args(TelemetryArchetype)))
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build enriched monitoring workbook")
    ap.add_argument(
        "--legacy-source",
        action="store_true",
        help="Use Monitoring Report.xlsx (25 models) instead of synthetic 50-model fleet",
    )
    args = ap.parse_args()
    use_syn = not args.legacy_source

    pre, post = load_pre_post(use_synthetic_fleet=use_syn)
    print(profile_summary(pre, post, synthetic=use_syn))

    enriched_pre = build_enriched_pre(pre, post)
    enriched_post = build_enriched_post(pre, post)
    for name, df in ("Pre", enriched_pre), ("Post", enriched_post):
        leaked = [c for c in df.columns if c in RISK_COLUMNS_DROP | LABEL_PROXY_COLUMNS]
        assert not leaked, f"{name}: leaked columns: {leaked}"

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        enriched_pre.to_excel(writer, sheet_name="Pre-Go-Live Monitoring Report", index=False)
        enriched_post.to_excel(writer, sheet_name="Post-Go-Live Monitoring Report", index=False)
        copy_dashboard(SOURCE_XLSX, writer, "Pre-Go-Live-Dashboard")
        copy_dashboard(SOURCE_XLSX, writer, "Post-Go-Live-Dashboard")

    md = f"""# Enrichment summary

## Outputs

- `{OUT_XLSX.name}` — enriched model sheets with **archetype-aware telemetry**; **risk tier / risk status columns are omitted** so you can train or rules-engine risk labels later from observables.
- Dashboard sheets are copied unchanged from `{SOURCE_XLSX.name}`.

## Telemetry archetypes

Each row includes `telemetry_archetype` (`tabular_ml`, `time_series`, `vision_cnn`, `vision_document`, `sequence_anomaly`, `nlp_classifier`, `speech_asr`, `reinforcement_learning`, `hybrid_rules_ml`, `llm_rag`, `llm_text`, `agentic_system`).

Only the block that matches the archetype is populated; others are `NaN` (e.g. LLM narrative fields for tabular models; document OCR fields for ViT KYC; WER proxy for speech).

## Omitted columns (feature-safe export)

**Legacy risk labels:** `risk_status`, `risk_tier`, `Risk Tier` — not written.

**Label proxies** (overlap intended predictions such as risk level, issue flags, human review, remediation action):  
`human_review_required`, `audit_readiness_score`, `sensitive_data_exposure_risk`, `evidence_linked_flag`, `concept_shift_flag`, `prompt_injection_flag`, `access_control_review_status`, `roi_pressure_flag`, `unresolved_audit_findings`, `remediation_cost_estimate_usd`, `reviewer_override_rate`, `customer_impact`, `compliance_incident_count_enriched` — not written. Use raw `compliance_incidents` where present.

## Internal stress shaping

Synthetic fields are scaled using **observable KPIs only** (drift, error, escalation, audit coverage, accuracy, compliance incidents), not using the removed risk labels.

## Row counts

- Pre-Go-Live: {len(enriched_pre)}
- Post-Go-Live: {len(enriched_post)}
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"\nWrote: {OUT_XLSX}")
    print(f"Wrote: {OUT_MD}")
    print("\nSample archetypes (Post-Go-Live):")
    print(enriched_post[["model", "telemetry_archetype"]].to_string(index=False))


if __name__ == "__main__":
    main()
