"""Aggregate dimension severities into risk score, level, and governance action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from governance_engine.evaluators import Breach, DimensionResult, run_all_evaluators
from governance_engine.expectations import load_expectations, merged_bands_for_archetype
from governance_engine.playbooks import human_review_gate, select_action
from governance_engine.stress import compute_observability_stress, traffic_light


@dataclass
class GovernanceDecision:
    model_id: str
    telemetry_archetype: str
    risk_score: float
    breach_component_score: float
    observability_stress_score: float
    risk_level: str
    risk_traffic_light: str
    issue_categories: list[str]
    recommended_action_id: str
    recommended_action_text: str
    human_review_required: bool
    human_review_priority: str
    dimension_results: list[dict[str, Any]]
    rationale_summary: str
    snapshot_date: str | None = None
    use_case: str | None = None
    business_function: str | None = None


def _risk_level_from_score(score: float, cutoffs: dict[str, float]) -> str:
    low = float(cutoffs.get("low", 30))
    med = float(cutoffs.get("medium", 55))
    high = float(cutoffs.get("high", 75))
    if score < low:
        return "low"
    if score < med:
        return "medium"
    if score < high:
        return "high"
    return "critical"


def _breach_to_dict(b: Breach) -> dict[str, Any]:
    return {
        "metric": b.metric,
        "observed": b.observed,
        "expected": b.expected,
        "detail": b.detail,
    }


def _dim_to_dict(d: DimensionResult) -> dict[str, Any]:
    return {
        "dimension": d.dimension,
        "severity": d.severity,
        "skipped": d.skipped,
        "skip_reason": d.skip_reason,
        "breaches": [_breach_to_dict(b) for b in d.breaches],
    }


def run_governance_for_row(
    row: pd.Series,
    config: dict[str, Any] | None = None,
) -> GovernanceDecision:
    if config is None:
        config = load_expectations()

    archetype = str(row.get("telemetry_archetype", "tabular_ml"))
    bands = merged_bands_for_archetype(config, archetype)
    results = run_all_evaluators(row, bands, archetype)

    weights = config.get("weights") or {}
    w_map = {
        "reliability": float(weights.get("reliability", 0.25)),
        "narrative_assurance": float(weights.get("narrative_assurance", 0.25)),
        "compliance_security": float(weights.get("compliance_security", 0.25)),
        "auditability": float(weights.get("auditability", 0.25)),
    }
    breach_s = sum(w_map.get(r.dimension, 0.0) * r.severity for r in results)
    breach_score = round(min(100.0, breach_s * 100.0), 2)

    stress_cfg = config.get("stress_index") or {}
    stress_idx = compute_observability_stress(row, stress_cfg if isinstance(stress_cfg, dict) else {})
    stress_score = round(stress_idx * 100.0, 2)

    blend = config.get("score_blend") or {}
    w_b = float(blend.get("breach", 0.5))
    w_s = float(blend.get("stress", 0.5))
    w_sum = w_b + w_s or 1.0
    w_b, w_s = w_b / w_sum, w_s / w_sum

    risk_score = round(min(100.0, w_b * breach_score + w_s * stress_score), 2)

    cutoffs = config.get("risk_level_cutoffs") or {}
    risk_level = _risk_level_from_score(risk_score, cutoffs)
    risk_tl = traffic_light(risk_level)

    active = [r for r in results if r.severity > 0 and not r.skipped]
    issue_categories = sorted({r.dimension for r in active})
    primary = max(active, key=lambda x: x.severity).dimension if active else None

    action_id, action_text = select_action(risk_level, primary)
    max_sev = max((r.severity for r in results), default=0.0)
    hitl, prio = human_review_gate(risk_level, max_sev)

    rationale = (
        f"risk_score={risk_score} = {w_b:.2f}*breach({breach_score}) + {w_s:.2f}*stress({stress_score}); "
        f"dimensions: {', '.join(f'{r.dimension}={r.severity:.2f}' for r in results)}; "
        f"primary_breach={primary or 'none'}."
    )

    snap = row.get("updated_date")
    if snap is not None and pd.notna(snap):
        snap_out = snap.isoformat() if hasattr(snap, "isoformat") else str(snap)
    else:
        snap_out = None

    return GovernanceDecision(
        model_id=str(row.get("model", "")),
        telemetry_archetype=archetype,
        risk_score=risk_score,
        breach_component_score=breach_score,
        observability_stress_score=stress_score,
        risk_level=risk_level,
        risk_traffic_light=risk_tl,
        issue_categories=issue_categories,
        recommended_action_id=action_id,
        recommended_action_text=action_text,
        human_review_required=hitl,
        human_review_priority=prio,
        dimension_results=[_dim_to_dict(r) for r in results],
        rationale_summary=rationale,
        snapshot_date=snap_out,
        use_case=str(row["use_case"]) if pd.notna(row.get("use_case")) else None,
        business_function=str(row["business_function"])
        if pd.notna(row.get("business_function"))
        else None,
    )


def decision_to_jsonable(d: GovernanceDecision) -> dict[str, Any]:
    out = asdict(d)
    return out
