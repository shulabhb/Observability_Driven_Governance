"""Map risk level + primary issue dimension to governance actions (POC)."""

from __future__ import annotations

# action_id -> description
PLAYBOOKS: dict[str, str] = {
    "MONITOR": "Continue scheduled monitoring; no governance change.",
    "TUNE_SAMPLING": "Increase offline evaluation sampling and drift checks for 2 weeks.",
    "NARRATIVE_REVIEW": "Run Narrative Assurance review (grounding / citations) before next release.",
    "RELIABILITY_REVIEW": "Open technology reliability review; validate data and model artifacts.",
    "COMPLIANCE_ESCALATION": "Notify Compliance / 2LoD; hold promotion until incident cleared.",
    "AUDIT_REMEDIATION": "Remediate audit/trace gaps; attach evidence pack to model inventory.",
    "FULL_ESCALATION": "Halt automated promotion; executive + risk committee triage with human sign-off.",
}


def select_action(risk_level: str, primary_category: str | None) -> tuple[str, str]:
    """
    primary_category: dimension name with highest severity, or None if none.
    Returns (action_id, action_text).
    """
    cat = primary_category or "none"
    rl = risk_level.lower()

    if rl in ("low",):
        return "MONITOR", PLAYBOOKS["MONITOR"]

    if rl == "medium":
        if cat == "narrative_assurance":
            return "NARRATIVE_REVIEW", PLAYBOOKS["NARRATIVE_REVIEW"]
        if cat == "compliance_security":
            return "COMPLIANCE_ESCALATION", PLAYBOOKS["COMPLIANCE_ESCALATION"]
        if cat == "auditability":
            return "AUDIT_REMEDIATION", PLAYBOOKS["AUDIT_REMEDIATION"]
        if cat == "reliability":
            return "RELIABILITY_REVIEW", PLAYBOOKS["RELIABILITY_REVIEW"]
        return "TUNE_SAMPLING", PLAYBOOKS["TUNE_SAMPLING"]

    if rl == "high":
        if cat == "compliance_security":
            return "COMPLIANCE_ESCALATION", PLAYBOOKS["COMPLIANCE_ESCALATION"]
        if cat == "auditability":
            return "AUDIT_REMEDIATION", PLAYBOOKS["AUDIT_REMEDIATION"]
        if cat == "narrative_assurance":
            return "NARRATIVE_REVIEW", PLAYBOOKS["NARRATIVE_REVIEW"]
        return "RELIABILITY_REVIEW", PLAYBOOKS["RELIABILITY_REVIEW"]

    # critical
    return "FULL_ESCALATION", PLAYBOOKS["FULL_ESCALATION"]


def human_review_gate(risk_level: str, max_severity: float) -> tuple[bool, str]:
    if risk_level.lower() in ("critical",):
        return True, "P1"
    if risk_level.lower() == "high":
        return True, "P2"
    if risk_level.lower() == "medium" and max_severity >= 0.55:
        return True, "P3"
    return False, "none"
