"""
Deterministic 50-model synthetic fleet: diverse archetypes × risk scenarios.

Scenarios drive post-go-live observables so governance_engine stress + breach
logic yields a spread from low → critical without exporting legacy risk labels.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np
import pandas as pd

Scenario = Literal["stable", "mild", "high_stress", "critical_telemetry"]

FLEET_SIZE = 50
GLOBAL = 20260410

# (model_type, use_case, risk_tier_L)
MODEL_BLUEPRINTS: list[tuple[str, str, str]] = [
    ("Logistic Regression", "Credit risk PD model", "L3 (Low)"),
    ("XGBoost", "Fraud detection", "L2 (Medium)"),
    ("Random Forest", "AML alert prioritization", "L2 (Medium)"),
    ("Time Series ARIMA", "Liquidity forecasting", "L3 (Low)"),
    ("Gradient Boosting", "Loan prepayment prediction", "L3 (Low)"),
    ("CNN", "Check image fraud", "L2 (Medium)"),
    ("LSTM", "Transaction anomaly detection", "L3 (Low)"),
    ("NLP BERT", "Complaint classification", "L3 (Low)"),
    ("Transformer", "Document summarization", "L3 (Low)"),
    ("GPT fine-tuned", "Policy drafting", "L3 (Low)"),
    ("LLM RAG", "Regulatory Q&A", "L2 (Medium)"),
    ("LLM RAG", "Treasury research assistant", "L2 (Medium)"),
    ("LLM internal", "Code assistant", "L2 (Medium)"),
    ("LLM customer email", "RM productivity drafting", "L3 (Low)"),
    ("Hybrid ML + Rules", "Sanctions screening", "L3 (Low)"),
    ("Reinforcement Learning", "Trading strategy optimization", "L2 (Medium)"),
    ("LLM with tools", "Risk report generation", "L2 (Medium)"),
    ("Vision Transformer", "KYC document validation", "L2 (Medium)"),
    ("Tabular DL", "Cross-sell prediction", "L2 (Medium)"),
    ("AutoML", "SME underwriting", "L2 (Medium)"),
    ("Agentic workflow", "Client onboarding assistant", "L1 (High)"),
    ("Multi-agent system", "Ops exception handling", "L2 (Medium)"),
    ("LLM decision assist", "Credit memo drafting", "L2 (Medium)"),
    ("Speech-to-text AI", "Call transcription", "L2 (Medium)"),
    ("Agentic research AI", "Market intelligence synthesis", "L2 (Medium)"),
    ("ElasticNet", "Loss-given-default estimation", "L2 (Medium)"),
    ("LightGBM", "Merchant fraud scoring", "L2 (Medium)"),
    ("Prophet", "Cash flow forecasting", "L3 (Low)"),
    ("Isolation Forest", "Payment anomaly scoring", "L3 (Low)"),
    ("Graph neural net", "Counterparty network risk", "L2 (Medium)"),
    ("LLM RAG", "Legal clause retrieval", "L1 (High)"),
    ("Small LLM", "Internal FAQ bot", "L3 (Low)"),
    ("Vision CNN", "Signature verification", "L2 (Medium)"),
    ("Gradient Boosting", "Deposit attrition", "L3 (Low)"),
    ("Logistic Regression", "Insurance propensity", "L3 (Low)"),
    ("Transformer encoder", "Named entity extraction", "L2 (Medium)"),
    ("LLM with tools", "Regulatory filing assistant", "L1 (High)"),
    ("Agentic workflow", "Trade surveillance triage", "L1 (High)"),
    ("RAG + reranker", "Policy exception lookup", "L2 (Medium)"),
    ("Time Series ARIMA", "NII sensitivity", "L3 (Low)"),
    ("XGBoost", "Card not-present fraud", "L2 (Medium)"),
    ("Random Forest", "Collections prioritization", "L3 (Low)"),
    ("Speech-to-text AI", "Advisor meeting notes", "L3 (Low)"),
    ("Vision Transformer", "Invoice OCR validation", "L2 (Medium)"),
    ("Multi-agent system", "Cyber alert summarization", "L2 (Medium)"),
    ("LLM RAG", "Model risk FAQ", "L2 (Medium)"),
    ("Tabular DL", "Next-best-offer", "L3 (Low)"),
    ("Reinforcement Learning", "Liquidity buffer policy", "L2 (Medium)"),
    ("NLP BERT", "Ticket routing", "L3 (Low)"),
    ("Hybrid ML + Rules", "PEP screening", "L2 (Medium)"),
]


def _rng(model_id: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"{GLOBAL}:{model_id}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h % (2**32))


def _scenario_list() -> list[Scenario]:
    """Target spread: many green, meaningful medium/high/critical."""
    base: list[Scenario] = (
        ["stable"] * 24
        + ["mild"] * 14
        + ["high_stress"] * 8
        + ["critical_telemetry"] * 4
    )
    assert len(base) == FLEET_SIZE
    rng = np.random.default_rng(GLOBAL)
    order = rng.permutation(FLEET_SIZE)
    return [base[i] for i in order]


def _sample_post_metrics(scenario: Scenario, model_id: str) -> dict[str, Any]:
    r = _rng(model_id + ":post")
    if scenario == "stable":
        acc = float(r.uniform(88.0, 96.5))
        err = float(r.uniform(1.4, 4.2))
        drift = float(r.uniform(0.02, 0.09))
        esc = float(r.uniform(2.0, 9.0))
        audit = float(r.uniform(96.0, 99.6))
        ci = int(r.integers(0, 2))
        af = int(r.integers(0, 2))
        pvr = float(r.uniform(0.0004, 0.002))
        sac = int(r.integers(0, 2))
        mrb = float(r.uniform(0.028, 0.072))
        hall = r.uniform(2.0, 7.0) if r.random() > 0.45 else np.nan
    elif scenario == "mild":
        acc = float(r.uniform(81.0, 90.0))
        err = float(r.uniform(4.0, 7.5))
        drift = float(r.uniform(0.08, 0.15))
        esc = float(r.uniform(9.0, 17.0))
        audit = float(r.uniform(93.0, 97.5))
        ci = int(r.integers(0, 2))
        af = int(r.integers(1, 4))
        pvr = float(r.uniform(0.002, 0.005))
        sac = int(r.integers(1, 3))
        mrb = float(r.uniform(0.065, 0.115))
        hall = r.uniform(4.0, 10.0) if r.random() > 0.35 else np.nan
    elif scenario == "high_stress":
        acc = float(r.uniform(74.0, 86.0))
        err = float(r.uniform(6.5, 11.0))
        drift = float(r.uniform(0.14, 0.22))
        esc = float(r.uniform(17.0, 26.0))
        audit = float(r.uniform(89.5, 94.5))
        ci = int(r.integers(1, 3))
        af = int(r.integers(3, 8))
        pvr = float(r.uniform(0.0045, 0.009))
        sac = int(r.integers(2, 5))
        mrb = float(r.uniform(0.11, 0.20))
        hall = r.uniform(7.0, 14.0) if r.random() > 0.25 else np.nan
    else:  # critical_telemetry
        acc = float(r.uniform(58.0, 76.0))
        err = float(r.uniform(11.0, 18.5))
        drift = float(r.uniform(0.22, 0.36))
        esc = float(r.uniform(27.0, 42.0))
        audit = float(r.uniform(82.0, 90.0))
        ci = int(r.integers(2, 5))
        af = int(r.integers(6, 14))
        pvr = float(r.uniform(0.009, 0.022))
        sac = int(r.integers(4, 10))
        mrb = float(r.uniform(0.19, 0.38))
        hall = r.uniform(12.0, 26.0) if r.random() > 0.15 else np.nan

    mau = int(r.integers(45, 480))
    cost = float(r.uniform(0.08, 1.15))
    inf = float(r.uniform(0.07, 1.12))
    roi = float(r.uniform(1.2, 6.5))

    return {
        "active_users_mau": mau,
        "accuracy_pct": round(acc, 1),
        "error_pct": round(err, 2),
        "hallucination_live_pct": round(float(hall), 2) if hall == hall else np.nan,
        "drift_index": round(drift, 3),
        "escalation_pct": int(round(esc)),
        "compliance_incidents": ci,
        "audit_coverage_pct": int(round(audit)),
        "cost_per_interaction": round(cost, 2),
        "inference_cost": round(inf, 2),
        "roi_multiple": f"{roi:.1f}x",
        # seed enriched-only path fields used by stress (enrich_row fills the rest)
        "anomaly_frequency": af,
        "policy_violation_rate": round(pvr, 5),
        "security_anomaly_count": sac,
        "manual_review_burden": round(mrb, 3),
    }


def _sample_pre_metrics(post: dict[str, Any], scenario: Scenario, model_id: str) -> dict[str, Any]:
    """Pre-go-live: slightly better or aligned baseline vs post."""
    r = _rng(model_id + ":pre")
    acc_p = float(post["accuracy_pct"])
    err_p = float(post["error_pct"])
    drift_p = float(post["drift_index"])
    esc_p = float(post["escalation_pct"])
    audit_p = float(post["audit_coverage_pct"])

    if scenario == "stable":
        acc = acc_p + float(r.uniform(0.3, 2.2))
        err = max(0.8, err_p - float(r.uniform(0.2, 1.2)))
        drift = max(0.015, drift_p - float(r.uniform(0.01, 0.04)))
        esc = max(1.0, esc_p - float(r.uniform(0.5, 3.0)))
        audit = min(99.9, audit_p + float(r.uniform(0.2, 1.5)))
    elif scenario == "mild":
        acc = acc_p + float(r.uniform(0.2, 1.5))
        err = max(0.9, err_p - float(r.uniform(0.1, 0.8)))
        drift = max(0.02, drift_p - float(r.uniform(0.005, 0.03)))
        esc = max(2.0, esc_p - float(r.uniform(0.5, 2.5)))
        audit = min(99.5, audit_p + float(r.uniform(0.1, 1.0)))
    elif scenario == "high_stress":
        acc = acc_p + float(r.uniform(-1.0, 1.5))
        err = err_p - float(r.uniform(-0.5, 0.8))
        drift = max(0.02, drift_p - float(r.uniform(0, 0.02)))
        esc = max(2.0, esc_p - float(r.uniform(0, 2.0)))
        audit = min(99.0, audit_p + float(r.uniform(0, 0.8)))
    else:
        acc = acc_p + float(r.uniform(-4.0, 2.0))
        err = err_p - float(r.uniform(-1.5, 1.0))
        drift = max(0.02, drift_p - float(r.uniform(0, 0.03)))
        esc = max(2.0, esc_p - float(r.uniform(0, 3.0)))
        audit = min(99.0, audit_p + float(r.uniform(-0.5, 0.5)))

    mau = int(post["active_users_mau"] * r.uniform(0.85, 1.08))
    return {
        "Ative Users (MAU)": mau,
        "Accuracy %": int(round(acc)),
        "Error %": round(err, 2),
        "Drift Index": round(drift, 3),
        "Escalation %": int(round(esc)),
        "Audit Coverage %": int(round(audit)),
        "Cost per Interaction ($)": post["cost_per_interaction"] * r.uniform(0.92, 1.06),
        "Bias Gap %": round(float(r.uniform(1.2, 4.2)), 2),
        "Red Team Pass %": round(float(r.uniform(78.0, 96.0)), 1) if r.random() > 0.35 else np.nan,
        "Hallucination % (Test)": round(float(r.uniform(3.0, 16.0)), 1)
        if r.random() > 0.4
        else np.nan,
        "Inference Cost ($)": post["inference_cost"] * r.uniform(0.9, 1.05),
    }


def build_synthetic_pre_post() -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = _scenario_list()
    pre_rows: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []

    for i in range(FLEET_SIZE):
        mid = f"M{i + 1:02d}"
        mt, uc, tier = MODEL_BLUEPRINTS[i]
        scen = scenarios[i]
        post_m = _sample_post_metrics(scen, mid)
        pre_m = _sample_pre_metrics(post_m, scen, mid)

        post_rows.append(
            {
                "updated_date": np.nan,
                "model": mid,
                "model_type": mt,
                "use_case": uc,
                "risk_tier": tier,
                "active_users_mau": post_m["active_users_mau"],
                "accuracy_pct": post_m["accuracy_pct"],
                "error_pct": post_m["error_pct"],
                "hallucination_live_pct": post_m["hallucination_live_pct"],
                "drift_index": post_m["drift_index"],
                "escalation_pct": post_m["escalation_pct"],
                "compliance_incidents": post_m["compliance_incidents"],
                "audit_coverage_pct": post_m["audit_coverage_pct"],
                "cost_per_interaction": post_m["cost_per_interaction"],
                "inference_cost": post_m["inference_cost"],
                "roi_multiple": post_m["roi_multiple"],
            }
        )
        pre_rows.append(
            {
                "Model": mid,
                "Model Type": mt,
                "Use Case": uc,
                "Risk Tier": tier,
                **pre_m,
            }
        )

    pre_df = pd.DataFrame(pre_rows)
    post_df = pd.DataFrame(post_rows)

    return pre_df, post_df


