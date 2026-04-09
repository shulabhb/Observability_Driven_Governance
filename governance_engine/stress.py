"""
Observability stress index (0–1) from raw telemetry only — no policy breach logic.

Used to blend with breach-based severity so risk levels spread on healthy-looking fleets
where explicit band breaches are rare.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


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


def compute_observability_stress(
    row: pd.Series, stress_cfg: dict[str, Any] | None = None
) -> float:
    """
    Return stress in [0, 1] from normalized telemetry signals.
    Config shape (optional):
      terms:
        drift_index: { cap: 0.22, weight: 0.15 }
        error_pct: { cap: 10, weight: 0.15 }
        ...
    If config missing, uses POC defaults tuned to enriched_monitoring_report ranges.
    """
    cfg = stress_cfg or {}
    terms = cfg.get("terms")
    if not isinstance(terms, dict) or not terms:
        terms = {
            "drift_index": {"cap": 0.22, "weight": 0.16},
            "error_pct": {"cap": 10.0, "weight": 0.16},
            "escalation_pct": {"cap": 32.0, "weight": 0.16},
            "audit_coverage_gap": {"weight": 0.12, "gap_scale": 6.0},
            "compliance_incidents": {"cap": 3.0, "weight": 0.10},
            "anomaly_frequency": {"cap": 12.0, "weight": 0.10},
            "policy_violation_rate": {"cap": 0.006, "weight": 0.08},
            "security_anomaly_count": {"cap": 6.0, "weight": 0.08},
            "manual_review_burden": {"cap": 0.22, "weight": 0.04},
        }

    total_w = 0.0
    acc = 0.0

    for name, spec in terms.items():
        if not isinstance(spec, dict):
            continue
        w = float(spec.get("weight", 0.0))
        if w <= 0:
            continue

        if name == "audit_coverage_gap":
            ac = _num(row, "audit_coverage_pct")
            if ac is None:
                continue
            scale = float(spec.get("gap_scale", 6.0))
            # Higher stress when coverage below 100
            norm = _clip01((100.0 - ac) / scale)
        else:
            cap = float(spec.get("cap", 1.0))
            if cap <= 0:
                continue
            val = _num(row, name)
            if val is None:
                continue
            norm = _clip01(val / cap)

        acc += w * norm
        total_w += w

    if total_w <= 0:
        return 0.0
    return _clip01(acc / total_w)


def traffic_light(risk_level: str) -> str:
    """Business-facing palette mapping (POC)."""
    m = {
        "low": "green",
        "medium": "yellow",
        "high": "orange",
        "critical": "red",
    }
    return m.get(risk_level.lower(), "green")
