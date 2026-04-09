"""
POC Streamlit dashboard: Home (fleet risk) → Models (cards) → Analyze (deep dive).

Run: streamlit run governance_dashboard.py
"""

from __future__ import annotations

import html as html_module
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance_engine.data_bridge import (  # noqa: E402
    DEFAULT_ENRICHED_PATH,
    load_enriched_post,
)

# POC: one fixed enriched workbook for every viewer (not configurable in the UI).
POC_ENRICHED_PATH = str(Path(DEFAULT_ENRICHED_PATH).resolve())
from governance_engine.engine import decision_to_jsonable, run_governance_for_row  # noqa: E402
from governance_engine.expectations import load_expectations  # noqa: E402
from governance_engine.live_sim.merge import build_effective_dataframe  # noqa: E402
from governance_engine.live_sim import metrics as live_metrics  # noqa: E402
from governance_engine.live_sim.simulator import (  # noqa: E402
    ensure_seeded,
    fetch_recent_events,
    fetch_series,
    is_running,
    register_workbook_path,
    reset_simulation,
    set_running,
)
from governance_engine.live_sim.store import connect, init_schema  # noqa: E402
from governance_engine.telemetry_display import (  # noqa: E402
    format_metric_value,
    format_snapshot_value,
    humanize_column_name,
    narrative_metric_keys_for_arch,
    primary_metrics_for_row,
    reliability_metric_keys_for_arch,
)

LEVEL_COLOR = {
    "low": "#2e7d32",
    "medium": "#f57c00",
    "high": "#ef6c00",
    "critical": "#c62828",
}

# Pastel fill, contrasting text (no reliance on color names in the UI copy)
LEVEL_TITLE = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
LEVEL_PILL_STYLE = {
    "low": ("#e8f5e9", "#1b5e20"),
    "medium": ("#fff8e1", "#e65100"),
    "high": ("#ffe0b2", "#bf360c"),
    "critical": ("#ffcdd2", "#b71c1c"),
}

TRAFFIC_LABEL = {
    "green": "Steady",
    "yellow": "Caution",
    "orange": "Elevated",
    "red": "Alert",
}
TRAFFIC_PILL_STYLE = {
    "green": ("#c8e6c9", "#1b5e20"),
    "yellow": ("#fff9c4", "#f57f17"),
    "orange": ("#ffe0b2", "#e65100"),
    "red": ("#ffcdd2", "#b71c1c"),
}

LEVEL_SORT = {"critical": 0, "high": 1, "medium": 2, "low": 3}

RISK_LEVEL_OPTIONS = ["critical", "high", "medium", "low"]

# Risk-level presets for Models tab cards only
FILTER_PRESETS: dict[str, list[str]] = {
    "All": ["critical", "high", "medium", "low"],
    "Elevated": ["critical", "high", "medium"],
    "Severe": ["critical", "high"],
    "Critical only": ["critical"],
    "Low risk": ["low"],
}

HOME_ACTION_OPTIONS = ("Any", "Needs action (non-MONITOR)", "Routine monitoring only")
HOME_OUTPERFORM_OPTIONS = ("Any", "Doing better than expected", "Not outperforming")
HOME_HITL_OPTIONS = ("Any", "HITL required", "No HITL")


def _model_sort_key(model_id: str) -> tuple[int, str]:
    """Sort M01…M50 (or similar) in numeric order; unknown ids sort last by string."""
    s = str(model_id).strip()
    if len(s) >= 2 and s[0].upper() == "M" and s[1:].isdigit():
        return (int(s[1:]), s)
    m = re.search(r"(\d+)$", s)
    if m:
        return (int(m.group(1)), s)
    return (10**9, s)


def _models_in_workbook_order(df: pd.DataFrame) -> list[str]:
    ids = df["model"].astype(str).unique().tolist()
    ids.sort(key=_model_sort_key)
    return ids


def _analyze_model_label(model_id: str) -> str:
    """Dropdown label: Model 1 (M01) … Model 50 (M50)."""
    s = str(model_id).strip()
    if len(s) >= 2 and s[0].upper() == "M" and s[1:].isdigit():
        n = int(s[1:])
        return f"Model {n} ({s})"
    return s


def _open_model_in_analyze(model_id: str) -> None:
    """Session-state update for card CTA; use as st.button(on_click=...) per Streamlit guidance."""
    st.session_state.analyze_pick = str(model_id).strip()
    st.session_state.nav_tab = "Analyze"


def _live_sim_start() -> None:
    c = connect()
    init_schema(c)
    register_workbook_path(c, POC_ENRICHED_PATH)
    set_running(c, True)
    c.close()


def _live_sim_stop() -> None:
    c = connect()
    init_schema(c)
    set_running(c, False)
    c.close()


def _live_sim_reset() -> None:
    c = connect()
    init_schema(c)
    df0 = load_enriched_post(Path(POC_ENRICHED_PATH))
    reset_simulation(c, df0, POC_ENRICHED_PATH)
    c.close()
    _load_data.clear()


def _set_nav_page(page: str) -> None:
    st.session_state.nav_tab = page


NAV_PAGES = ("Home", "Models", "Analyze", "Data source")


def _nav_rail_item(
    icon: str,
    label: str,
    page: str,
    *,
    key: str,
    active: bool,
    help_tt: str | None = None,
) -> None:
    """Minimal vertical nav cell: icon button, name below."""
    st.button(
        icon,
        key=key,
        use_container_width=False,
        on_click=_set_nav_page,
        args=(page,),
        type="primary" if active else "secondary",
        help=help_tt or label,
    )
    st.markdown(
        f'<p class="nav-rail-cute-label">{html_module.escape(label)}</p>',
        unsafe_allow_html=True,
    )


def _home_filter_popover_contents(model_type_opts: list[str], archetype_opts: list[str]) -> None:
    """Filter controls for Home (must run before filtered tables use session keys)."""
    st.caption("All criteria combine with **AND**.")
    st.multiselect(
        "Risk level",
        options=RISK_LEVEL_OPTIONS,
        key="home_level_filter",
    )
    st.multiselect(
        "Model type",
        options=model_type_opts,
        key="home_model_types",
        placeholder="All types",
    )
    st.multiselect(
        "Telemetry archetype",
        options=archetype_opts,
        key="home_archetypes",
        placeholder="All archetypes",
    )
    st.selectbox(
        "Action posture",
        options=HOME_ACTION_OPTIONS,
        key="home_action_posture",
    )
    st.selectbox(
        "vs expectations",
        options=HOME_OUTPERFORM_OPTIONS,
        key="home_outperform",
        help="Uses accuracy vs expected accuracy and error vs expected error when both are present.",
    )
    st.selectbox(
        "Human review (HITL)",
        options=HOME_HITL_OPTIONS,
        key="home_hitl_filter",
    )
    if st.button("Reset filters", key="home_filter_reset", use_container_width=True):
        st.session_state.home_level_filter = list(RISK_LEVEL_OPTIONS)
        st.session_state.home_model_types = []
        st.session_state.home_archetypes = []
        st.session_state.home_action_posture = HOME_ACTION_OPTIONS[0]
        st.session_state.home_outperform = HOME_OUTPERFORM_OPTIONS[0]
        st.session_state.home_hitl_filter = HOME_HITL_OPTIONS[0]
        st.rerun()


def _fleet_overview_score_box(label: str, value: float, *, scale: float = 100.0) -> None:
    """Small card: value / scale + progress toward full scale."""
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"**{value:.1f}** / **{scale:.0f}**")
        st.progress(min(max(value / scale, 0.0), 1.0))


NAV_RAIL_ENTRIES: tuple[tuple[str, str, str, str, str | None], ...] = (
    ("🏠", "Home", "Home", "rail_nav_home", None),
    ("📇", "Models", "Models", "rail_nav_models", None),
    ("🔬", "Analyze", "Analyze", "rail_nav_analyze", None),
    ("💾", "Data", "Data source", "rail_nav_datasource", "Workbook path & reload"),
)

# Analyze tab: grouped fields (order matches enriched workbook intent)
PROFILE_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Identity & registration",
        [
            "model",
            "model_type",
            "use_case",
            "telemetry_archetype",
            "updated_date",
        ],
    ),
    (
        "Scale & usage",
        ["active_users_mau"],
    ),
    (
        "Core production KPIs",
        [
            "accuracy_pct",
            "error_pct",
            "drift_index",
            "hallucination_live_pct",
            "escalation_pct",
            "compliance_incidents",
            "audit_coverage_pct",
        ],
    ),
    (
        "Modality-specific telemetry",
        [
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
        ],
    ),
    (
        "Reliability & latency targets",
        [
            "expected_accuracy",
            "expected_error_rate",
            "expected_latency_p95_ms",
            "observed_latency_p95_ms",
            "false_positive_rate",
            "false_negative_rate",
            "output_consistency_score",
        ],
    ),
    (
        "Drift & stability",
        [
            "expected_drift_score",
            "drift_velocity",
            "feature_distribution_shift",
            "anomaly_frequency",
            "confidence_stability_score",
        ],
    ),
    (
        "Business & ownership",
        [
            "business_function",
            "deployment_scope",
            "regulatory_sensitivity",
            "model_owner",
            "control_owner",
            "review_frequency",
            "last_validation_days_ago",
        ],
    ),
    (
        "Governance controls",
        [
            "minimum_required_audit_coverage",
            "decision_trace_completeness",
            "previous_escalation_count",
            "policy_violation_rate",
            "security_anomaly_count",
        ],
    ),
    (
        "Cost & economics",
        [
            "cost_per_interaction",
            "inference_cost",
            "roi_multiple",
            "expected_cost_per_1k_requests",
            "observed_cost_per_1k_requests",
            "manual_review_burden",
        ],
    ),
]


def _pill_span(bg: str, fg: str, text: str) -> str:
    return (
        f"<span style=\"display:inline-block;background:{bg};color:{fg};"
        f"padding:4px 12px;border-radius:999px;font-weight:600;font-size:0.78rem;"
        f'margin:2px 6px 2px 0;white-space:nowrap\">{text}</span>'
    )


def _orchestrator_risk_row_html(dec) -> str:
    tl = str(dec.risk_traffic_light).lower()
    rl = str(dec.risk_level).lower()
    ttxt = TRAFFIC_LABEL.get(tl, tl.title())
    rtxt = LEVEL_TITLE.get(rl, rl.title())
    tbg, tfg = TRAFFIC_PILL_STYLE.get(tl, ("#eceff1", "#333"))
    rbg, rfg = LEVEL_PILL_STYLE.get(rl, ("#eceff1", "#333"))
    return (
        "<div style=\"display:flex;flex-wrap:wrap;align-items:center;margin:4px 0 10px 0\">"
        f"{_pill_span(rbg, rfg, rtxt + ' risk')}"
        f"{_pill_span(tbg, tfg, ttxt + ' signal')}"
        "</div>"
    )


_TRAFFIC_BY_LABEL = {v: k for k, v in TRAFFIC_LABEL.items()}
_LEVEL_BY_TITLE = {v: k for k, v in LEVEL_TITLE.items()}


def _style_traffic_cell(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    raw = _TRAFFIC_BY_LABEL.get(s, s.lower())
    bg, fg = TRAFFIC_PILL_STYLE.get(raw, ("#f5f5f5", "#424242"))
    return f"background-color: {bg}; color: {fg}; font-weight: 600;"


def _style_level_cell(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    raw = _LEVEL_BY_TITLE.get(s, s.lower())
    bg, fg = LEVEL_PILL_STYLE.get(raw, ("#f5f5f5", "#424242"))
    return f"background-color: {bg}; color: {fg}; font-weight: 600;"


def _fleet_table_for_display(summary: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = summary.loc[:, [c for c in cols if c in summary.columns]].copy()
    if "risk_traffic_light" in out.columns:
        out["risk_traffic_light"] = out["risk_traffic_light"].map(
            lambda x: TRAFFIC_LABEL.get(str(x).lower(), x) if pd.notna(x) else x
        )
    if "risk_level" in out.columns:
        out["risk_level"] = out["risk_level"].map(
            lambda x: LEVEL_TITLE.get(str(x).lower(), x) if pd.notna(x) else x
        )
    return out


def _style_fleet_table(df: pd.DataFrame) -> object:
    if df.empty:
        return df.style
    sty = df.style
    if "risk_traffic_light" in df.columns:
        sty = sty.map(_style_traffic_cell, subset=["risk_traffic_light"])
    if "risk_level" in df.columns:
        sty = sty.map(_style_level_cell, subset=["risk_level"])
    try:
        sty = sty.hide(axis="index")
    except (AttributeError, TypeError):
        pass
    return sty


def _dimension_severity_chart(dim_df: pd.DataFrame) -> None:
    try:
        import altair as alt

        chart = (
            alt.Chart(dim_df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("dimension:N", sort="-y", title="Dimension"),
                y=alt.Y("severity:Q", title="Severity", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "severity:Q",
                    scale=alt.Scale(domain=[0, 50, 100], range=["#2e7d32", "#f9a825", "#c62828"]),
                    legend=alt.Legend(title="Severity"),
                ),
                tooltip=[
                    alt.Tooltip("dimension:N", title="Dimension"),
                    alt.Tooltip("severity:Q", format=".1f"),
                    alt.Tooltip("breaches:Q", title="Breaches"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(dim_df.set_index("dimension")["severity"], use_container_width=True)


def _section_metrics_dataframe(row: pd.Series, columns: list[str]) -> pd.DataFrame | None:
    rows_out: list[dict[str, str]] = []
    for c in columns:
        if c not in row.index:
            continue
        v = row.get(c)
        if pd.isna(v) or v == "":
            continue
        rows_out.append(
            {
                "Field": humanize_column_name(c),
                "Value": format_snapshot_value(c, v),
            }
        )
    if not rows_out:
        return None
    return pd.DataFrame(rows_out)


def _remaining_telemetry_dataframe(row: pd.Series) -> pd.DataFrame | None:
    used: set[str] = set()
    for _, cols in PROFILE_SECTIONS:
        used.update(cols)
    rows_out: list[dict[str, str]] = []
    for c in row.index:
        if c in used:
            continue
        v = row[c]
        if pd.isna(v) or v == "":
            continue
        rows_out.append(
            {
                "Field": humanize_column_name(str(c)),
                "Value": format_snapshot_value(str(c), v),
            }
        )
    if not rows_out:
        return None
    rows_out.sort(key=lambda r: r["Field"].lower())
    return pd.DataFrame(rows_out)


@st.cache_data
def _load_data(path_str: str) -> pd.DataFrame:
    return load_enriched_post(Path(path_str))


@st.cache_resource
def _cfg():
    return load_expectations()


def _render_live_monitoring_section(row: pd.Series) -> None:
    """Overlapping normalized time series for this model's available simulated metrics."""
    mcols = live_metrics.metrics_for_row(row)
    if not mcols:
        return
    mid = str(row.get("model", ""))
    label_map = {c: humanize_column_name(c) for c in mcols}
    st.markdown("##### Live monitoring (simulated)")
    st.caption(
        "Parameters match this model's workbook columns and archetype. "
        "Series are scaled to **0–100** per metric band. Requires the generator to be **running** (Data tab)."
    )

    ss_key = f"live_chart_metrics_{mid}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = list(mcols)
    else:
        pruned = [x for x in st.session_state[ss_key] if x in mcols]
        st.session_state[ss_key] = pruned if pruned else list(mcols)

    st.multiselect(
        "Metrics to show",
        options=mcols,
        key=ss_key,
        format_func=lambda c: label_map.get(c, c),
        help="Toggle which parameters appear on the chart. All available metrics are selected initially.",
    )

    @st.fragment(run_every=2)
    def _go() -> None:
        try:
            import altair as alt
        except ImportError:
            st.warning("Altair is required for live charts.")
            return
        selected = [x for x in st.session_state.get(ss_key, mcols) if x in mcols]
        if not selected:
            st.warning("Select at least one metric above.")
            return
        c = connect()
        init_schema(c)
        running = is_running(c)
        ser = fetch_series(c, mid, selected)
        c.close()
        if not running:
            st.caption("Start **Start generator** on the Data tab to stream KPIs here.")
            return
        if ser.empty:
            st.caption("Collecting points… use **Reset & re-seed** on the Data tab if the store is empty.")
            return
        ser = ser.copy()
        ser = ser[ser["metric"].isin(selected)]
        if ser.empty:
            st.caption("No data yet for the selected metrics.")
            return
        ser["t"] = pd.to_datetime(ser["ts"], unit="s", utc=True)

        def _norm_axis(r: pd.Series) -> float:
            lo, hi = live_metrics.bounds_for_metric(str(r["metric"]))
            span = (hi - lo) or 1.0
            return (float(r["value"]) - lo) / span * 100.0

        ser["y"] = ser.apply(_norm_axis, axis=1)
        ser["metric_label"] = ser["metric"].map(lambda m: label_map.get(m, m))
        chart = (
            alt.Chart(ser)
            .mark_line(interpolate="monotone")
            .encode(
                x=alt.X("t:T", title="Time (UTC)"),
                y=alt.Y("y:Q", title="Normalized (0–100)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("metric_label:N", title="Metric"),
                tooltip=[
                    alt.Tooltip("t:T", title="Time"),
                    alt.Tooltip("metric_label:N", title="Metric"),
                    alt.Tooltip("value:Q", format=".4f", title="Raw value"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(chart, use_container_width=True)

    _go()


def _doing_better_than_expected(row: pd.Series) -> bool:
    """
    True when every available observed-vs-expected pair is favorable:
    accuracy >= expected_accuracy and error <= expected_error_rate.
    """
    checks: list[bool] = []
    acc, ea = row.get("accuracy_pct"), row.get("expected_accuracy")
    if pd.notna(acc) and pd.notna(ea):
        try:
            checks.append(float(acc) >= float(ea))
        except (TypeError, ValueError):
            pass
    err, ee = row.get("error_pct"), row.get("expected_error_rate")
    if pd.notna(err) and pd.notna(ee):
        try:
            checks.append(float(err) <= float(ee))
        except (TypeError, ValueError):
            pass
    return bool(checks) and all(checks)


def _build_fleet_summary(df: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        d = run_governance_for_row(row, cfg)
        mt = str(row.get("model_type", "") or "").strip()
        rows.append(
            {
                "model": str(row["model"]),
                "model_type": mt,
                "use_case": str(row.get("use_case", ""))[:48]
                + ("…" if len(str(row.get("use_case", ""))) > 48 else ""),
                "archetype": str(row.get("telemetry_archetype", "")),
                "risk_score": d.risk_score,
                "risk_level": d.risk_level,
                "risk_traffic_light": d.risk_traffic_light,
                "breach_component": d.breach_component_score,
                "stress_component": d.observability_stress_score,
                "hitl": bool(d.human_review_required),
                "action_id": d.recommended_action_id,
                "needs_action": d.recommended_action_id != "MONITOR",
                "doing_better": _doing_better_than_expected(row),
                "issues": ", ".join(d.issue_categories) if d.issue_categories else "—",
                "error_pct": row.get("error_pct"),
                "drift_index": row.get("drift_index"),
                "audit_coverage_pct": row.get("audit_coverage_pct"),
                "escalation_pct": row.get("escalation_pct"),
                "updated_date": row.get("updated_date"),
            }
        )
    out = pd.DataFrame(rows)
    out["_sort"] = out["risk_level"].map(lambda x: LEVEL_SORT.get(x, 9))
    return out.sort_values(["_sort", "risk_score"], ascending=[True, False]).drop(
        columns=["_sort"]
    )


def _apply_home_filters(
    summary: pd.DataFrame,
    *,
    risk_levels: list[str],
    model_types: list[str],
    archetypes: list[str],
    action_posture: str,
    outperform: str,
    hitl_choice: str,
) -> pd.DataFrame:
    f = summary
    if risk_levels:
        f = f[f["risk_level"].isin(risk_levels)]
    if model_types:
        f = f[f["model_type"].isin(model_types)]
    if archetypes:
        f = f[f["archetype"].isin(archetypes)]
    if action_posture == "Needs action (non-MONITOR)":
        f = f[f["needs_action"]]
    elif action_posture == "Routine monitoring only":
        f = f[~f["needs_action"]]
    if outperform == "Doing better than expected":
        f = f[f["doing_better"]]
    elif outperform == "Not outperforming":
        f = f[~f["doing_better"]]
    if hitl_choice == "HITL required":
        f = f[f["hitl"]]
    elif hitl_choice == "No HITL":
        f = f[~f["hitl"]]
    return f


def _home_filter_summary_line(
    *,
    risk_levels: list[str],
    model_types: list[str],
    archetypes: list[str],
    action_posture: str,
    outperform: str,
    hitl_choice: str,
    n_after: int,
    n_total: int,
) -> str:
    posture_short = {
        "Needs action (non-MONITOR)": "needs action",
        "Routine monitoring only": "routine only",
    }.get(action_posture, "")
    out_short = {
        "Doing better than expected": "beating expectations",
        "Not outperforming": "not beating expectations",
    }.get(outperform, "")
    parts: list[str] = []
    if len(risk_levels) < len(RISK_LEVEL_OPTIONS):
        parts.append(f"risk: {', '.join(risk_levels) or '—'}")
    if model_types:
        parts.append(f"{len(model_types)} model type(s)")
    if archetypes:
        parts.append(f"{len(archetypes)} archetype(s)")
    if posture_short:
        parts.append(posture_short)
    if out_short:
        parts.append(out_short)
    if hitl_choice != "Any":
        parts.append(hitl_choice.lower())
    active = " · ".join(parts) if parts else "full fleet (risk levels only)"
    return f"**{n_after}** / {n_total} models — _{active}_"


def _render_model_detail(row: pd.Series, dec) -> None:
    """Deep-dive panels (shared by Analyze tab)."""
    arch = str(row.get("telemetry_archetype", "tabular_ml"))
    primary = primary_metrics_for_row(row)
    mid = str(row.get("model", ""))
    mtype = str(row.get("model_type", "—"))
    uc = str(row.get("use_case", "—"))
    arch_label = html_module.escape(humanize_column_name("telemetry_archetype") + f": {arch}")

    st.markdown(
        f"<div class='analyze-hero'><h2 style='margin:0 0 6px 0;font-weight:700'>"
        f"{html_module.escape(mid)}</h2>"
        f"<p style='margin:0;opacity:0.88;font-size:1.05rem'><b>Model type:</b> "
        f"{html_module.escape(mtype)}</p>"
        f"<p style='margin:4px 0 0 0;opacity:0.8;font-size:0.95rem'>{arch_label}</p>"
        f"<p style='margin:6px 0 0 0;opacity:0.85;font-size:0.92rem'><b>Use case:</b> "
        f"{html_module.escape(uc)}</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### Archetype signals")
    st.caption(
        "KPIs emphasize what matters for this model family (e.g. tabular: calibration; "
        "LLM/RAG: grounding and retrieval). Metrics with no value are omitted."
    )
    if primary:
        n = min(6, len(primary))
        pm_cols = st.columns(n)
        for i in range(n):
            col, label, val = primary[i]
            with pm_cols[i]:
                st.metric(label, format_metric_value(col, val))
    else:
        st.info("No primary metrics available for this row.")

    with st.expander("Full model record (navigate by section)", expanded=False):
        st.caption("Each section lists only fields that have values in the enriched workbook.")
        for title, cols in PROFILE_SECTIONS:
            sub = _section_metrics_dataframe(row, cols)
            if sub is None or sub.empty:
                continue
            st.markdown(f"**{title}**")
            st.dataframe(
                sub,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Field": st.column_config.TextColumn(width="medium"),
                    "Value": st.column_config.TextColumn(width="large"),
                },
            )
        rest = _remaining_telemetry_dataframe(row)
        if rest is not None and not rest.empty:
            st.markdown("**Other workbook fields**")
            st.dataframe(
                rest,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Field": st.column_config.TextColumn(width="medium"),
                    "Value": st.column_config.TextColumn(width="large"),
                },
            )

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("Risk orchestration")
        color = LEVEL_COLOR.get(dec.risk_level, "#333")
        st.markdown(
            f"<div style='font-size:2.2rem;font-weight:700;color:{color}'>{dec.risk_score:.1f}</div>"
            "<div style='opacity:0.75;font-size:0.9rem'>Composite score (0–100)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_orchestrator_risk_row_html(dec), unsafe_allow_html=True)
        st.markdown(
            f"**Breach component** `{dec.breach_component_score:.1f}` · "
            f"**Observability stress** `{dec.observability_stress_score:.1f}`"
        )
        st.markdown(f"**Issue categories:** {', '.join(dec.issue_categories) or '—'}")
        st.markdown(
            f"**Recommended action** `{dec.recommended_action_id}` — {dec.recommended_action_text}"
        )
        hitl = "Yes" if dec.human_review_required else "No"
        st.markdown(f"**Human review:** {hitl} (priority: {dec.human_review_priority})")
        st.caption(dec.rationale_summary)

    with c2:
        st.subheader("Narrative / modality")
        nar_keys = narrative_metric_keys_for_arch(arch)
        nar_rows = [
            {
                "Metric": humanize_column_name(k),
                "Value": format_snapshot_value(k, row.get(k)),
            }
            for k in nar_keys
            if k in row.index and pd.notna(row.get(k))
        ]
        if not nar_rows:
            st.info("No narrative or modality metrics for this archetype.")
        else:
            st.dataframe(
                pd.DataFrame(nar_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn(width="medium"),
                    "Value": st.column_config.TextColumn(width="small"),
                },
            )

    _render_live_monitoring_section(row)

    st.markdown("---")
    dim_active = [d for d in dec.dimension_results if not d.get("skipped")]
    dim_rows = [
        {
            "dimension": d["dimension"].replace("_", " ").title(),
            "severity": d["severity"],
            "breaches": len(d["breaches"]),
        }
        for d in dim_active
    ]
    if dim_rows:
        dim_df = pd.DataFrame(dim_rows)
        st.markdown("##### Governance dimensions")
        st.caption("Severity by dimension (skipped dimensions are hidden). Bar color reflects severity.")
        _dimension_severity_chart(dim_df)
    else:
        st.info("No active governance dimensions for this profile.")

    tabs = st.tabs(
        ["Reliability", "Narrative eval.", "Compliance", "Auditability", "Raw JSON"]
    )
    dim_map = {d["dimension"]: d for d in dec.dimension_results}

    with tabs[0]:
        with st.expander("Evaluator payload (reliability)", expanded=False):
            st.json(dim_map.get("reliability", {}))
        rel_specs = reliability_metric_keys_for_arch(arch)
        rel_data = [
            {"Metric": label, "Value": format_snapshot_value(col, row.get(col))}
            for col, label in rel_specs
            if col in row.index and pd.notna(row.get(col))
        ]
        st.dataframe(
            pd.DataFrame(rel_data) if rel_data else pd.DataFrame(columns=["Metric", "Value"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(width="large"),
                "Value": st.column_config.TextColumn(width="small"),
            },
        )

    with tabs[1]:
        with st.expander("Evaluator payload (narrative assurance)", expanded=False):
            st.json(dim_map.get("narrative_assurance", {}))

    with tabs[2]:
        with st.expander("Evaluator payload (compliance & security)", expanded=False):
            st.json(dim_map.get("compliance_security", {}))
        comp_rows = [
            {
                "Metric": humanize_column_name("policy_violation_rate"),
                "Value": format_snapshot_value("policy_violation_rate", row.get("policy_violation_rate")),
            },
            {
                "Metric": humanize_column_name("security_anomaly_count"),
                "Value": format_snapshot_value("security_anomaly_count", row.get("security_anomaly_count")),
            },
            {
                "Metric": humanize_column_name("compliance_incidents"),
                "Value": format_snapshot_value("compliance_incidents", row.get("compliance_incidents")),
            },
        ]
        st.dataframe(
            pd.DataFrame(comp_rows) if comp_rows else pd.DataFrame(columns=["Metric", "Value"]),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[3]:
        with st.expander("Evaluator payload (auditability)", expanded=False):
            st.json(dim_map.get("auditability", {}))
        aud_rows = [
            {
                "Metric": humanize_column_name("audit_coverage_pct"),
                "Value": format_snapshot_value("audit_coverage_pct", row.get("audit_coverage_pct")),
            },
            {
                "Metric": humanize_column_name("decision_trace_completeness"),
                "Value": format_snapshot_value(
                    "decision_trace_completeness", row.get("decision_trace_completeness")
                ),
            },
            {
                "Metric": humanize_column_name("minimum_required_audit_coverage"),
                "Value": format_snapshot_value(
                    "minimum_required_audit_coverage",
                    row.get("minimum_required_audit_coverage"),
                ),
            },
        ]
        st.dataframe(
            pd.DataFrame(aud_rows) if aud_rows else pd.DataFrame(columns=["Metric", "Value"]),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[4]:
        st.json(decision_to_jsonable(dec))


def main() -> None:
    st.set_page_config(
        page_title="Observability Governance POC",
        layout="wide",
    )

    st.markdown(
        """
<style>
    .block-container { padding-top: 1rem; max-width: 1400px; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    .analyze-hero {
        padding: 14px 18px;
        margin-bottom: 12px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        background: linear-gradient(135deg, rgba(25,118,210,0.06) 0%, rgba(0,0,0,0) 55%);
    }
    /* Fixed micro-rail flush to viewport left (no st.sidebar) */
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:first-child {
        flex: 0 0 0 !important;
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
        border: none !important;
    }
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:first-child
        > div[data-testid="stVerticalBlock"] {
        position: fixed !important;
        left: 0 !important;
        top: 3.5rem !important;
        bottom: 0 !important;
        width: 2.65rem !important;
        min-width: 2.65rem !important;
        max-width: 2.65rem !important;
        z-index: 1000002 !important;
        padding: 0.3rem 0.12rem 0.45rem 0.12rem !important;
        box-sizing: border-box !important;
        background: rgba(252, 252, 253, 0.97) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.04) !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
    }
    @media (prefers-color-scheme: dark) {
        section[data-testid="stAppViewContainer"] section.main
            div[data-testid="stHorizontalBlock"]:first-of-type
            > div[data-testid="column"]:first-child
            > div[data-testid="stVerticalBlock"] {
            background: rgba(20, 22, 28, 0.98) !important;
            border-right-color: rgba(255, 255, 255, 0.08) !important;
            box-shadow: 2px 0 16px rgba(0, 0, 0, 0.35) !important;
        }
    }
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:nth-child(2) {
        margin-left: 2.65rem !important;
        flex: 1 1 auto !important;
        max-width: calc(100% - 2.65rem) !important;
    }
    /* Nav item: active left bar */
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:first-child
        .element-container:has(button[kind="primary"]) {
        display: flex !important;
        justify-content: center !important;
        border-left: 2px solid #1769bd !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
        border-radius: 0 6px 6px 0 !important;
        background: linear-gradient(90deg, rgba(23, 105, 189, 0.11) 0%, transparent 85%) !important;
        margin-bottom: 0.02rem !important;
    }
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:first-child
        .element-container:has(button[kind="secondary"]) {
        display: flex !important;
        justify-content: center !important;
        border-left: 2px solid transparent !important;
        margin-bottom: 0.02rem !important;
    }
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:first-child
        button[kind="primary"],
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:first-child
        button[kind="secondary"] {
        width: 1.85rem !important;
        min-width: 1.85rem !important;
        max-width: 1.85rem !important;
        padding: 0.08rem 0.04rem !important;
        min-height: 1.58rem !important;
        font-size: 0.88rem !important;
        line-height: 1 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:first-child
        .nav-rail-cute-label {
        font-size: 0.52rem !important;
        font-weight: 600 !important;
        text-align: center !important;
        margin: 0.02rem 0 0.32rem 0 !important;
        line-height: 1.02 !important;
        opacity: 0.85 !important;
        padding: 0 !important;
        word-break: break-word !important;
        letter-spacing: -0.03em !important;
    }
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:first-child hr {
        margin: 0.28rem 0.1rem !important;
        border-color: rgba(128, 128, 128, 0.18) !important;
    }
    section[data-testid="stAppViewContainer"] section.main
        div[data-testid="stHorizontalBlock"]:first-of-type
        > div[data-testid="column"]:first-child
        .element-container:has(.nav-rail-cute-label) {
        display: flex !important;
        justify-content: center !important;
    }
    .nav-rail-micro {
        font-size: 0.5rem;
        font-weight: 800;
        text-align: center;
        opacity: 0.45;
        margin: 0 0 0.28rem 0;
        letter-spacing: 0.06em;
    }
</style>
""",
        unsafe_allow_html=True,
    )

    if "nav_tab" not in st.session_state:
        st.session_state.nav_tab = "Home"

    nav_current = st.session_state.nav_tab
    if nav_current not in NAV_PAGES:
        st.session_state.nav_tab = "Home"
        nav_current = "Home"

    _rail, _main_panel = st.columns([1, 6], gap="small")

    with _rail:
        st.markdown('<p class="nav-rail-micro">GOV</p>', unsafe_allow_html=True)
        for i, (icon, label, page, key, hlp) in enumerate(NAV_RAIL_ENTRIES):
            if i == 3:
                st.divider()
            _nav_rail_item(
                icon,
                label,
                page,
                key=key,
                active=nav_current == page,
                help_tt=hlp,
            )

    with _main_panel:
        nav_tab = st.session_state.nav_tab

        st.title("Observability-driven governance")
        if nav_tab == "Data source":
            st.caption(
                "POC uses a **fixed** enriched workbook — every viewer reads the same file on the server."
            )
            st.markdown(f"**Enriched workbook:** `{POC_ENRICHED_PATH}`")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Reload from disk", help="Clear cached workbook and re-read the file"):
                    _load_data.clear()
                    st.rerun()
            with rc2:
                st.caption("**Phase 1 POC:** rule bands + observability stress + HITL gates.")
        else:
            st.caption(
                "Baseline vs observed telemetry → risk score → action. "
                "Home highlights elevation; Models shows the full fleet at a glance."
            )

        path_in = POC_ENRICHED_PATH
        try:
            df = _load_data(path_in)
        except Exception as e:
            st.error(f"Could not load data: {e}")
            if nav_tab != "Data source":
                st.info(f"Ensure the workbook exists at `{POC_ENRICHED_PATH}` (see **Data** tab).")
            st.stop()

        _lc = connect()
        init_schema(_lc)
        ensure_seeded(_lc, df, path_in)
        register_workbook_path(_lc, path_in)
        live_running = is_running(_lc)
        df_eff = build_effective_dataframe(df, _lc, live_running=live_running)
        _lc.close()

        cfg = _cfg()
        summary = _build_fleet_summary(df_eff, cfg)
        models_ordered = _models_in_workbook_order(df_eff)
        if not models_ordered:
            st.error("No models in workbook.")
            st.stop()

        if "home_level_filter" not in st.session_state:
            st.session_state.home_level_filter = list(RISK_LEVEL_OPTIONS)
        if "home_model_types" not in st.session_state:
            st.session_state.home_model_types = []
        if "home_archetypes" not in st.session_state:
            st.session_state.home_archetypes = []
        if "home_action_posture" not in st.session_state:
            st.session_state.home_action_posture = HOME_ACTION_OPTIONS[0]
        if "home_outperform" not in st.session_state:
            st.session_state.home_outperform = HOME_OUTPERFORM_OPTIONS[0]
        if "home_hitl_filter" not in st.session_state:
            st.session_state.home_hitl_filter = HOME_HITL_OPTIONS[0]
        if "models_level_filter" not in st.session_state:
            st.session_state.models_level_filter = list(RISK_LEVEL_OPTIONS)
        _mo_set = set(models_ordered)
        if "analyze_pick" not in st.session_state:
            st.session_state.analyze_pick = models_ordered[0]
        elif str(st.session_state.analyze_pick).strip() not in _mo_set:
            # Pandas/numpy model ids can fail plain `in list` checks; normalize to str.
            st.session_state.analyze_pick = models_ordered[0]

        # ----- HOME -----
        if nav_tab == "Home":
            if live_running:
                st.info(
                    "Live telemetry simulation is **running** — fleet scores use simulated KPIs merged from the shared store."
                )
            model_type_opts = sorted(
                {x.strip() for x in summary["model_type"].astype(str) if x.strip()},
                key=str.lower,
            )
            archetype_opts = sorted(
                {x for x in summary["archetype"].astype(str).unique().tolist() if x and str(x).strip()},
                key=str.lower,
            )

            st.subheader("Fleet overview")
            st.caption(
                "Whole-workbook signals on a **0–100** scale (not affected by table filters below)."
            )
            ov1, ov2, ov3, ov4, ov5 = st.columns(5)
            with ov1:
                with st.container(border=True):
                    st.caption("Models in workbook")
                    st.markdown(f"**{len(summary)}**")
            with ov2:
                _fleet_overview_score_box(
                    "Avg composite risk",
                    float(summary["risk_score"].mean()),
                )
            with ov3:
                _fleet_overview_score_box(
                    "Avg breach load",
                    float(summary["breach_component"].mean()),
                )
            with ov4:
                _fleet_overview_score_box(
                    "Avg observability stress",
                    float(summary["stress_component"].mean()),
                )
            with ov5:
                _fleet_overview_score_box(
                    "Peak composite risk",
                    float(summary["risk_score"].max()),
                )

            st.divider()
            # Slot reserved here so Priority renders above Filters + All models while filters run first in script.
            _home_priority_slot = st.empty()

            am_col, filt_col = st.columns([5, 1])
            with am_col:
                st.markdown("#### All models")
                st.caption("Filtered set; use **Filters** on the right, then review the grid below.")
            with filt_col:
                with st.popover("Filters", use_container_width=True):
                    _home_filter_popover_contents(model_type_opts, archetype_opts)

            risk_sel = list(st.session_state.home_level_filter)
            mt_sel = list(st.session_state.home_model_types)
            arch_sel = list(st.session_state.home_archetypes)
            posture = str(st.session_state.home_action_posture)
            outperf = str(st.session_state.home_outperform)
            hitl_sel = str(st.session_state.home_hitl_filter)

            if not risk_sel:
                st.warning("Select at least one **risk level** in Filters (or reset).")
                filtered = summary.iloc[0:0]
            else:
                filtered = _apply_home_filters(
                    summary,
                    risk_levels=risk_sel,
                    model_types=mt_sel,
                    archetypes=arch_sel,
                    action_posture=posture,
                    outperform=outperf,
                    hitl_choice=hitl_sel,
                )

            with _home_priority_slot.container():
                st.markdown("#### Priority watchlist")
                st.caption(
                    "Elevated models (**critical / high / medium**) from the same filtered set, "
                    "sorted by severity then score."
                )
                priority = filtered[filtered["risk_level"].isin(["critical", "high", "medium"])]
                if priority.empty:
                    st.success("No elevated models in the current filter — fleet within POC bands.")
                else:
                    watch_cols = [
                        "model",
                        "model_type",
                        "risk_score",
                        "risk_level",
                        "risk_traffic_light",
                        "breach_component",
                        "stress_component",
                        "hitl",
                        "action_id",
                        "issues",
                        "use_case",
                    ]
                    wdf = _fleet_table_for_display(priority, watch_cols)
                    st.dataframe(
                        _style_fleet_table(wdf),
                        use_container_width=True,
                        column_config={
                            "risk_score": st.column_config.NumberColumn(format="%.1f"),
                            "breach_component": st.column_config.NumberColumn(format="%.1f"),
                            "stress_component": st.column_config.NumberColumn(format="%.1f"),
                        },
                    )

            st.markdown(
                _home_filter_summary_line(
                    risk_levels=risk_sel,
                    model_types=mt_sel,
                    archetypes=arch_sel,
                    action_posture=posture,
                    outperform=outperf,
                    hitl_choice=hitl_sel,
                    n_after=len(filtered),
                    n_total=len(summary),
                )
            )

            show_cols = [
                "model",
                "model_type",
                "risk_score",
                "risk_level",
                "risk_traffic_light",
                "archetype",
                "error_pct",
                "audit_coverage_pct",
                "escalation_pct",
                "hitl",
                "action_id",
            ]
            adf = _fleet_table_for_display(filtered, show_cols)
            st.dataframe(
                _style_fleet_table(adf),
                use_container_width=True,
                column_config={
                    "risk_score": st.column_config.NumberColumn(format="%.1f"),
                    "error_pct": st.column_config.NumberColumn(format="%.1f"),
                    "audit_coverage_pct": st.column_config.NumberColumn(format="%.0f"),
                    "escalation_pct": st.column_config.NumberColumn(format="%.0f"),
                },
            )

        # ----- MODELS (card grid) -----
        elif nav_tab == "Models":
            st.subheader("Model cards")
            st.caption("Each box shows archetype-relevant KPIs (not a fixed drift/error/audit triple for every model).")

            mp = st.columns(len(FILTER_PRESETS))
            for i, (plabel, levels) in enumerate(FILTER_PRESETS.items()):
                slug = plabel.lower().replace(" ", "_").replace("+", "plus")
                with mp[i]:
                    if st.button(plabel, key=f"models_preset_{slug}", use_container_width=True):
                        st.session_state.models_level_filter = list(levels)
                        st.rerun()
            inc_m = st.multiselect(
                "Custom risk levels",
                options=RISK_LEVEL_OPTIONS,
                key="models_level_filter",
            )
            cards = summary[summary["risk_level"].isin(inc_m)] if inc_m else summary.iloc[0:0]
            if len(cards) > 0:
                cards = cards.copy()
                cards["_s"] = cards["risk_level"].map(lambda x: LEVEL_SORT.get(x, 9))
                cards = cards.sort_values(["_s", "risk_score"], ascending=[True, False]).drop(
                    columns=["_s"]
                )

            n_cols = 4
            idx = 0
            mids = cards["model"].tolist()
            while idx < len(mids):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    if idx >= len(mids):
                        break
                    mid = mids[idx]
                    idx += 1
                    srow = cards[cards["model"] == mid].iloc[0]
                    frow = df_eff.loc[df_eff["model"].astype(str) == mid].iloc[0]
                    pm = primary_metrics_for_row(frow)[:3]
                    tl = str(srow["risk_traffic_light"]).lower()
                    rl = str(srow["risk_level"]).lower()
                    tbg, tfg = TRAFFIC_PILL_STYLE.get(tl, ("#eceff1", "#333"))
                    border_color = LEVEL_COLOR.get(srow["risk_level"], "#9e9e9e")
                    sig = TRAFFIC_LABEL.get(tl, tl.title())
                    rtitle = LEVEL_TITLE.get(rl, rl.title())
                    mt_short = str(srow["model_type"])
                    if len(mt_short) > 36:
                        mt_short = mt_short[:36] + "…"
                    with cols[j]:
                        with st.container(border=True):
                            st.markdown(
                                f"<div style='border-left:4px solid {border_color};padding-left:10px;margin:-8px 0 6px 0'>"
                                f"<span style='font-size:1.12rem;font-weight:700'>{html_module.escape(mid)}</span><br/>"
                                f"<span style='opacity:0.9;font-size:0.88rem'>{html_module.escape(mt_short)}</span><br/>"
                                f"<span style='display:inline-block;margin-top:6px'>{_pill_span(tbg, tfg, sig)}"
                                f"<span style='font-weight:700;color:{border_color}'>{rtitle}</span>"
                                f" · score {srow['risk_score']:.1f}</span></div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(srow["archetype"][:28] + ("…" if len(str(srow["archetype"])) > 28 else ""))
                            if pm:
                                bits = [
                                    f"**{lbl[:14]}** `{format_metric_value(c, v)}`"
                                    for c, lbl, v in pm
                                ]
                                st.markdown("<br/>".join(bits), unsafe_allow_html=True)
                            else:
                                st.caption("—")
                            st.caption(f"B {srow['breach_component']:.0f} / S {srow['stress_component']:.0f}")
                            st.button(
                                "Open in Analyze",
                                key=f"open_{str(mid)}",
                                use_container_width=True,
                                on_click=_open_model_in_analyze,
                                args=(str(mid),),
                            )

        # ----- ANALYZE -----
        elif nav_tab == "Analyze":
            st.subheader("Single-model analysis")
            st.caption("Models are listed as **Model 1 … Model N** in workbook order (e.g. M01–M50).")
            pick = st.selectbox(
                "Model",
                models_ordered,
                format_func=_analyze_model_label,
                key="analyze_pick",
            )
            row = df_eff.loc[df_eff["model"].astype(str) == pick].iloc[0]
            dec = run_governance_for_row(row, cfg)
            _render_model_detail(row, dec)

        # ----- DATA SOURCE -----
        elif nav_tab == "Data source":
            st.subheader("Load status")
            st.success(f"Workbook OK — **{len(df)}** models in view (`{path_in}`).")
            st.caption("Replace the file on disk at the path above, then use **Reload from disk** to pick up changes.")

            st.markdown("#### Live telemetry (simulated)")
            st.caption(
                "Shared SQLite store (`STREAMLIT_LIVE_DB` env overrides path). "
                "When **running**, KPI snapshots merge into the workbook frame and feed governance. "
                "**Stop** returns scores to the static workbook baseline."
            )
            _cstat = connect()
            init_schema(_cstat)
            _lr = is_running(_cstat)
            _cstat.close()
            b1, b2, b3 = st.columns(3)
            with b1:
                st.button("Start generator", on_click=_live_sim_start, disabled=_lr, key="live_btn_start")
            with b2:
                st.button("Stop generator", on_click=_live_sim_stop, disabled=not _lr, key="live_btn_stop")
            with b3:
                st.button("Reset & re-seed", on_click=_live_sim_reset, key="live_btn_reset")

            @st.fragment(run_every=2)
            def _live_events_sheet() -> None:
                c = connect()
                init_schema(c)
                running = is_running(c)
                ev = fetch_recent_events(c, 500)
                c.close()
                st.caption(f"Generator: **{'running' if running else 'stopped'}** — recent events (newest first).")
                if ev.empty:
                    st.info("No events yet — load a workbook and use **Reset & re-seed** to populate history.")
                else:
                    st.dataframe(
                        ev,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "timestamp": st.column_config.DatetimeColumn("Timestamp (UTC)"),
                            "value": st.column_config.NumberColumn(format="%.6f"),
                        },
                    )

            _live_events_sheet()

        else:
            st.error(f"Unknown section: {nav_tab!r}")

    st.divider()
    st.caption(
        "POC only — not production authorization. False positives, missed detections, and over-automation remain primary governance risks."
    )


main()
