"""
Load enriched monitoring workbooks and validate rows for the governance engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ENRICHED_PATH = WORKSPACE_ROOT / "enriched_monitoring_report.xlsx"
SHEET_POST = "Post-Go-Live Monitoring Report"
SHEET_PRE = "Pre-Go-Live Monitoring Report"

# Identity + core observability (post-go-live production telemetry)
REQUIRED_POST_COLUMNS: tuple[str, ...] = (
    "model",
    "model_type",
    "use_case",
    "telemetry_archetype",
    "updated_date",
    "accuracy_pct",
    "error_pct",
    "drift_index",
    "escalation_pct",
    "audit_coverage_pct",
)

REQUIRED_PRE_COLUMNS: tuple[str, ...] = (
    "Model",
    "telemetry_snapshot_date",
    "Model Type",
    "Use Case",
    "telemetry_archetype",
    "Accuracy %",
    "Error %",
    "Drift Index",
    "Escalation %",
    "Audit Coverage %",
)

NARRATIVE_ARCHETYPES = frozenset({"llm_rag", "llm_text", "agentic_system"})
VISION_DOC_ARCHETYPES = frozenset({"vision_document"})
VISION_CNN_ARCHETYPES = frozenset({"vision_cnn"})
SPEECH_ARCHETYPES = frozenset({"speech_asr"})
NLP_CLS_ARCHETYPES = frozenset({"nlp_classifier"})


def load_enriched_post(
    path: str | Path | None = None,
    *,
    sheet_name: str = SHEET_POST,
) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_ENRICHED_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Enriched workbook not found: {p}")
    df = pd.read_excel(p, sheet_name=sheet_name)
    missing = [c for c in REQUIRED_POST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Post sheet missing required columns: {missing}")
    return df


def load_enriched_pre(
    path: str | Path | None = None,
    *,
    sheet_name: str = SHEET_PRE,
) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_ENRICHED_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Enriched workbook not found: {p}")
    df = pd.read_excel(p, sheet_name=sheet_name)
    missing = [c for c in REQUIRED_PRE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Pre sheet missing required columns: {missing}")
    return df


def _is_blank_series(s: pd.Series) -> pd.Series:
    if s.dtype == "datetime64[ns]" or pd.api.types.is_datetime64_any_dtype(s):
        return s.isna()
    return s.isna() | (s.astype(str).str.strip() == "") | (s.astype(str) == "nan")


def validate_post_dataframe(df: pd.DataFrame) -> list[str]:
    """Return human-readable issues; empty list means OK for POC."""
    issues: list[str] = []
    if df.empty:
        return ["DataFrame is empty"]
    blank_model = _is_blank_series(df["model"])
    if blank_model.any():
        issues.append(f"model missing on rows: {df.index[blank_model].tolist()}")
    blank_arch = _is_blank_series(df["telemetry_archetype"])
    if blank_arch.any():
        issues.append(f"telemetry_archetype missing on rows: {df.index[blank_arch].tolist()}")
    if "updated_date" in df.columns and df["updated_date"].isna().all():
        issues.append("updated_date is all null")
    return issues


def validate_pre_dataframe(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if df.empty:
        return ["DataFrame is empty"]
    if _is_blank_series(df["Model"]).any():
        issues.append("Model missing on some rows")
    if "telemetry_snapshot_date" in df.columns and df["telemetry_snapshot_date"].isna().all():
        issues.append("telemetry_snapshot_date is all null")
    return issues


def post_row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Serialize a row for JSON / engine (NaN -> None)."""
    d = row.to_dict()
    out: dict[str, Any] = {}
    for k, v in d.items():
        if pd.isna(v):
            out[k] = None
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat() if hasattr(v, "hour") else str(v)
        else:
            out[k] = v
    return out
