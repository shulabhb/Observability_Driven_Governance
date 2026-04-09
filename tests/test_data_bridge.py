"""Step 1: data bridge — load workbook, required columns, dates, validation."""

from pathlib import Path

import pandas as pd
import pytest

from governance_engine.data_bridge import (
    REQUIRED_POST_COLUMNS,
    REQUIRED_PRE_COLUMNS,
    load_enriched_post,
    load_enriched_pre,
    validate_post_dataframe,
    validate_pre_dataframe,
)


def test_load_post_has_required_columns(enriched_path: Path) -> None:
    df = load_enriched_post(enriched_path)
    assert len(df) >= 25
    assert len(df) == 50, "Rebuild with: python enrich_monitoring.py (synthetic 50-model fleet)"
    for c in REQUIRED_POST_COLUMNS:
        assert c in df.columns, f"missing {c}"


def test_load_pre_has_required_columns(enriched_path: Path) -> None:
    df = load_enriched_pre(enriched_path)
    assert len(df) == 50, "Rebuild with: python enrich_monitoring.py (synthetic fleet)"
    for c in REQUIRED_PRE_COLUMNS:
        assert c in df.columns, f"missing {c}"


def test_post_dates_populated(enriched_path: Path) -> None:
    df = load_enriched_post(enriched_path)
    assert df["updated_date"].notna().all()


def test_pre_dates_populated(enriched_path: Path) -> None:
    df = load_enriched_pre(enriched_path)
    assert df["telemetry_snapshot_date"].notna().all()


def test_validate_post_clean(enriched_path: Path) -> None:
    df = load_enriched_post(enriched_path)
    issues = validate_post_dataframe(df)
    assert issues == [], issues


def test_validate_pre_clean(enriched_path: Path) -> None:
    df = load_enriched_pre(enriched_path)
    issues = validate_pre_dataframe(df)
    assert issues == [], issues


def test_missing_workbook_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_enriched_post(tmp_path / "nope.xlsx")
