"""Tests for live telemetry SQLite simulation and merge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from governance_engine.engine import run_governance_for_row
from governance_engine.expectations import load_expectations
from governance_engine.live_sim.merge import build_effective_dataframe
from governance_engine.live_sim.simulator import seed_history, workbook_fingerprint
from governance_engine.live_sim.store import clear_all_data, connect, init_schema


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "live.db"
    monkeypatch.setenv("STREAMLIT_LIVE_DB", str(db))
    return db


def _minimal_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "M01",
                "model_type": "Classifier",
                "use_case": "Test",
                "telemetry_archetype": "tabular_ml",
                "updated_date": "2024-06-01",
                "accuracy_pct": 92.0,
                "error_pct": 3.0,
                "drift_index": 0.08,
                "escalation_pct": 5.0,
                "audit_coverage_pct": 90.0,
            }
        ]
    )


def test_merge_pass_through_when_not_running(isolated_db: Path) -> None:
    df = _minimal_row()
    c = connect()
    init_schema(c)
    out = build_effective_dataframe(df, c, live_running=False)
    c.close()
    pd.testing.assert_frame_equal(df.reset_index(drop=True), out.reset_index(drop=True))


def test_merge_overwrites_when_running(isolated_db: Path) -> None:
    df = _minimal_row()
    c = connect()
    init_schema(c)
    c.execute(
        "INSERT INTO snapshot_latest (model_id, metric, value, updated_at) VALUES (?,?,?,?)",
        ("M01", "error_pct", 17.5, 1.0),
    )
    c.commit()
    out = build_effective_dataframe(df, c, live_running=True)
    c.close()
    assert float(out.loc[out["model"] == "M01", "error_pct"].iloc[0]) == 17.5


def test_seed_idempotent_fingerprint(isolated_db: Path) -> None:
    df = _minimal_row()
    fp1 = workbook_fingerprint("/a/b.xlsx", df)
    fp2 = workbook_fingerprint("/a/b.xlsx", df)
    assert fp1 == fp2
    assert fp1 != workbook_fingerprint("/other.xlsx", df)


def test_seed_and_governance_runs(isolated_db: Path) -> None:
    df = _minimal_row()
    c = connect()
    init_schema(c)
    clear_all_data(c)
    seed_history(c, df, str(Path("/virtual/workbook.xlsx")), rng_base=7)
    n = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert n > 0
    merged = build_effective_dataframe(df, c, live_running=True)
    c.close()
    cfg = load_expectations()
    dec = run_governance_for_row(merged.iloc[0], cfg)
    assert 0 <= dec.risk_score <= 100
