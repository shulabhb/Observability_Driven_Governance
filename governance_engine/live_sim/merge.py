"""Merge latest live snapshots into workbook dataframe when simulation is running."""

from __future__ import annotations

import sqlite3

import pandas as pd


def build_effective_dataframe(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    *,
    live_running: bool,
) -> pd.DataFrame:
    """
    When live_running is True, overwrite numeric columns from snapshot_latest per model.
    When False, return the workbook frame unchanged (baseline).
    """
    if not live_running or df.empty:
        return df

    rows = conn.execute(
        "SELECT model_id, metric, value FROM snapshot_latest",
    ).fetchall()
    if not rows:
        return df.copy()

    snap = pd.DataFrame(rows, columns=["model_id", "metric", "value"])
    piv = snap.pivot(index="model_id", columns="metric", values="value")
    out = df.copy()
    mid_series = out["model"].astype(str)

    for mid in mid_series.unique():
        if mid not in piv.index:
            continue
        idx = mid_series == mid
        for col in piv.columns:
            if col in out.columns:
                try:
                    out.loc[idx, col] = float(piv.loc[mid, col])
                except (TypeError, ValueError, KeyError):
                    pass
    return out
