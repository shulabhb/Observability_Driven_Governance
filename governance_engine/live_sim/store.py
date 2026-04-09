"""SQLite persistence for shared live telemetry simulation."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

from governance_engine.data_bridge import WORKSPACE_ROOT

store_lock = threading.RLock()

DEFAULT_DB_PATH = WORKSPACE_ROOT / "live_telemetry.db"


def live_db_path() -> Path:
    p = os.environ.get("STREAMLIT_LIVE_DB", "").strip()
    return Path(p) if p else DEFAULT_DB_PATH


def connect() -> sqlite3.Connection:
    path = live_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            ts REAL NOT NULL,
            model_id TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (ts, model_id, metric)
        );
        CREATE INDEX IF NOT EXISTS idx_events_model_ts ON events (model_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts DESC);
        CREATE TABLE IF NOT EXISTS snapshot_latest (
            model_id TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (model_id, metric)
        );
        """
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    if row is None:
        return default
    return str(row[0])


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        (key, value),
    )
    conn.commit()


def clear_all_data(conn: sqlite3.Connection) -> None:
    conn.executescript("DELETE FROM events; DELETE FROM snapshot_latest;")
    conn.commit()
