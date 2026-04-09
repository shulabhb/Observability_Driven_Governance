"""Shared SQLite-backed live telemetry simulation for the governance POC dashboard."""

from governance_engine.live_sim.merge import build_effective_dataframe
from governance_engine.live_sim.simulator import (
    catch_up_ticks,
    ensure_background_writer,
    ensure_seeded,
    fetch_recent_events,
    fetch_series,
    is_running,
    register_workbook_path,
    reset_simulation,
    set_running,
    tick_once,
)
from governance_engine.live_sim.store import connect, init_schema
from governance_engine.live_sim import metrics as live_metrics

__all__ = [
    "build_effective_dataframe",
    "catch_up_ticks",
    "connect",
    "ensure_background_writer",
    "ensure_seeded",
    "fetch_recent_events",
    "fetch_series",
    "init_schema",
    "is_running",
    "live_metrics",
    "register_workbook_path",
    "reset_simulation",
    "set_running",
    "tick_once",
]
