"""Seed history, advance live simulation, optional background writer thread."""

from __future__ import annotations

import atexit
import hashlib
import threading
import time

import numpy as np
import pandas as pd

from governance_engine.live_sim import metrics as mlib
from governance_engine.data_bridge import load_enriched_post
from governance_engine.live_sim.store import clear_all_data, connect, get_meta, init_schema, set_meta, store_lock

HORIZON_SEC = 48 * 3600
TICK_INTERVAL_SEC = 2.0
MAX_TICKS_PER_CATCHUP = 5
SEED_MEAN_GAP_SEC = 900.0  # ~15 min between seed events
SEED_MIN_GAP = 120.0
SEED_MAX_GAP = 7200.0
MAX_SEED_STEPS = 280

_writer_thread: threading.Thread | None = None
_writer_stop = threading.Event()


def _stable_seed(path_str: str, base: int) -> int:
    h = hashlib.sha256(path_str.encode()).hexdigest()
    return (int(h[:8], 16) + base) % (2**31)


def is_running(conn) -> bool:
    return get_meta(conn, "running", "0") == "1"


def set_running(conn, running: bool) -> None:
    set_meta(conn, "running", "1" if running else "0")
    if running:
        ensure_background_writer()


def workbook_fingerprint(path_str: str, df: pd.DataFrame) -> str:
    models = ",".join(sorted(df["model"].astype(str).unique().tolist()))
    return hashlib.sha256(f"{path_str}|{models}".encode()).hexdigest()[:24]


def trim_old_events(conn, now_ts: float | None = None) -> None:
    now_ts = now_ts or time.time()
    cutoff = now_ts - HORIZON_SEC
    conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
    conn.commit()


def _next_ts(prev: float, rng: np.random.Generator) -> float:
    gap = float(rng.exponential(SEED_MEAN_GAP_SEC))
    gap = max(SEED_MIN_GAP, min(SEED_MAX_GAP, gap))
    return prev + gap


def _perturb(
    metric: str,
    value: float,
    baseline: float,
    rng: np.random.Generator,
    *,
    shock: bool,
) -> float:
    """
    Fluctuate around the workbook baseline — not a pure random walk on [lo, hi].

    A symmetric walk + clipping biases "good" values toward worse outcomes (more room
    to move toward the middle of the band). Mean reversion to `baseline` removes that
    drift; noise and optional shocks are zero-mean so some metrics move up and some down.
    """
    lo, hi = mlib.bounds_for_metric(metric)
    span = hi - lo or 1.0
    theta = 0.085
    pull = theta * (float(baseline) - float(value))
    noise = rng.normal(0.0, 0.017 * span)
    if shock:
        noise += rng.normal(0.0, 0.085 * span)
    v = float(value) + pull + noise
    return mlib.clip_metric(metric, v)


def _insert_event(conn, ts: float, model_id: str, metric: str, value: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO events (ts, model_id, metric, value) VALUES (?, ?, ?, ?)",
        (ts, model_id, metric, float(value)),
    )
    conn.execute(
        """INSERT INTO snapshot_latest (model_id, metric, value, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(model_id, metric) DO UPDATE SET
             value = excluded.value, updated_at = excluded.updated_at""",
        (model_id, metric, float(value), ts),
    )


def seed_history(conn, df: pd.DataFrame, path_str: str, rng_base: int = 42) -> None:
    """Populate ~48h of irregular events from workbook baselines."""
    init_schema(conn)
    fp = workbook_fingerprint(path_str, df)
    set_meta(conn, "workbook_fp", fp)
    set_meta(conn, "seed", str(rng_base))
    set_meta(conn, "workbook_path", path_str)
    now = time.time()
    start = now - HORIZON_SEC

    for _, row in df.iterrows():
        mid = str(row["model"])
        seed = _stable_seed(path_str, rng_base) + sum(ord(c) for c in mid)
        rng = np.random.default_rng(seed % (2**32))
        mets = mlib.metrics_for_row(row)
        if not mets:
            continue
        state = {met: mlib.initial_value_for_row(row, met) for met in mets}
        baselines = {met: float(state[met]) for met in mets}

        t = start
        step_i = 0
        while t < now and step_i < MAX_SEED_STEPS:
            for met in mets:
                met_shock = rng.random() < 0.05
                state[met] = _perturb(met, state[met], baselines[met], rng, shock=met_shock)
                _insert_event(conn, t, mid, met, state[met])
            t = _next_ts(t, rng)
            step_i += 1

        set_meta(conn, f"last_tick_{mid}", str(now))

    set_meta(conn, "last_tick_global", str(now))
    trim_old_events(conn, now)


def ensure_seeded(conn, df: pd.DataFrame, path_str: str) -> None:
    """Seed if workbook changed or DB empty."""
    init_schema(conn)
    fp = workbook_fingerprint(path_str, df)
    prev = get_meta(conn, "workbook_fp", "")
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if prev != fp or n_events == 0:
        with store_lock:
            clear_all_data(conn)
            seed_history(conn, df, path_str)
            conn.commit()


def load_snapshot_dict(conn, model_id: str) -> dict[str, float]:
    rows = conn.execute(
        "SELECT metric, value FROM snapshot_latest WHERE model_id = ?",
        (str(model_id),),
    ).fetchall()
    return {str(r[0]): float(r[1]) for r in rows}


def tick_once(conn, df: pd.DataFrame, now_ts: float | None = None) -> int:
    """Advance simulation one step for a random subset of models. Returns event count."""
    with store_lock:
        now_ts = now_ts or time.time()
        if not is_running(conn):
            return 0
        rng = np.random.default_rng(int(now_ts * 1000) % (2**32))
        mids = df["model"].astype(str).tolist()
        if not mids:
            return 0
        k = min(len(mids), max(3, len(mids) // 4))
        if len(mids) <= 8:
            pick = mids
        else:
            pick = list(rng.choice(mids, size=k, replace=False))

        n = 0
        for mid in pick:
            srow = df.loc[df["model"].astype(str) == mid].iloc[0]
            mets = mlib.metrics_for_row(srow)
            snap = load_snapshot_dict(conn, mid)
            for met in mets:
                baseline = mlib.initial_value_for_row(srow, met)
                cur = snap.get(met)
                if cur is None:
                    cur = baseline
                met_shock = rng.random() < 0.05
                nv = _perturb(met, float(cur), baseline, rng, shock=met_shock)
                _insert_event(conn, now_ts, mid, met, nv)
                n += 1

        set_meta(conn, "last_tick_global", str(now_ts))
        trim_old_events(conn, now_ts)
        conn.commit()
        return n


def catch_up_ticks(conn, df: pd.DataFrame) -> None:
    """Advance up to MAX_TICKS_PER_CATCHUP based on wall clock (complements background thread)."""
    if not is_running(conn):
        return
    last = float(get_meta(conn, "last_tick_global", "0") or 0)
    now = time.time()
    if last <= 0:
        last = now - TICK_INTERVAL_SEC
    elapsed = now - last
    if elapsed < TICK_INTERVAL_SEC:
        return
    n_steps = min(MAX_TICKS_PER_CATCHUP, int(elapsed / TICK_INTERVAL_SEC))
    for i in range(n_steps):
        tick_once(conn, df, now_ts=last + TICK_INTERVAL_SEC * (i + 1))


def _writer_loop() -> None:
    while not _writer_stop.is_set():
        time.sleep(TICK_INTERVAL_SEC)
        try:
            c = connect()
            init_schema(c)
            if not is_running(c):
                c.close()
                continue
            path = get_meta(c, "workbook_path")
            if not path:
                c.close()
                continue
            try:
                df = load_enriched_post(path)
            except Exception:
                c.close()
                continue
            tick_once(c, df)
            c.close()
        except Exception:
            pass


def ensure_background_writer() -> None:
    global _writer_thread
    if _writer_thread is not None and _writer_thread.is_alive():
        return
    _writer_stop.clear()
    _writer_thread = threading.Thread(target=_writer_loop, name="live_telemetry_writer", daemon=True)
    _writer_thread.start()


def stop_background_writer() -> None:
    _writer_stop.set()


def reset_simulation(conn, df: pd.DataFrame, path_str: str, seed: int = 42) -> None:
    with store_lock:
        set_running(conn, False)
        clear_all_data(conn)
        seed_history(conn, df, path_str, rng_base=seed)
        set_meta(conn, "workbook_path", path_str)
        conn.commit()


def register_workbook_path(conn, path_str: str) -> None:
    set_meta(conn, "workbook_path", path_str)


def fetch_recent_events(conn, limit: int = 500) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT ts, model_id, metric, value FROM events
        ORDER BY ts DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "model", "metric", "value"])
    df = pd.DataFrame(rows, columns=["ts", "model", "metric", "value"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df[["timestamp", "model", "metric", "value"]].sort_values("timestamp", ascending=False)


def fetch_series(
    conn,
    model_id: str,
    metrics: list[str],
    *,
    since_ts: float | None = None,
) -> pd.DataFrame:
    """Long-format series for charts."""
    since_ts = since_ts or (time.time() - HORIZON_SEC)
    if metrics:
        ph = ",".join("?" * len(metrics))
        q = f"""
            SELECT ts, metric, value FROM events
            WHERE model_id = ? AND ts >= ? AND metric IN ({ph})
            ORDER BY ts ASC
        """
        args: tuple = (str(model_id), since_ts, *metrics)
    else:
        q = """
            SELECT ts, metric, value FROM events
            WHERE model_id = ? AND ts >= ?
            ORDER BY ts ASC
        """
        args = (str(model_id), since_ts)
    rows = conn.execute(q, args).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts", "metric", "value"])
    return pd.DataFrame(rows, columns=["ts", "metric", "value"])


atexit.register(stop_background_writer)
