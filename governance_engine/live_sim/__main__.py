"""Run the live telemetry writer loop without Streamlit (daemon / dev helper)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance_engine.data_bridge import load_enriched_post
from governance_engine.live_sim.simulator import (
    ensure_seeded,
    is_running,
    register_workbook_path,
    set_running,
    tick_once,
)
from governance_engine.live_sim.store import connect, init_schema


def main() -> None:
    p = argparse.ArgumentParser(description="Advance live telemetry SQLite store on an interval.")
    p.add_argument("workbook", nargs="?", help="Path to enriched workbook (xlsx)")
    p.add_argument("--interval", type=float, default=2.0, help="Seconds between ticks")
    args = p.parse_args()
    if not args.workbook:
        print("usage: python -m governance_engine.live_sim <workbook.xlsx>", file=sys.stderr)
        sys.exit(2)
    path = str(Path(args.workbook).resolve())
    df = load_enriched_post(path)
    conn = connect()
    init_schema(conn)
    ensure_seeded(conn, df, path)
    register_workbook_path(conn, path)
    set_running(conn, True)
    print(f"Live sim running for {len(df)} models; Ctrl+C to stop.")
    try:
        while True:
            if not is_running(conn):
                break
            tick_once(conn, df)
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        set_running(conn, False)
        print("Stopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
