"""CLI: run governance engine over enriched workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from governance_engine.data_bridge import DEFAULT_ENRICHED_PATH, SHEET_POST, load_enriched_post
from governance_engine.engine import decision_to_jsonable, run_governance_for_row
from governance_engine.expectations import load_expectations


def main() -> None:
    p = argparse.ArgumentParser(description="Observability governance POC engine")
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_ENRICHED_PATH,
        help="Path to enriched_monitoring_report.xlsx",
    )
    p.add_argument(
        "--sheet",
        default=SHEET_POST,
        help="Worksheet name (default: post-go-live)",
    )
    p.add_argument(
        "--expectations",
        type=Path,
        default=None,
        help="Path to expectations YAML",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSONL decisions to this path",
    )
    args = p.parse_args()

    cfg = load_expectations(args.expectations) if args.expectations else load_expectations()
    df = load_enriched_post(args.input, sheet_name=args.sheet)
    lines: list[str] = []
    for _, row in df.iterrows():
        dec = run_governance_for_row(row, cfg)
        lines.append(json.dumps(decision_to_jsonable(dec), default=str))

    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(lines)} decisions to {args.out}")
    else:
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
