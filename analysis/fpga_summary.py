#!/usr/bin/env python3
"""CLI summary for FPGA aggregate datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fpga_results import load_aggregate, resolve_aggregate_path


def _compact_params(params: dict[str, Any], max_len: int = 56) -> str:
    if not params:
        return "-"
    txt = ",".join(f"{k}={params[k]}" for k in sorted(params.keys()))
    return txt if len(txt) <= max_len else txt[: max_len - 3] + "..."


def _fmt(val: Any, ndigits: int = 3) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{ndigits}f}"
    return str(val)


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "run_id",
        "status",
        "return",
        "params",
        "lut",
        "ff",
        "dsp",
        "bram",
        "wns_ns",
        "fmax_mhz",
    ]

    table_rows = []
    for r in rows:
        table_rows.append(
            [
                str(r.get("run_id", "-")),
                str(r.get("status", "-")),
                _fmt(r.get("returncode"), 0),
                _compact_params(r.get("params", {})),
                _fmt(r.get("lut"), 0),
                _fmt(r.get("ff"), 0),
                _fmt(r.get("dsp"), 0),
                _fmt(r.get("bram"), 2),
                _fmt(r.get("wns_ns"), 3),
                _fmt(r.get("fmax_mhz_est"), 2),
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(items: list[str]) -> str:
        return "  ".join(items[i].ljust(widths[i]) for i in range(len(items)))

    print(line(headers))
    print(line(["-" * w for w in widths]))
    for row in table_rows:
        print(line(row))


def main() -> int:
    ap = argparse.ArgumentParser(description="Print concise FPGA experiment summary")
    ap.add_argument("--experiment-id", default="baseline_fpga", help="Experiment id (default: baseline_fpga)")
    ap.add_argument("--aggregate", default=None, help="Path to aggregate .json/.csv")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--max-rows", type=int, default=200)
    args = ap.parse_args()

    try:
        aggregate_path = resolve_aggregate_path(args.experiment_id, args.aggregate)
        experiment_id, rows = load_aggregate(aggregate_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Could not load aggregate dataset: {exc}")

    if not args.include_failed:
        rows = [r for r in rows if r.get("status") == "succeeded"]

    rows = rows[: args.max_rows]

    print(f"experiment_id: {experiment_id}")
    print(f"aggregate: {aggregate_path}")
    print(f"rows: {len(rows)}")
    if not rows:
        print("No rows to display.")
        return 0

    status_counts: dict[str, int] = {}
    for r in rows:
        s = str(r.get("status", "unknown"))
        status_counts[s] = status_counts.get(s, 0) + 1
    print("status_counts:", ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    print()

    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
