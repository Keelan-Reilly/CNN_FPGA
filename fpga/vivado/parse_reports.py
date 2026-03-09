#!/usr/bin/env python3
"""Parse Vivado reports into machine-readable metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def _to_number(token: str) -> Optional[float]:
    token = token.replace(",", "").strip()
    if token in {"", "N/A"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _search_first(text: str, patterns: list[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            return _to_number(m.group(1))
    return None


def parse_utilization(util_text: str) -> dict:
    lut = _search_first(
        util_text,
        [
            r"\|\s*CLB LUTs\*?\s*\|\s*([0-9,\.]+)\s*\|",
            r"\|\s*Slice LUTs\s*\|\s*([0-9,\.]+)\s*\|",
            r"\|\s*LUT as Logic\s*\|\s*([0-9,\.]+)\s*\|",
            r"CLB LUTs\*?\s*\|\s*([0-9,\.]+)",
        ],
    )
    ff = _search_first(
        util_text,
        [
            r"\|\s*CLB Registers\s*\|\s*([0-9,\.]+)\s*\|",
            r"\|\s*Slice Registers\s*\|\s*([0-9,\.]+)\s*\|",
            r"CLB Registers\s*\|\s*([0-9,\.]+)",
        ],
    )
    dsp = _search_first(util_text, [r"\|\s*DSPs\s*\|\s*([0-9,\.]+)\s*\|"])

    bram_tile = _search_first(util_text, [r"\|\s*Block RAM Tile\s*\|\s*([0-9,\.]+)\s*\|"])
    if bram_tile is None:
        ramb36 = _search_first(util_text, [r"\|\s*RAMB36(?:E1|E2)?\s*\|\s*([0-9,\.]+)\s*\|"])
        ramb18 = _search_first(util_text, [r"\|\s*RAMB18(?:E1|E2)?\s*\|\s*([0-9,\.]+)\s*\|"])
        if ramb36 is not None or ramb18 is not None:
            bram_tile = (ramb36 or 0.0) + (ramb18 or 0.0) / 2.0

    return {
        "lut": int(lut) if lut is not None else None,
        "ff": int(ff) if ff is not None else None,
        "dsp": int(dsp) if dsp is not None else None,
        "bram": bram_tile,
    }


def parse_timing_summary(text: str) -> dict:
    wns = None

    table_match = re.search(
        r"WNS\(ns\).*?\n\s*[-+| ]+\n\s*\|\s*([-+]?\d+(?:\.\d+)?)\s*\|",
        text,
        flags=re.DOTALL,
    )
    if table_match:
        wns = _to_number(table_match.group(1))

    if wns is None:
        wns = _search_first(text, [r"WNS\(ns\)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)"])

    return {"wns_ns": wns}


def parse_timing_bottleneck(path_text: str) -> Optional[str]:
    if not path_text:
        return None

    slack = None
    startpoint = None
    endpoint = None
    delay = None

    m = re.search(r"Slack\s*\([^)]*\)\s*:\s*([-+]?\d+(?:\.\d+)?)", path_text)
    if m:
        slack = m.group(1)

    m = re.search(r"Startpoint:\s*(.+)", path_text)
    if m:
        startpoint = m.group(1).strip()

    m = re.search(r"Endpoint:\s*(.+)", path_text)
    if m:
        endpoint = m.group(1).strip()

    m = re.search(r"Data Path Delay:\s*([-+]?\d+(?:\.\d+)?)ns", path_text)
    if m:
        delay = m.group(1)

    fields = []
    if slack is not None:
        fields.append(f"slack={slack}ns")
    if delay is not None:
        fields.append(f"delay={delay}ns")
    if startpoint:
        fields.append(f"start={startpoint}")
    if endpoint:
        fields.append(f"end={endpoint}")

    return "; ".join(fields) if fields else None


def parse_reports(reports_dir: Path, clock_period_ns: Optional[float]) -> dict:
    post_route_util = _read_text(reports_dir / "post_route_utilization.rpt")
    post_route_timing = _read_text(reports_dir / "post_route_timing_summary.rpt")
    post_route_paths = _read_text(reports_dir / "post_route_timing_paths.rpt")

    util = parse_utilization(post_route_util)
    timing = parse_timing_summary(post_route_timing)
    bottleneck = parse_timing_bottleneck(post_route_paths)

    wns = timing.get("wns_ns")
    fmax_mhz = None
    if clock_period_ns is not None and wns is not None:
        effective_period = clock_period_ns - wns
        if effective_period > 0:
            fmax_mhz = 1000.0 / effective_period

    return {
        "utilization": util,
        "timing": {
            "wns_ns": wns,
            "fmax_mhz_est": fmax_mhz,
            "clock_period_target_ns": clock_period_ns,
            "bottleneck_summary": bottleneck,
        },
        "reports_present": {
            "post_route_utilization": bool(post_route_util),
            "post_route_timing_summary": bool(post_route_timing),
            "post_route_timing_paths": bool(post_route_paths),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse Vivado reports into metrics JSON")
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--clock-period-ns", type=float, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    out_path = Path(args.output)

    metrics = parse_reports(reports_dir, args.clock_period_ns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
