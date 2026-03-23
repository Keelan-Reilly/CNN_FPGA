#!/usr/bin/env python3
"""Collect latency/throughput metrics for the direct MAC-array slice."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOP = "mac_array_direct_top"
MEASUREMENT_SOURCE = "verilator_mac_array_direct_slice"
LATENCY_KIND = "fixed_mac_array_workload"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect direct MAC-array slice metrics")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--clock-hz", required=True, type=int)
    ap.add_argument("--top", default=DEFAULT_TOP)
    ap.add_argument("--generic", action="append", default=[])
    return ap.parse_args()


def parse_generic(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Invalid generic '{text}', expected KEY=VALUE")
    key, value = text.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"Invalid generic '{text}', expected KEY=VALUE")
    return key, value


def parse_frame_cycles(log_text: str) -> int:
    match = re.search(r"Frame cycles:\s*(\d+)", log_text)
    if not match:
        raise ValueError("Could not find 'Frame cycles:' in direct MAC-array log")
    return int(match.group(1))


def parse_signature(log_text: str) -> int | None:
    match = re.search(r"Signature:\s*(\d+)", log_text)
    return int(match.group(1)) if match else None


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    build_dir = run_dir / "verilator_build"
    perf_dir = run_dir / "verilator_perf"
    build_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    hdl_files = sorted((REPO_ROOT / "hdl").glob("*.sv"))
    if not hdl_files:
        raise SystemExit("No HDL files found under hdl/")

    generic_pairs = [parse_generic(g) for g in args.generic]
    generic_map = {key: value for key, value in generic_pairs}
    rows = int(generic_map.get("ARRAY_ROWS", 4))
    cols = int(generic_map.get("ARRAY_COLS", 4))
    k_depth = int(generic_map.get("K_DEPTH", 16))

    verilator_cmd = [
        "verilator",
        "-sv",
        "-Wall",
        "-Wno-fatal",
        "-CFLAGS",
        "-std=c++17",
        "--Mdir",
        str(build_dir),
        "--cc",
        *[str(p) for p in hdl_files],
        "--top-module",
        args.top,
        "--exe",
        str(REPO_ROOT / "tb" / "tb_mac_array_direct.cpp"),
    ]
    for key, value in generic_pairs:
        verilator_cmd.append(f"-G{key}={value}")

    subprocess.run(verilator_cmd, cwd=REPO_ROOT, check=True)
    subprocess.run(["make", "-C", str(build_dir), "-f", f"V{args.top}.mk", "-j"], cwd=REPO_ROOT, check=True)

    sim_bin = build_dir / f"V{args.top}"
    log_path = perf_dir / "verilator_perf.log"
    with log_path.open("w") as logf:
        rc = subprocess.run([str(sim_bin)], cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"Direct MAC-array Verilator run failed with exit code {rc}")

    log_text = log_path.read_text(errors="ignore")
    latency_cycles = parse_frame_cycles(log_text)
    signature = parse_signature(log_text)
    total_ops = rows * cols * k_depth
    eff_tput = float(total_ops) / float(latency_cycles)
    payload = {
        "latency_cycles": latency_cycles,
        "latency_time_ms": (1000.0 * latency_cycles) / args.clock_hz,
        "throughput_inferences_per_sec": None,
        "effective_throughput_ops_per_cycle": eff_tput,
        "total_operations": total_ops,
        "measurement_source": MEASUREMENT_SOURCE,
        "latency_kind": LATENCY_KIND,
        "clock_hz": args.clock_hz,
        "measured_fields": ["latency_cycles", "total_operations"],
        "derived_fields": ["latency_time_ms", "effective_throughput_ops_per_cycle"],
        "signature": signature,
        "sim_log": str(log_path.relative_to(run_dir)),
    }
    out_path = perf_dir / "performance.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
