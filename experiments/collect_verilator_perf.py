#!/usr/bin/env python3
"""Collect compute-latency metrics from the Verilator full-pipeline flow."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOP_MODULE = "top_level"
MEASUREMENT_SOURCE = "verilator_full_pipeline_frame_cycles"
LATENCY_KIND = "compute_only"
STAGE_COUNTER_KEYS = [
    "stage_cycles_conv",
    "stage_cycles_relu",
    "stage_cycles_pool",
    "stage_cycles_dense",
    "stage_cycles_argmax",
    "bubble_cycles",
    "busy_cycles",
    "tx_wait_cycles",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect Verilator performance metrics for one architecture point")
    ap.add_argument("--run-dir", required=True, help="Experiment run directory")
    ap.add_argument("--clock-hz", required=True, type=int, help="Clock frequency used for time/throughput derivation")
    ap.add_argument("--top", default=TOP_MODULE, help="Top module name (default: top_level)")
    ap.add_argument("--generic", action="append", default=[], help="Top-level Verilog generic KEY=VALUE")
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


def image_mem_to_bin(src_mem: Path, out_bin: Path) -> None:
    values = []
    for line in src_mem.read_text().splitlines():
        token = line.strip()
        if not token:
            continue
        values.append(int(token, 16) & 0xFF)
    out_bin.write_bytes(bytes(values))


def parse_frame_cycles(log_text: str) -> int:
    match = re.search(r"Frame cycles:\s*(\d+)", log_text)
    if not match:
        raise ValueError("Could not find 'Frame cycles:' in Verilator log output")
    return int(match.group(1))


def parse_perf_metrics(log_text: str) -> dict[str, int]:
    found = {
        key: int(value)
        for key, value in re.findall(r"PERF_METRIC\s+([A-Za-z0-9_]+)=(\d+)", log_text)
    }
    missing = [key for key in STAGE_COUNTER_KEYS if key not in found]
    if missing:
        raise ValueError(f"Missing PERF_METRIC fields in Verilator log: {', '.join(missing)}")
    return {key: found[key] for key in STAGE_COUNTER_KEYS}


def main() -> int:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    build_dir = run_dir / "verilator_build"
    perf_dir = run_dir / "verilator_perf"
    build_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    input_mem = REPO_ROOT / "weights" / "input_image.mem"
    if not input_mem.exists():
        raise SystemExit(f"Missing deterministic input image source: {input_mem}")

    input_bin = perf_dir / "input_image.bin"
    image_mem_to_bin(input_mem, input_bin)

    hdl_files = sorted((REPO_ROOT / "hdl").glob("*.sv"))
    if not hdl_files:
        raise SystemExit("No HDL files found under hdl/")

    generic_pairs = [parse_generic(g) for g in args.generic]
    verilator_cmd = [
        "verilator",
        "-sv",
        "-Wall",
        "-Wno-fatal",
        "--trace",
        "-CFLAGS",
        "-std=c++17",
        "--Mdir",
        str(build_dir),
        "--cc",
        *[str(p) for p in hdl_files],
        "--top-module",
        args.top,
        "--exe",
        str(REPO_ROOT / "tb" / "tb_full_pipeline.cpp"),
    ]
    for key, value in generic_pairs:
        verilator_cmd.append(f"-G{key}={value}")

    subprocess.run(verilator_cmd, cwd=REPO_ROOT, check=True)
    subprocess.run(["make", "-C", str(build_dir), "-f", f"V{args.top}.mk", "-j"], cwd=REPO_ROOT, check=True)

    sim_bin = build_dir / f"V{args.top}"
    sim_cmd = [str(sim_bin), "--images", str(input_bin), "--n", "1"]

    log_path = perf_dir / "verilator_perf.log"
    with log_path.open("w") as logf:
        rc = subprocess.run(sim_cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"Verilator performance run failed with exit code {rc}")

    log_text = log_path.read_text(errors="ignore")
    latency_cycles = parse_frame_cycles(log_text)
    perf_metrics = parse_perf_metrics(log_text)

    latency_time_ms = (1000.0 * latency_cycles) / args.clock_hz
    throughput_ips = args.clock_hz / latency_cycles

    payload = {
        "latency_cycles": latency_cycles,
        "latency_time_ms": latency_time_ms,
        "throughput_inferences_per_sec": throughput_ips,
        "measurement_source": MEASUREMENT_SOURCE,
        "latency_kind": LATENCY_KIND,
        "clock_hz": args.clock_hz,
        "measured_fields": ["latency_cycles", *STAGE_COUNTER_KEYS],
        "derived_fields": ["latency_time_ms", "throughput_inferences_per_sec"],
        "sim_log": str(log_path.relative_to(run_dir)),
    }
    payload.update(perf_metrics)

    out_path = perf_dir / "performance.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
