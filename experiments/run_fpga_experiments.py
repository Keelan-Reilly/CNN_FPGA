#!/usr/bin/env python3
"""Config-driven FPGA Vivado experiment runner."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "fpga" / "vivado" / "run_batch.sh"
PARSE_SCRIPT = REPO_ROOT / "fpga" / "vivado" / "parse_reports.py"
PERF_SCRIPT = REPO_ROOT / "experiments" / "collect_verilator_perf.py"

SUPPORTED_SWEEP_PARAMS = {"DATA_WIDTH", "FRAC_BITS", "CLK_FREQ_HZ", "BAUD_RATE", "DENSE_OUT_PAR"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def merge_params(base_params: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_params)
    merged.update(override)
    return merged


def expand_runs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base_params = cfg.get("base_params", {})
    runs: list[dict[str, Any]] = []

    explicit_runs = cfg.get("runs", [])
    for idx, item in enumerate(explicit_runs, start=1):
        params = merge_params(base_params, item.get("params", {}))
        run_id = sanitize_id(item.get("run_id", f"run_{idx:03d}"))
        runs.append(
            {
                "run_id": run_id,
                "params": params,
                "clock_period_ns": item.get("clock_period_ns", cfg.get("clock_period_ns")),
            }
        )

    sweep = cfg.get("sweep_parameters", {})
    if sweep:
        keys = sorted(sweep.keys())
        values = [sweep[k] for k in keys]
        for combo in itertools.product(*values):
            override = dict(zip(keys, combo))
            params = merge_params(base_params, override)
            parts = [f"{k}{params[k]}" for k in keys]
            run_id = sanitize_id("sweep_" + "_".join(parts))
            runs.append(
                {
                    "run_id": run_id,
                    "params": params,
                    "clock_period_ns": cfg.get("clock_period_ns"),
                }
            )

    if not runs:
        runs.append(
            {
                "run_id": "baseline",
                "params": base_params,
                "clock_period_ns": cfg.get("clock_period_ns"),
            }
        )

    return runs


def run_vivado(run_dir: Path, vivado_cfg: dict[str, Any], params: dict[str, Any], clock_period_ns: Any) -> tuple[int, str]:
    cmd = [
        str(RUN_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--repo-root",
        str(REPO_ROOT),
        "--part",
        str(vivado_cfg["part"]),
        "--top",
        str(vivado_cfg.get("top", "top_level")),
        "--xdc",
        str(vivado_cfg.get("xdc", "CNN_constraints.xdc")),
        "--jobs",
        str(vivado_cfg.get("jobs", 4)),
    ]

    if clock_period_ns is not None:
        cmd += ["--clock-period-ns", str(clock_period_ns)]

    for key in sorted(params.keys()):
        if key in SUPPORTED_SWEEP_PARAMS:
            cmd += ["--generic", f"{key}={params[key]}"]

    log_path = run_dir / "vivado.log"
    with log_path.open("w") as logf:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT).returncode

    return rc, " ".join(cmd)


def parse_metrics(run_dir: Path, clock_period_ns: Any) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    cmd = [
        sys.executable,
        str(PARSE_SCRIPT),
        "--reports-dir",
        str(run_dir / "reports"),
        "--output",
        str(metrics_path),
    ]
    if clock_period_ns is not None:
        cmd += ["--clock-period-ns", str(clock_period_ns)]

    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if rc != 0:
        metrics = {
            "utilization": {"lut": None, "ff": None, "dsp": None, "bram": None},
            "timing": {
                "wns_ns": None,
                "fmax_mhz_est": None,
                "clock_period_target_ns": clock_period_ns,
                "bottleneck_summary": None,
            },
            "reports_present": {},
            "parse_error": True,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics

    return json.loads(metrics_path.read_text())


def collect_performance(run_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    perf_dir = run_dir / "verilator_perf"
    perf_json = perf_dir / "performance.json"
    perf_dir.mkdir(parents=True, exist_ok=True)
    clk_hz = int(params.get("CLK_FREQ_HZ", 100_000_000))
    cmd = [
        sys.executable,
        str(PERF_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--clock-hz",
        str(clk_hz),
    ]
    for key in sorted(params.keys()):
        if key in SUPPORTED_SWEEP_PARAMS:
            cmd += ["--generic", f"{key}={params[key]}"]

    helper_log = perf_dir / "collect_verilator_perf.log"
    with helper_log.open("w") as logf:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not perf_json.exists():
        return {
            "latency_cycles": None,
            "latency_time_ms": None,
            "throughput_inferences_per_sec": None,
            "stage_cycles_conv": None,
            "stage_cycles_relu": None,
            "stage_cycles_pool": None,
            "stage_cycles_dense": None,
            "stage_cycles_argmax": None,
            "bubble_cycles": None,
            "busy_cycles": None,
            "tx_wait_cycles": None,
            "measurement_source": "verilator_full_pipeline_frame_cycles",
            "latency_kind": "compute_only",
            "clock_hz": clk_hz,
            "measured_fields": [
                "latency_cycles",
                "stage_cycles_conv",
                "stage_cycles_relu",
                "stage_cycles_pool",
                "stage_cycles_dense",
                "stage_cycles_argmax",
                "bubble_cycles",
                "busy_cycles",
                "tx_wait_cycles",
            ],
            "derived_fields": ["latency_time_ms", "throughput_inferences_per_sec"],
            "perf_error": True,
        }
    return json.loads(perf_json.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def aggregate(experiment_id: str, rows: list[dict[str, Any]]) -> None:
    out_dir = REPO_ROOT / "results" / "fpga" / "aggregates"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{experiment_id}.json"
    csv_path = out_dir / f"{experiment_id}.csv"

    write_json(json_path, {"experiment_id": experiment_id, "runs": rows})

    fields = [
        "experiment_id",
        "run_id",
        "status",
        "returncode",
        "part",
        "top",
        "clock_period_ns",
        "started_utc",
        "ended_utc",
        "duration_sec",
        "lut",
        "ff",
        "dsp",
        "bram",
        "wns_ns",
        "fmax_mhz_est",
        "timing_bottleneck",
        "latency_cycles",
        "latency_time_ms",
        "throughput_inferences_per_sec",
        "stage_cycles_conv",
        "stage_cycles_relu",
        "stage_cycles_pool",
        "stage_cycles_dense",
        "stage_cycles_argmax",
        "bubble_cycles",
        "busy_cycles",
        "tx_wait_cycles",
        "measurement_source",
        "latency_kind",
        "clock_hz",
        "run_dir",
        "params",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Vivado FPGA experiments from JSON config")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--experiment-id", default=None, help="Override experiment_id in config")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failed run")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    experiment_id = sanitize_id(args.experiment_id or cfg.get("experiment_id", "fpga_experiment"))

    vivado_cfg = cfg.get("vivado", {})
    if "part" not in vivado_cfg:
        raise SystemExit("Config missing vivado.part")

    runs = expand_runs(cfg)
    root = REPO_ROOT / "results" / "fpga" / "runs" / experiment_id
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for run in runs:
        run_id = sanitize_id(run["run_id"])
        run_dir = root / run_id
        if run_dir.exists():
            run_id = f"{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        params = run["params"]
        unsupported = sorted(k for k in params.keys() if k not in SUPPORTED_SWEEP_PARAMS)

        resolved = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "params": params,
            "clock_period_ns": run.get("clock_period_ns"),
            "vivado": vivado_cfg,
            "unsupported_params_ignored": unsupported,
            "supported_params": sorted(SUPPORTED_SWEEP_PARAMS),
        }
        write_json(run_dir / "config_resolved.json", resolved)

        started = utc_now()
        t0 = time.time()
        rc, cmd = run_vivado(run_dir, vivado_cfg, params, run.get("clock_period_ns"))
        duration = time.time() - t0
        ended = utc_now()

        metrics = parse_metrics(run_dir, run.get("clock_period_ns"))
        performance = collect_performance(run_dir, params)
        metrics["performance"] = performance
        write_json(run_dir / "metrics.json", metrics)
        status = "succeeded" if rc == 0 else "failed"

        run_meta = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "status": status,
            "returncode": rc,
            "started_utc": started,
            "ended_utc": ended,
            "duration_sec": round(duration, 3),
            "host": socket.gethostname(),
            "command": cmd,
        }
        write_json(run_dir / "run_meta.json", run_meta)

        util = metrics.get("utilization", {})
        timing = metrics.get("timing", {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "status": status,
                "returncode": rc,
                "part": vivado_cfg.get("part", ""),
                "top": vivado_cfg.get("top", "top_level"),
                "clock_period_ns": run.get("clock_period_ns"),
                "started_utc": started,
                "ended_utc": ended,
                "duration_sec": round(duration, 3),
                "lut": util.get("lut"),
                "ff": util.get("ff"),
                "dsp": util.get("dsp"),
                "bram": util.get("bram"),
                "wns_ns": timing.get("wns_ns"),
                "fmax_mhz_est": timing.get("fmax_mhz_est"),
                "timing_bottleneck": timing.get("bottleneck_summary"),
                "latency_cycles": performance.get("latency_cycles"),
                "latency_time_ms": performance.get("latency_time_ms"),
                "throughput_inferences_per_sec": performance.get("throughput_inferences_per_sec"),
                "stage_cycles_conv": performance.get("stage_cycles_conv"),
                "stage_cycles_relu": performance.get("stage_cycles_relu"),
                "stage_cycles_pool": performance.get("stage_cycles_pool"),
                "stage_cycles_dense": performance.get("stage_cycles_dense"),
                "stage_cycles_argmax": performance.get("stage_cycles_argmax"),
                "bubble_cycles": performance.get("bubble_cycles"),
                "busy_cycles": performance.get("busy_cycles"),
                "tx_wait_cycles": performance.get("tx_wait_cycles"),
                "measurement_source": performance.get("measurement_source"),
                "latency_kind": performance.get("latency_kind"),
                "clock_hz": performance.get("clock_hz"),
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "params": json.dumps(params, sort_keys=True),
            }
        )

        print(f"[{status}] {run_id} -> {run_dir}")
        if rc != 0 and args.fail_fast:
            break

    aggregate(experiment_id, rows)
    print(f"Wrote aggregates: results/fpga/aggregates/{experiment_id}.json/.csv")

    return 0 if all(r["status"] == "succeeded" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
