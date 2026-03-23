#!/usr/bin/env python3
"""Config-driven FPGA Vivado experiment runner."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vivado_scheduler import SchedulerConfig, can_launch_more, scheduler_summary_rows, snapshot_resources, wait_for_capacity


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "fpga" / "vivado" / "run_batch.sh"
PARSE_SCRIPT = REPO_ROOT / "fpga" / "vivado" / "parse_reports.py"
PERF_SCRIPT = REPO_ROOT / "experiments" / "collect_verilator_perf.py"

SUPPORTED_SWEEP_PARAMS = {
    "DATA_WIDTH",
    "FRAC_BITS",
    "CLK_FREQ_HZ",
    "BAUD_RATE",
    "DENSE_OUT_PAR",
    "ARRAY_ROWS",
    "ARRAY_COLS",
    "K_DEPTH",
}


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


def build_vivado_cmd(run_dir: Path, vivado_cfg: dict[str, Any], params: dict[str, Any], clock_period_ns: Any) -> list[str]:
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

    return cmd


def run_vivado(run_dir: Path, vivado_cfg: dict[str, Any], params: dict[str, Any], clock_period_ns: Any) -> tuple[int, str]:
    cmd = build_vivado_cmd(run_dir, vivado_cfg, params, clock_period_ns)
    log_path = run_dir / "vivado.log"
    with log_path.open("w") as logf:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT).returncode

    return rc, " ".join(cmd)


def start_vivado_process(run_dir: Path, vivado_cfg: dict[str, Any], params: dict[str, Any], clock_period_ns: Any) -> tuple[subprocess.Popen[bytes], str]:
    cmd = build_vivado_cmd(run_dir, vivado_cfg, params, clock_period_ns)
    log_path = run_dir / "vivado.log"
    logf = log_path.open("w")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)
    return proc, " ".join(cmd)


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


def _empty_performance(clock_hz: int, measurement_source: str, latency_kind: str) -> dict[str, Any]:
    return {
        "latency_cycles": None,
        "latency_time_ms": None,
        "throughput_inferences_per_sec": None,
        "effective_throughput_ops_per_cycle": None,
        "total_operations": None,
        "stage_cycles_conv": None,
        "stage_cycles_relu": None,
        "stage_cycles_pool": None,
        "stage_cycles_dense": None,
        "stage_cycles_argmax": None,
        "bubble_cycles": None,
        "busy_cycles": None,
        "tx_wait_cycles": None,
        "measurement_source": measurement_source,
        "latency_kind": latency_kind,
        "clock_hz": clock_hz,
        "measured_fields": ["latency_cycles"],
        "derived_fields": ["latency_time_ms", "throughput_inferences_per_sec", "effective_throughput_ops_per_cycle"],
        "perf_error": True,
    }


def collect_performance(run_dir: Path, params: dict[str, Any], perf_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    perf_cfg = perf_cfg or {}
    if perf_cfg.get("enabled", True) is False or perf_cfg.get("script") == "none":
        return _empty_performance(
            int(params.get("CLK_FREQ_HZ", 100_000_000)),
            "performance_collection_disabled",
            "not_collected",
        )

    perf_dir = run_dir / "verilator_perf"
    perf_json = perf_dir / "performance.json"
    perf_dir.mkdir(parents=True, exist_ok=True)
    clk_hz = int(params.get("CLK_FREQ_HZ", 100_000_000))
    perf_script = REPO_ROOT / perf_cfg.get("script", str(PERF_SCRIPT.relative_to(REPO_ROOT)))
    perf_top = perf_cfg.get("top", "top_level")
    cmd = [
        sys.executable,
        str(perf_script),
        "--run-dir",
        str(run_dir),
        "--clock-hz",
        str(clk_hz),
        "--top",
        str(perf_top),
    ]
    for key in sorted(params.keys()):
        if key in SUPPORTED_SWEEP_PARAMS:
            cmd += ["--generic", f"{key}={params[key]}"]

    helper_log = perf_dir / f"{perf_script.stem}.log"
    with helper_log.open("w") as logf:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not perf_json.exists():
        return _empty_performance(
            clk_hz,
            perf_cfg.get("measurement_source", "verilator_full_pipeline_frame_cycles"),
            perf_cfg.get("latency_kind", "compute_only"),
        )
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
        "effective_throughput_ops_per_cycle",
        "total_operations",
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


def prepare_run_root(experiment_id: str, runs: list[dict[str, Any]], vivado_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    root = REPO_ROOT / "results" / "fpga" / "runs" / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
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
        prepared.append(
            {
                "run_id": run_id,
                "run_dir": run_dir,
                "params": params,
                "clock_period_ns": run.get("clock_period_ns"),
            }
        )
    return prepared


def finalize_run(
    experiment_id: str,
    run_id: str,
    run_dir: Path,
    vivado_cfg: dict[str, Any],
    params: dict[str, Any],
    clock_period_ns: Any,
    rc: int,
    cmd: str,
    started: str,
    ended: str,
    duration: float,
    perf_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = parse_metrics(run_dir, clock_period_ns)
    performance = collect_performance(run_dir, params, perf_cfg)
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
    row = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "status": status,
        "returncode": rc,
        "part": vivado_cfg.get("part", ""),
        "top": vivado_cfg.get("top", "top_level"),
        "clock_period_ns": clock_period_ns,
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
        "effective_throughput_ops_per_cycle": performance.get("effective_throughput_ops_per_cycle"),
        "total_operations": performance.get("total_operations"),
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
    print(f"[{status}] {run_id} -> {run_dir}")
    return row


def scheduler_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(message + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Vivado FPGA experiments from JSON config")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--experiment-id", default=None, help="Override experiment_id in config")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failed run")
    ap.add_argument("--scheduler", choices=["serial", "resource-aware"], default="serial")
    ap.add_argument("--dry-run", action="store_true", help="Preview run queue without launching Vivado")
    ap.add_argument("--max-concurrent-jobs", type=int, default=1)
    ap.add_argument("--cpu-threshold-pct", type=float, default=85.0)
    ap.add_argument("--min-free-mem-gb", type=float, default=4.0)
    ap.add_argument("--per-job-mem-gb", type=float, default=8.0)
    ap.add_argument("--poll-interval-sec", type=float, default=5.0)
    ap.add_argument("--vivado-jobs-override", type=int, default=None, help="Override Vivado threads/jobs per run")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    perf_cfg = cfg.get("performance", {})
    experiment_id = sanitize_id(args.experiment_id or cfg.get("experiment_id", "fpga_experiment"))

    vivado_cfg = cfg.get("vivado", {})
    if "part" not in vivado_cfg:
        raise SystemExit("Config missing vivado.part")
    if args.vivado_jobs_override is not None:
        vivado_cfg = dict(vivado_cfg)
        vivado_cfg["jobs"] = int(args.vivado_jobs_override)

    runs = expand_runs(cfg)
    prepared_runs = prepare_run_root(experiment_id, runs, vivado_cfg)
    scheduler_cfg = SchedulerConfig(
        enabled=args.scheduler == "resource-aware",
        dry_run=args.dry_run,
        max_concurrent_jobs=max(1, args.max_concurrent_jobs),
        cpu_utilization_threshold_pct=args.cpu_threshold_pct,
        min_free_mem_gb=args.min_free_mem_gb,
        per_job_mem_gb=args.per_job_mem_gb,
        poll_interval_sec=args.poll_interval_sec,
    )
    queue_preview = [
        {
            "run_id": run["run_id"],
            "run_dir": str(run["run_dir"].relative_to(REPO_ROOT)),
            "clock_period_ns": run["clock_period_ns"],
        }
        for run in prepared_runs
    ]
    scheduler_rows = scheduler_summary_rows(queue_preview, scheduler_cfg)
    scheduler_dir = REPO_ROOT / "results" / "fpga" / "runs" / experiment_id
    write_json(scheduler_dir / "scheduler_queue.json", scheduler_rows)
    with (scheduler_dir / "scheduler_queue.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scheduler_rows[0].keys()) if scheduler_rows else ["queue_index"])
        writer.writeheader()
        for row in scheduler_rows:
            writer.writerow(row)

    if args.dry_run:
        print(f"Dry-run queue for {experiment_id}:")
        for row in scheduler_rows:
            print(f"  [{row['queue_index']}] {row['run_id']} -> {row['run_dir']}")
        return 0

    rows: list[dict[str, Any]] = []
    scheduler_log_path = scheduler_dir / "scheduler.log"
    scheduler_log(scheduler_log_path, f"experiment_id={experiment_id} scheduler={args.scheduler}")

    if not scheduler_cfg.enabled:
        for run in prepared_runs:
            started = utc_now()
            t0 = time.time()
            rc, cmd = run_vivado(run["run_dir"], vivado_cfg, run["params"], run["clock_period_ns"])
            duration = time.time() - t0
            ended = utc_now()
            rows.append(
                finalize_run(
                    experiment_id=experiment_id,
                    run_id=run["run_id"],
                    run_dir=run["run_dir"],
                    vivado_cfg=vivado_cfg,
                    params=run["params"],
                    clock_period_ns=run["clock_period_ns"],
                    rc=rc,
                    cmd=cmd,
                    started=started,
                    ended=ended,
                    duration=duration,
                    perf_cfg=perf_cfg,
                )
            )
            if rc != 0 and args.fail_fast:
                break
    else:
        pending = list(prepared_runs)
        running: list[dict[str, Any]] = []
        while pending or running:
            while pending:
                snap = snapshot_resources()
                allowed, reason = can_launch_more(snap, len(running), scheduler_cfg)
                if not allowed:
                    scheduler_log(
                        scheduler_log_path,
                        f"launch_blocked reason={reason} running={len(running)} cpu={snap.cpu_utilization_pct:.1f} avail_mem_gb={snap.available_mem_gb:.2f}",
                    )
                    break
                run = pending.pop(0)
                started = utc_now()
                proc, cmd = start_vivado_process(run["run_dir"], vivado_cfg, run["params"], run["clock_period_ns"])
                running.append(
                    {
                        **run,
                        "proc": proc,
                        "cmd": cmd,
                        "started": started,
                        "t0": time.time(),
                    }
                )
                scheduler_log(
                    scheduler_log_path,
                    f"launched run_id={run['run_id']} pid={proc.pid} running={len(running)} pending={len(pending)}",
                )
                print(f"[queued] launched {run['run_id']} (pid={proc.pid})")

            if not running and pending:
                snap, _ = wait_for_capacity(scheduler_cfg, 0)
                scheduler_log(
                    scheduler_log_path,
                    f"capacity_ready cpu={snap.cpu_utilization_pct:.1f} avail_mem_gb={snap.available_mem_gb:.2f}",
                )
                continue

            finished_indices = []
            for idx, item in enumerate(running):
                rc = item["proc"].poll()
                if rc is None:
                    continue
                duration = time.time() - item["t0"]
                ended = utc_now()
                rows.append(
                    finalize_run(
                        experiment_id=experiment_id,
                        run_id=item["run_id"],
                        run_dir=item["run_dir"],
                        vivado_cfg=vivado_cfg,
                        params=item["params"],
                        clock_period_ns=item["clock_period_ns"],
                        rc=rc,
                        cmd=item["cmd"],
                        started=item["started"],
                        ended=ended,
                        duration=duration,
                        perf_cfg=perf_cfg,
                    )
                )
                scheduler_log(
                    scheduler_log_path,
                    f"completed run_id={item['run_id']} rc={rc} running={len(running)-1} pending={len(pending)}",
                )
                finished_indices.append(idx)
                if rc != 0 and args.fail_fast:
                    pending.clear()
            for idx in reversed(finished_indices):
                running.pop(idx)
            if pending or running:
                time.sleep(scheduler_cfg.poll_interval_sec)

    aggregate(experiment_id, rows)
    write_json(
        scheduler_dir / "scheduler_summary.json",
        {
            "experiment_id": experiment_id,
            "scheduler_mode": args.scheduler,
            "dry_run": args.dry_run,
            "max_concurrent_jobs": scheduler_cfg.max_concurrent_jobs,
            "cpu_threshold_pct": scheduler_cfg.cpu_utilization_threshold_pct,
            "min_free_mem_gb": scheduler_cfg.min_free_mem_gb,
            "per_job_mem_gb": scheduler_cfg.per_job_mem_gb,
            "queued_runs": len(prepared_runs),
            "completed_runs": len(rows),
            "failed_runs": sum(1 for row in rows if row["status"] != "succeeded"),
            "vivado_jobs_per_run": vivado_cfg.get("jobs", 4),
        },
    )
    print(f"Wrote aggregates: results/fpga/aggregates/{experiment_id}.json/.csv")

    return 0 if all(r["status"] == "succeeded" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
