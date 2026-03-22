#!/usr/bin/env python3
"""Conservative resource-aware scheduler for Vivado experiment queues."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    import psutil  # type: ignore
except ModuleNotFoundError:
    psutil = None


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool
    dry_run: bool
    max_concurrent_jobs: int
    cpu_utilization_threshold_pct: float
    min_free_mem_gb: float
    per_job_mem_gb: float
    poll_interval_sec: float


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_utilization_pct: float
    available_mem_gb: float


def snapshot_resources() -> ResourceSnapshot:
    if psutil is not None:
        vm = psutil.virtual_memory()
        return ResourceSnapshot(
            cpu_utilization_pct=float(psutil.cpu_percent(interval=0.0)),
            available_mem_gb=float(vm.available) / (1024**3),
        )

    cpu_pct = 0.0
    if hasattr(os, "getloadavg"):
        load1, _, _ = os.getloadavg()
        cpu_count = max(1, os.cpu_count() or 1)
        cpu_pct = min(100.0, 100.0 * load1 / cpu_count)

    available_gb = 0.0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values = {}
        for line in meminfo.read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key.strip()] = raw.strip()
        if "MemAvailable" in values:
            available_kb = float(values["MemAvailable"].split()[0])
            available_gb = available_kb / (1024**2)

    return ResourceSnapshot(cpu_utilization_pct=cpu_pct, available_mem_gb=available_gb)


def can_launch_more(
    snapshot: ResourceSnapshot,
    running_jobs: int,
    cfg: SchedulerConfig,
) -> tuple[bool, str]:
    if running_jobs >= cfg.max_concurrent_jobs:
        return False, "max_concurrent_reached"
    if snapshot.cpu_utilization_pct > cfg.cpu_utilization_threshold_pct:
        return False, "cpu_threshold_exceeded"
    remaining_after_launch = snapshot.available_mem_gb - cfg.per_job_mem_gb
    if remaining_after_launch < cfg.min_free_mem_gb:
        return False, "memory_reserve_violation"
    return True, "launch_allowed"


def scheduler_summary_rows(
    queue_rows: list[dict[str, Any]],
    cfg: SchedulerConfig,
) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(queue_rows, start=1):
        rows.append(
            {
                "queue_index": idx,
                "run_id": row["run_id"],
                "run_dir": row["run_dir"],
                "clock_period_ns": row["clock_period_ns"],
                "scheduler_enabled": cfg.enabled,
                "scheduler_dry_run": cfg.dry_run,
                "max_concurrent_jobs": cfg.max_concurrent_jobs,
                "cpu_threshold_pct": cfg.cpu_utilization_threshold_pct,
                "min_free_mem_gb": cfg.min_free_mem_gb,
                "per_job_mem_gb": cfg.per_job_mem_gb,
            }
        )
    return rows


def wait_for_capacity(
    cfg: SchedulerConfig,
    running_jobs: int,
) -> tuple[ResourceSnapshot, str]:
    while True:
        snap = snapshot_resources()
        allowed, reason = can_launch_more(snap, running_jobs, cfg)
        if allowed:
            return snap, reason
        time.sleep(cfg.poll_interval_sec)
