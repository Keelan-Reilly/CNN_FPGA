#!/usr/bin/env python3
"""Workload definitions and validation helpers for the MAC-array framework."""

from __future__ import annotations

from typing import Any

try:
    from .mac_array_types import PhaseSpec, WorkloadSpec, WORKLOAD_CLASSES
except ImportError:
    from mac_array_types import PhaseSpec, WorkloadSpec, WORKLOAD_CLASSES


def _require_float(raw: Any, field: str) -> float:
    value = float(raw)
    if value < 0:
        raise ValueError(f"{field} must be >= 0")
    return value


def phase_from_dict(payload: dict[str, Any]) -> PhaseSpec:
    return PhaseSpec(
        name=str(payload["name"]),
        ops=int(payload["ops"]),
        utilization=_require_float(payload["utilization"], "utilization"),
        parallelism=_require_float(payload["parallelism"], "parallelism"),
        fixed_overhead_cycles=int(payload.get("fixed_overhead_cycles", 0)),
    )


def workload_from_dict(payload: dict[str, Any]) -> WorkloadSpec:
    workload_class = str(payload["workload_class"])
    if workload_class not in WORKLOAD_CLASSES:
        raise ValueError(f"Unsupported workload_class '{workload_class}'")

    phases = tuple(phase_from_dict(item) for item in payload["phases"])
    if not phases:
        raise ValueError("Each workload must define at least one phase")

    return WorkloadSpec(
        name=str(payload["name"]),
        workload_class=workload_class,
        phases=phases,
        notes=str(payload.get("notes", "")),
    )


def workloads_from_config(items: list[dict[str, Any]]) -> list[WorkloadSpec]:
    workloads = [workload_from_dict(item) for item in items]
    names = [w.name for w in workloads]
    if len(names) != len(set(names)):
        raise ValueError("Workload names must be unique")
    return workloads
