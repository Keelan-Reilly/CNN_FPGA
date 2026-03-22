#!/usr/bin/env python3
"""Typed data models for the MAC-array evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


ARCHITECTURES = ("baseline", "shared", "replicated")
WORKLOAD_CLASSES = ("dense_steady", "short_burst", "underfilled", "phase_changing")


@dataclass(frozen=True)
class GridSpec:
    label: str
    rows: int
    cols: int

    @property
    def mac_units(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    capacity_scale: float
    dsp_scale: float
    lut_per_mac: float
    lut_fixed: float
    latency_penalty_fraction: float
    fixed_phase_overhead_cycles: int
    wns_bias_ns: float
    wns_per_effective_mac: float
    default_implementation_status: str

    def dsp_for_grid(self, grid: GridSpec) -> int:
        return max(1, int(ceil(grid.mac_units * self.dsp_scale)))

    def lut_for_grid(self, grid: GridSpec) -> int:
        return int(round(self.lut_fixed + self.lut_per_mac * grid.mac_units))

    def capacity_for_grid(self, grid: GridSpec) -> float:
        return max(1.0, grid.mac_units * self.capacity_scale)

    def wns_estimate_for_grid(self, grid: GridSpec) -> float:
        complexity = self.wns_per_effective_mac * grid.mac_units * self.capacity_scale
        return round(self.wns_bias_ns - complexity, 3)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    ops: int
    utilization: float
    parallelism: float
    fixed_overhead_cycles: int


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    workload_class: str
    phases: tuple[PhaseSpec, ...]
    notes: str = ""

    @property
    def total_ops(self) -> int:
        return sum(phase.ops for phase in self.phases)

    @property
    def phase_count(self) -> int:
        return len(self.phases)

    @property
    def dominant_utilization(self) -> float:
        return max(phase.utilization for phase in self.phases)

    @property
    def burstiness(self) -> float:
        if len(self.phases) <= 1:
            return 0.0
        ops = [phase.ops for phase in self.phases]
        return max(ops) / max(1, min(ops))

    @property
    def utilization_variance(self) -> float:
        utils = [phase.utilization for phase in self.phases]
        mean = sum(utils) / len(utils)
        return sum((value - mean) ** 2 for value in utils) / len(utils)


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    dsp_budget: int
    lut_budget: int
    min_throughput_ops_per_cycle: float | None = None
    target_grid: str | None = None


@dataclass(frozen=True)
class PhaseEvaluation:
    architecture: str
    grid: str
    workload: str
    phase: str
    latency_cycles: int
    throughput_ops_per_cycle: float
    utilization_estimate: float


@dataclass(frozen=True)
class EvaluationSummary:
    architecture: str
    grid: str
    workload: str
    total_ops: int
    latency_cycles: int
    effective_throughput_ops_per_cycle: float
    dsp: int
    lut: int
    dsp_efficiency: float
    lut_efficiency: float
    utilization_estimate: float
    timing_feasible: bool
    implementation_status: str
    measurement_basis: str
