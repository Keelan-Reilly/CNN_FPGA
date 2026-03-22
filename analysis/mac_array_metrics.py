#!/usr/bin/env python3
"""Static and workload-aware metrics for the MAC-array framework."""

from __future__ import annotations

import math
from typing import Any

try:
    from .mac_array_evidence import EvidenceBundle, static_field_provenance, static_override_value
    from .mac_array_types import (
        ArchitectureSpec,
        EvaluationSummary,
        GridSpec,
        PhaseEvaluation,
        PhaseSpec,
        WorkloadSpec,
    )
except ImportError:
    from mac_array_evidence import EvidenceBundle, static_field_provenance, static_override_value
    from mac_array_types import (
        ArchitectureSpec,
        EvaluationSummary,
        GridSpec,
        PhaseEvaluation,
        PhaseSpec,
        WorkloadSpec,
    )


def _flatten_provenance(prefix: str, value: Any, kind: str, source_id: str, source_path: str, source_desc: str, derivation: str) -> dict[str, Any]:
    return {
        prefix: value,
        f"{prefix}_provenance_kind": kind,
        f"{prefix}_source_id": source_id,
        f"{prefix}_source_path": source_path,
        f"{prefix}_source_desc": source_desc,
        f"{prefix}_derivation": derivation,
    }


def derive_static_row(
    grid: GridSpec,
    arch: ArchitectureSpec,
    evidence: EvidenceBundle,
) -> dict[str, Any]:
    default_dsp = arch.dsp_for_grid(grid)
    default_lut = arch.lut_for_grid(grid)
    default_wns = arch.wns_estimate_for_grid(grid)
    default_impl_status = arch.default_implementation_status
    note = static_override_value(evidence, arch.name, grid.label, "note") or ""

    dsp = static_override_value(evidence, arch.name, grid.label, "dsp")
    lut = static_override_value(evidence, arch.name, grid.label, "lut")
    wns = static_override_value(evidence, arch.name, grid.label, "wns_estimate_ns")
    impl_status = static_override_value(evidence, arch.name, grid.label, "implementation_status")

    dsp_prov = static_field_provenance(evidence, arch.name, grid.label, "dsp")
    lut_prov = static_field_provenance(evidence, arch.name, grid.label, "lut")
    wns_prov = static_field_provenance(evidence, arch.name, grid.label, "wns_estimate_ns")
    latency_penalty_prov = static_field_provenance(evidence, arch.name, grid.label, "latency_penalty_fraction")
    fixed_overhead_prov = static_field_provenance(evidence, arch.name, grid.label, "fixed_phase_overhead_cycles")
    impl_prov = static_field_provenance(evidence, arch.name, grid.label, "implementation_status")

    kinds = {
        dsp_prov.value_kind,
        lut_prov.value_kind,
        wns_prov.value_kind,
        latency_penalty_prov.value_kind,
        fixed_overhead_prov.value_kind,
        impl_prov.value_kind,
    }
    measurement_basis = "modelled"
    if any(kind.startswith("anchored") for kind in kinds):
        measurement_basis = "mixed_anchored_and_modelled"

    row = {
        "grid": grid.label,
        "architecture": arch.name,
        "rows": grid.rows,
        "cols": grid.cols,
        "mac_units": grid.mac_units,
        "measurement_basis": measurement_basis,
        "record_provenance_summary": "; ".join(sorted(kinds)),
        "note": note,
    }
    row.update(
        _flatten_provenance(
            "dsp",
            default_dsp if dsp is None else dsp,
            dsp_prov.value_kind,
            dsp_prov.source_id,
            dsp_prov.source_path,
            dsp_prov.source_desc,
            dsp_prov.derivation,
        )
    )
    row.update(
        _flatten_provenance(
            "lut",
            default_lut if lut is None else lut,
            lut_prov.value_kind,
            lut_prov.source_id,
            lut_prov.source_path,
            lut_prov.source_desc,
            lut_prov.derivation,
        )
    )
    row.update(
        _flatten_provenance(
            "wns_estimate_ns",
            default_wns if wns is None else wns,
            wns_prov.value_kind,
            wns_prov.source_id,
            wns_prov.source_path,
            wns_prov.source_desc,
            wns_prov.derivation,
        )
    )
    row.update(
        _flatten_provenance(
            "latency_penalty_fraction",
            arch.latency_penalty_fraction,
            latency_penalty_prov.value_kind,
            latency_penalty_prov.source_id,
            latency_penalty_prov.source_path,
            latency_penalty_prov.source_desc,
            latency_penalty_prov.derivation,
        )
    )
    row.update(
        _flatten_provenance(
            "fixed_phase_overhead_cycles",
            arch.fixed_phase_overhead_cycles,
            fixed_overhead_prov.value_kind,
            fixed_overhead_prov.source_id,
            fixed_overhead_prov.source_path,
            fixed_overhead_prov.source_desc,
            fixed_overhead_prov.derivation,
        )
    )
    row.update(
        _flatten_provenance(
            "implementation_status",
            default_impl_status if impl_status is None else impl_status,
            impl_prov.value_kind,
            impl_prov.source_id,
            impl_prov.source_path,
            impl_prov.source_desc,
            impl_prov.derivation,
        )
    )
    return row


def evaluate_phase(
    grid: GridSpec,
    arch: ArchitectureSpec,
    workload: WorkloadSpec,
    phase: PhaseSpec,
) -> PhaseEvaluation:
    demand_parallelism = grid.mac_units * max(0.05, phase.utilization) * max(0.25, phase.parallelism)
    served_parallelism = min(arch.capacity_for_grid(grid), max(1.0, demand_parallelism))
    ideal_cycles = int(math.ceil(phase.ops / served_parallelism))
    latency_cycles = ideal_cycles + phase.fixed_overhead_cycles + arch.fixed_phase_overhead_cycles
    latency_cycles += int(math.ceil(ideal_cycles * arch.latency_penalty_fraction))
    throughput = phase.ops / latency_cycles if latency_cycles > 0 else 0.0
    utilization = min(1.0, demand_parallelism / arch.capacity_for_grid(grid))
    return PhaseEvaluation(
        architecture=arch.name,
        grid=grid.label,
        workload=workload.name,
        phase=phase.name,
        latency_cycles=latency_cycles,
        throughput_ops_per_cycle=throughput,
        utilization_estimate=utilization,
    )


def summarize_workload(
    grid: GridSpec,
    arch: ArchitectureSpec,
    workload: WorkloadSpec,
    evidence: EvidenceBundle,
) -> tuple[EvaluationSummary, list[PhaseEvaluation]]:
    static_row = derive_static_row(grid, arch, evidence)
    phase_evals = [evaluate_phase(grid, arch, workload, phase) for phase in workload.phases]
    total_ops = workload.total_ops
    total_latency = sum(item.latency_cycles for item in phase_evals)
    throughput = total_ops / total_latency if total_latency > 0 else 0.0
    weighted_util = 0.0
    if total_latency > 0:
        weighted_util = sum(
            item.utilization_estimate * item.latency_cycles for item in phase_evals
        ) / total_latency

    dsp = int(static_row["dsp"])
    lut = int(static_row["lut"])
    wns = float(static_row["wns_estimate_ns"])
    implementation_status = str(static_row["implementation_status"])
    timing_feasible = wns >= 0.0 and "failed" not in implementation_status

    summary = EvaluationSummary(
        architecture=arch.name,
        grid=grid.label,
        workload=workload.name,
        total_ops=total_ops,
        latency_cycles=total_latency,
        effective_throughput_ops_per_cycle=throughput,
        dsp=dsp,
        lut=lut,
        dsp_efficiency=throughput / dsp if dsp else 0.0,
        lut_efficiency=throughput / lut if lut else 0.0,
        utilization_estimate=weighted_util,
        timing_feasible=timing_feasible,
        implementation_status=implementation_status,
        measurement_basis=str(static_row["measurement_basis"]),
    )
    return summary, phase_evals
