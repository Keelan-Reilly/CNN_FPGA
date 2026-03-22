#!/usr/bin/env python3
"""Adaptive switching and break-even analysis for the MAC-array framework."""

from __future__ import annotations

import math
from typing import Any

try:
    from .mac_array_evidence import EvidenceBundle, switching_pair_record
    from .mac_array_metrics import evaluate_phase
    from .mac_array_types import ArchitectureSpec, GridSpec, WorkloadSpec
except ImportError:
    from mac_array_evidence import EvidenceBundle, switching_pair_record
    from mac_array_metrics import evaluate_phase
    from mac_array_types import ArchitectureSpec, GridSpec, WorkloadSpec


def evaluate_adaptive_workload(
    grid: GridSpec,
    workload: WorkloadSpec,
    architectures: dict[str, ArchitectureSpec],
    allowed_modes: tuple[str, ...],
    evidence: EvidenceBundle,
) -> dict[str, Any]:
    if not allowed_modes:
        raise ValueError("allowed_modes must not be empty")

    phase_candidates: list[dict[str, Any]] = []
    dp: list[dict[str, tuple[int, str | None]]] = []

    for phase_idx, phase in enumerate(workload.phases):
        candidate_map: dict[str, Any] = {}
        for mode in allowed_modes:
            phase_eval = evaluate_phase(grid, architectures[mode], workload, phase)
            candidate_map[mode] = phase_eval
        phase_candidates.append(candidate_map)

        phase_dp: dict[str, tuple[int, str | None]] = {}
        for mode in allowed_modes:
            own_latency = candidate_map[mode].latency_cycles
            if phase_idx == 0:
                phase_dp[mode] = (own_latency, None)
                continue

            best_total = None
            best_prev = None
            for prev_mode in allowed_modes:
                prev_total = dp[phase_idx - 1][prev_mode][0]
                switch = 0
                if prev_mode != mode:
                    switch = switching_pair_record(evidence, prev_mode, mode)["switch_cycles"]
                total = prev_total + switch + own_latency
                if best_total is None or total < best_total:
                    best_total = total
                    best_prev = prev_mode
            assert best_total is not None
            phase_dp[mode] = (best_total, best_prev)
        dp.append(phase_dp)

    final_mode = min(allowed_modes, key=lambda mode: dp[-1][mode][0])
    total_latency = dp[-1][final_mode][0]
    total_ops = workload.total_ops

    reversed_modes = [final_mode]
    current_mode = final_mode
    for phase_idx in range(len(workload.phases) - 1, 0, -1):
        prev_mode = dp[phase_idx][current_mode][1]
        assert prev_mode is not None
        reversed_modes.append(prev_mode)
        current_mode = prev_mode
    mode_sequence = list(reversed(reversed_modes))

    transitions = 0
    total_switch_cycles = 0
    phase_rows: list[dict[str, Any]] = []
    prev_mode: str | None = None
    for phase_idx, phase in enumerate(workload.phases):
        selected_mode = mode_sequence[phase_idx]
        selected_eval = phase_candidates[phase_idx][selected_mode]
        transition_cost = 0
        transition_record = None
        if prev_mode is not None and prev_mode != selected_mode:
            transition_record = switching_pair_record(evidence, prev_mode, selected_mode)
            transition_cost = transition_record["switch_cycles"]
            total_switch_cycles += transition_cost
            transitions += 1

        alternatives = sorted(
            (
                {
                    "mode": mode,
                    "latency_cycles": phase_candidates[phase_idx][mode].latency_cycles,
                }
                for mode in allowed_modes
            ),
            key=lambda item: item["latency_cycles"],
        )
        phase_rows.append(
            {
                "phase": phase.name,
                "selected_mode": selected_mode,
                "selected_mode_rank": 1 + next(
                    idx for idx, item in enumerate(alternatives) if item["mode"] == selected_mode
                ),
                "phase_latency_cycles": selected_eval.latency_cycles,
                "phase_ops": phase.ops,
                "phase_throughput_ops_per_cycle": selected_eval.throughput_ops_per_cycle,
                "transition_from_mode": prev_mode or "",
                "transition_switch_cycles": transition_cost,
                "transition_record_id": transition_record["record_id"] if transition_record else "",
                "transition_source_desc": transition_record["source_desc"] if transition_record else "",
                "phase_candidate_latencies": alternatives,
            }
        )
        prev_mode = selected_mode

    throughput = total_ops / total_latency if total_latency > 0 else 0.0
    return {
        "grid": grid.label,
        "workload": workload.name,
        "allowed_modes": list(allowed_modes),
        "mode_sequence": mode_sequence,
        "transitions": transitions,
        "total_switch_cycles": total_switch_cycles,
        "latency_cycles": total_latency,
        "switching_adjusted_throughput_ops_per_cycle": throughput,
        "adaptive_provenance_kind": "analytical_dynamic_programming",
        "adaptive_derivation": "dynamic programming over phase latency plus mode-pair switching costs",
        "phase_rows": phase_rows,
    }


def break_even_phase_cycles(
    current_cycles_per_op: float,
    candidate_cycles_per_op: float,
    switch_cycles: int,
) -> int | None:
    improvement = current_cycles_per_op - candidate_cycles_per_op
    if improvement <= 0:
        return None
    min_ops = int(math.ceil(switch_cycles / improvement))
    return int(math.ceil(min_ops * candidate_cycles_per_op))


def build_break_even_rows(
    grid: GridSpec,
    workload: WorkloadSpec,
    architectures: dict[str, ArchitectureSpec],
    switch_pairs: list[tuple[str, str]],
    evidence: EvidenceBundle,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in workload.phases:
        for current_mode, candidate_mode in switch_pairs:
            current_eval = evaluate_phase(grid, architectures[current_mode], workload, phase)
            candidate_eval = evaluate_phase(grid, architectures[candidate_mode], workload, phase)
            switch_record = switching_pair_record(evidence, current_mode, candidate_mode)
            switch_cycles = switch_record["switch_cycles"]
            current_cpo = current_eval.latency_cycles / phase.ops
            candidate_cpo = candidate_eval.latency_cycles / phase.ops
            min_cycles = break_even_phase_cycles(current_cpo, candidate_cpo, switch_cycles)
            rows.append(
                {
                    "grid": grid.label,
                    "workload": workload.name,
                    "phase": phase.name,
                    "from_mode": current_mode,
                    "to_mode": candidate_mode,
                    "switch_cycles": switch_cycles,
                    "switch_record_id": switch_record["record_id"],
                    "switch_value_kind": switch_record["value_kind"],
                    "switch_source_path": switch_record["source_path"],
                    "switch_source_desc": switch_record["source_desc"],
                    "switch_derivation": switch_record["derivation"],
                    "stay_latency_cycles": current_eval.latency_cycles,
                    "switch_latency_cycles": candidate_eval.latency_cycles + switch_cycles,
                    "beneficial_now": candidate_eval.latency_cycles + switch_cycles < current_eval.latency_cycles,
                    "min_phase_cycles_for_break_even": min_cycles,
                    "break_even_provenance_kind": "analytical_break_even",
                    "break_even_derivation": "switch_cycles / (current_cycles_per_op - candidate_cycles_per_op)",
                }
            )
    return rows
