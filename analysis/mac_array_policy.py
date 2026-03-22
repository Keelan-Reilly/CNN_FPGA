#!/usr/bin/env python3
"""Constraint-aware architecture selection policy with explainability outputs."""

from __future__ import annotations

from typing import Any

try:
    from .mac_array_types import ConstraintSpec
except ImportError:
    from mac_array_types import ConstraintSpec


def _fixed_candidate_diagnostic(row: dict[str, Any], constraint: ConstraintSpec) -> dict[str, Any]:
    reasons: list[str] = []
    within_dsp = row["dsp"] <= constraint.dsp_budget
    within_lut = row["lut"] <= constraint.lut_budget
    timing_ok = row["timing_feasible"]
    target = constraint.min_throughput_ops_per_cycle
    meets_target = target is None or row["effective_throughput_ops_per_cycle"] >= target

    if not within_dsp:
        reasons.append("over_dsp_budget")
    if not within_lut:
        reasons.append("over_lut_budget")
    if not timing_ok:
        reasons.append("timing_infeasible")
    if not meets_target and target is not None:
        reasons.append("below_throughput_target")

    return {
        "candidate_type": "fixed",
        "candidate": row["architecture"],
        "grid": row["grid"],
        "workload": row["workload"],
        "constraint": constraint.name,
        "feasible": within_dsp and within_lut and timing_ok,
        "within_dsp_budget": within_dsp,
        "within_lut_budget": within_lut,
        "timing_feasible": timing_ok,
        "meets_throughput_target": meets_target,
        "rejected_reasons": reasons,
        "effective_throughput_ops_per_cycle": row["effective_throughput_ops_per_cycle"],
        "latency_cycles": row["latency_cycles"],
        "dsp": row["dsp"],
        "lut": row["lut"],
        "score_primary": row["effective_throughput_ops_per_cycle"],
        "score_secondary": -row["latency_cycles"],
    }


def _adaptive_candidate_diagnostic(
    adaptive_row: dict[str, Any],
    best_fixed: dict[str, Any] | None,
    constraint: ConstraintSpec,
    workload_class: str,
) -> dict[str, Any]:
    target = constraint.min_throughput_ops_per_cycle
    meets_target = target is None or adaptive_row["switching_adjusted_throughput_ops_per_cycle"] >= target
    gain_vs_fixed = None
    reasons: list[str] = []
    if best_fixed is not None:
        gain_vs_fixed = (
            adaptive_row["switching_adjusted_throughput_ops_per_cycle"]
            - best_fixed["effective_throughput_ops_per_cycle"]
        )
        if gain_vs_fixed <= 0.02:
            reasons.append("adaptive_gain_too_small")
    if workload_class != "phase_changing":
        reasons.append("workload_not_phase_changing")
    if not meets_target and target is not None:
        reasons.append("below_throughput_target")
    if adaptive_row["transitions"] == 0:
        reasons.append("no_switching_observed")

    return {
        "candidate_type": "adaptive",
        "candidate": "adaptive_mode_switching",
        "grid": adaptive_row["grid"],
        "workload": adaptive_row["workload"],
        "constraint": constraint.name,
        "feasible": workload_class == "phase_changing" and not reasons,
        "within_dsp_budget": True,
        "within_lut_budget": True,
        "timing_feasible": True,
        "meets_throughput_target": meets_target,
        "rejected_reasons": reasons,
        "effective_throughput_ops_per_cycle": adaptive_row["switching_adjusted_throughput_ops_per_cycle"],
        "latency_cycles": adaptive_row["latency_cycles"],
        "dsp": None,
        "lut": None,
        "score_primary": adaptive_row["switching_adjusted_throughput_ops_per_cycle"],
        "score_secondary": -adaptive_row["latency_cycles"],
        "gain_vs_best_fixed": gain_vs_fixed,
        "mode_sequence": adaptive_row["mode_sequence"],
        "transitions": adaptive_row["transitions"],
    }


def evaluate_policy(
    fixed_rows: list[dict[str, Any]],
    adaptive_row: dict[str, Any] | None,
    constraint: ConstraintSpec,
    workload_class: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics = [_fixed_candidate_diagnostic(row, constraint) for row in fixed_rows]
    feasible_fixed = [row for row in diagnostics if row["feasible"]]
    fixed_lookup = {row["architecture"]: row for row in fixed_rows}

    if feasible_fixed:
        target = constraint.min_throughput_ops_per_cycle
        target_ok = [row for row in feasible_fixed if row["meets_throughput_target"]]
        pool = target_ok or feasible_fixed
        best_fixed_diag = max(pool, key=lambda row: (row["score_primary"], row["score_secondary"]))
        best_fixed_row = fixed_lookup[best_fixed_diag["candidate"]]
    else:
        best_fixed_diag = None
        best_fixed_row = None

    adaptive_diag = None
    if adaptive_row is not None:
        adaptive_diag = _adaptive_candidate_diagnostic(adaptive_row, best_fixed_row, constraint, workload_class)
        diagnostics.append(adaptive_diag)

    if best_fixed_diag is None:
        recommendation = {
            "constraint": constraint.name,
            "workload": fixed_rows[0]["workload"] if fixed_rows else adaptive_row["workload"],
            "grid": fixed_rows[0]["grid"] if fixed_rows else adaptive_row["grid"],
            "recommendation": "no_feasible_static_mode",
            "reason": "All fixed modes exceed resource or timing constraints.",
            "runner_up": "",
            "runner_up_reason": "",
            "selected_throughput_ops_per_cycle": None,
        }
        return recommendation, diagnostics

    winner = best_fixed_diag
    if adaptive_diag is not None and adaptive_diag["feasible"] and adaptive_diag["score_primary"] > best_fixed_diag["score_primary"]:
        winner = adaptive_diag

    ranked = sorted(
        diagnostics,
        key=lambda row: (
            row["feasible"],
            row["meets_throughput_target"],
            row["score_primary"],
            row["score_secondary"],
        ),
        reverse=True,
    )
    runner_up = next((row for row in ranked if row["candidate"] != winner["candidate"]), None)

    if winner["candidate"] == "adaptive_mode_switching":
        reason = "Phase-varying demand justifies switching overhead and improves switching-adjusted throughput."
    elif winner["candidate"] == "shared":
        reason = "Shared satisfies the constraint while using materially less DSP/LUT than baseline."
    elif winner["candidate"] == "replicated":
        reason = "Replicated is the only feasible high-throughput option under this demand."
    else:
        reason = "Baseline is the best fixed mode once throughput target, latency, and timing are applied."

    recommendation = {
        "constraint": constraint.name,
        "workload": fixed_rows[0]["workload"] if fixed_rows else adaptive_row["workload"],
        "grid": fixed_rows[0]["grid"] if fixed_rows else adaptive_row["grid"],
        "recommendation": winner["candidate"],
        "reason": reason,
        "runner_up": runner_up["candidate"] if runner_up else "",
        "runner_up_reason": ",".join(runner_up["rejected_reasons"]) if runner_up else "",
        "selected_throughput_ops_per_cycle": winner["score_primary"],
        "selected_latency_cycles": -winner["score_secondary"],
        "selection_basis": "ranked feasibility, throughput target, throughput, then latency",
    }
    return recommendation, diagnostics


def choose_architecture(
    fixed_rows: list[dict[str, Any]],
    adaptive_row: dict[str, Any] | None,
    constraint: ConstraintSpec,
    workload_class: str,
) -> dict[str, Any]:
    recommendation, _ = evaluate_policy(fixed_rows, adaptive_row, constraint, workload_class)
    return recommendation
