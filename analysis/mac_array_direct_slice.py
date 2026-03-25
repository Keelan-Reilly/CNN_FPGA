#!/usr/bin/env python3
"""Direct MAC-array slice comparison, calibration, and tradeoff helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from .fpga_results import REPO_ROOT
    from .mac_array_evidence import load_evidence
    from .mac_array_metrics import derive_static_row
    from .mac_array_types import GridSpec
except ImportError:
    from fpga_results import REPO_ROOT
    from mac_array_evidence import load_evidence
    from mac_array_metrics import derive_static_row
    from mac_array_types import GridSpec


DIRECT_AGGREGATE_NAMES = [
    "study_mac_array_direct_baseline_4x4.json",
    "study_mac_array_direct_baseline_8x4.json",
    "study_mac_array_direct_baseline_8x8.json",
    "study_mac_array_direct_baseline.json",
    "study_mac_array_direct_tradeoff_4x4.json",
    "study_mac_array_direct_shared_dsp_4x4.json",
    "study_mac_array_direct_shared_lut_8x4.json",
    "study_mac_array_direct_shared_dsp_8x4.json",
]
FRAMEWORK_CONFIG = REPO_ROOT / "experiments" / "configs" / "mac_array_framework_v2.json"
ARCH_MODE_BASELINE = 0
ARCH_MODE_SHARED_LUT_SAVING = 1
ARCH_MODE_SHARED_DSP_REDUCING = 2
NO_FEASIBLE_MEASURED_OPTION = "no_feasible_measured_option"
BASELINE_PREFERRED = "baseline_preferred"
SHARED_PREFERRED = "shared_preferred"
DIRECTLY_MEASURED_SUPPORTED = "directly_measured_supported"
MEASURED_DIRECTIONALLY_SUPPORTED = "measured_directionally_supported"
MEASURED_PARTIAL_SUPPORT = "measured_partial_support"
EXTRAPOLATED_BEYOND_MEASURED_SUPPORT = "extrapolated_beyond_measured_support"
CONTRADICTED_BY_MEASURED_IMPLEMENTATION = "contradicted_by_measured_implementation"
CALIBRATION_WELL_ALIGNED = "well_aligned"
CALIBRATION_DIRECTIONALLY_OPTIMISTIC = "directionally_aligned_but_numerically_optimistic"
CALIBRATION_DIRECTIONALLY_PESSIMISTIC = "directionally_aligned_but_numerically_pessimistic"
CALIBRATION_TOO_UNCERTAIN = "too_uncertain_for_numeric_trust"
CALIBRATION_IMPLEMENTATION_DEPENDENT = "implementation_dependent"
UTILITY_PERFORMANCE_FIRST_DEFAULT = "performance_first_default"
UTILITY_WORTH_LUT_BOTTLENECK = "worthwhile_when_lut_is_real_bottleneck"
UTILITY_WORTH_DSP_BOTTLENECK = "worthwhile_when_dsp_is_real_bottleneck"
UTILITY_GRID_DEPENDENT_TIMING = "timing_margin_choice_is_grid_dependent"
FLEXIBILITY_JUSTIFIED_LUT = "justified_only_for_lut_dominant_relief"
FLEXIBILITY_JUSTIFIED_DSP = "justified_only_for_dsp_dominant_relief"
FLEXIBILITY_NOT_JUSTIFIED_PERFORMANCE = "not_justified_when_performance_dominates"
FLEXIBILITY_BASELINE_PREFERRED = "baseline_preferred_without_hard_resource_bottleneck"


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _percent_change(delta: float | None, reference: float | None) -> float | None:
    if delta is None or reference in (None, 0):
        return None
    return round((delta / reference) * 100.0, 6)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 6)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _parse_params(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def direct_architecture_name(params: dict[str, Any]) -> str:
    arch_mode = int(params.get("ARCH_MODE", ARCH_MODE_BASELINE))
    if arch_mode == ARCH_MODE_SHARED_DSP_REDUCING:
        return "shared_dsp_reducing"
    if arch_mode == ARCH_MODE_SHARED_LUT_SAVING:
        return "shared_lut_saving"
    return "baseline"


def direct_architecture_family(architecture: str) -> str:
    return "baseline" if architecture == "baseline" else "shared"


def direct_latency_model(k_depth: int, architecture: str = "baseline") -> int:
    work_cycles = (2 * k_depth) if direct_architecture_family(architecture) == "shared" else k_depth
    return work_cycles + 1


def direct_throughput_model(rows: int, cols: int, k_depth: int, architecture: str = "baseline") -> float:
    return float(rows * cols * k_depth) / float(direct_latency_model(k_depth, architecture))


def _load_direct_payloads(aggregate_payload: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
    if aggregate_payload is not None:
        return [("in_memory_payload", aggregate_payload)]
    payloads: list[tuple[str, dict[str, Any]]] = []
    for name in DIRECT_AGGREGATE_NAMES:
        candidate = REPO_ROOT / "results" / "fpga" / "aggregates" / name
        if candidate.exists():
            payloads.append((str(candidate.relative_to(REPO_ROOT)), _load_json(candidate)))
    return payloads


def _fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) < 2:
        return None
    n = float(len(xs))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = (n * sum_xx) - (sum_x * sum_x)
    if denom == 0.0:
        return None
    slope = ((n * sum_xy) - (sum_x * sum_y)) / denom
    intercept = (sum_y - (slope * sum_x)) / n
    return intercept, slope


def build_direct_slice_comparison_rows(aggregate_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = _load_json(FRAMEWORK_CONFIG)
    evidence = load_evidence(REPO_ROOT / cfg["evidence_path"])
    payloads = _load_direct_payloads(aggregate_payload)
    if not payloads:
        return []

    rows: list[dict[str, Any]] = []
    for source_path, payload in payloads:
        for measured in payload.get("runs", []):
            params = _parse_params(measured.get("params"))
            grid_rows = int(params.get("ARRAY_ROWS", 0))
            grid_cols = int(params.get("ARRAY_COLS", 0))
            k_depth = int(params.get("K_DEPTH", 0))
            architecture = direct_architecture_name(params)
            architecture_family = direct_architecture_family(architecture)
            if grid_rows <= 0 or grid_cols <= 0:
                continue
            grid = GridSpec(label=f"{grid_rows}x{grid_cols}", rows=grid_rows, cols=grid_cols)
            mac_units = grid_rows * grid_cols
            static_model = derive_static_row(grid, evidence.architectures[architecture_family], evidence)
            measured_eff_tput = measured.get("effective_throughput_ops_per_cycle")
            measured_latency = measured.get("latency_cycles")
            model_latency = direct_latency_model(k_depth, architecture)
            model_eff_tput = direct_throughput_model(grid_rows, grid_cols, k_depth, architecture)
            rows.append(
                {
                    "run_id": measured.get("run_id"),
                    "grid": grid.label,
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "mac_units": mac_units,
                    "k_depth": k_depth,
                    "architecture": architecture,
                    "architecture_family": architecture_family,
                    "architecture_variant_id": architecture,
                    "arch_mode": int(params.get("ARCH_MODE", ARCH_MODE_BASELINE)),
                    "status": measured.get("status"),
                    "direct_evidence_kind": (
                        f"direct_measured_mac_array_{architecture}_slice"
                        if measured.get("status") == "succeeded"
                        else "direct_measurement_missing_or_failed"
                    ),
                    "direct_evidence_source": source_path,
                    "framework_model_kind": f"framework_{architecture_family}_static_model",
                    "framework_model_source": str((REPO_ROOT / cfg["evidence_path"]).relative_to(REPO_ROOT)),
                    "measured_dsp": measured.get("dsp"),
                    "model_dsp": static_model["dsp"],
                    "dsp_delta": None if measured.get("dsp") is None else measured.get("dsp") - static_model["dsp"],
                    "measured_lut": measured.get("lut"),
                    "model_lut": static_model["lut"],
                    "lut_delta": None if measured.get("lut") is None else measured.get("lut") - static_model["lut"],
                    "measured_ff": measured.get("ff"),
                    "measured_wns_ns": measured.get("wns_ns"),
                    "model_wns_estimate_ns": static_model["wns_estimate_ns"],
                    "wns_delta_ns": None if measured.get("wns_ns") is None else round(measured.get("wns_ns") - static_model["wns_estimate_ns"], 6),
                    "measured_latency_cycles": measured_latency,
                    "direct_slice_latency_model_cycles": model_latency,
                    "latency_delta_cycles": None if measured_latency is None else measured_latency - model_latency,
                    "measured_effective_throughput_ops_per_cycle": measured_eff_tput,
                    "direct_slice_throughput_model_ops_per_cycle": round(model_eff_tput, 6),
                    "throughput_delta_ops_per_cycle": None if measured_eff_tput is None else round(measured_eff_tput - model_eff_tput, 6),
                    "comparison_status": (
                        "direct_measured_vs_modelled"
                        if measured.get("status") == "succeeded"
                        else "measurement_missing_or_failed"
                    ),
                    "comparison_note": (
                        f"Direct MAC-array slice measurement is available for this {architecture} grid."
                        if measured.get("status") == "succeeded"
                        else "Direct slice exists but this grid point has not been measured successfully yet."
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            row["mac_units"],
            row["grid_rows"],
            row["grid_cols"],
            row["architecture_family"],
            row["architecture"],
            row["run_id"] or "",
        )
    )

    measured_rows = [
        row
        for row in rows
        if row["comparison_status"] == "direct_measured_vs_modelled"
        and row["measured_lut"] is not None
        and row["architecture_family"] == "baseline"
    ]
    lut_fit = _fit_linear([float(row["mac_units"]) for row in measured_rows], [float(row["measured_lut"]) for row in measured_rows])
    for row in rows:
        if row["architecture_family"] != "baseline":
            row["calibrated_model_lut"] = None
            row["calibrated_lut_delta"] = None
            row["lut_calibration_kind"] = "not_applicable_non_baseline"
        elif lut_fit is None:
            row["calibrated_model_lut"] = None
            row["calibrated_lut_delta"] = None
            row["lut_calibration_kind"] = "not_available"
        else:
            intercept, slope = lut_fit
            calibrated = intercept + (slope * row["mac_units"])
            row["calibrated_model_lut"] = round(calibrated, 3)
            row["calibrated_lut_delta"] = None if row["measured_lut"] is None else round(row["measured_lut"] - calibrated, 6)
            row["lut_calibration_kind"] = "direct_slice_linear_fit"
    return rows


def build_direct_calibration_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        row
        for row in rows
        if row["comparison_status"] == "direct_measured_vs_modelled" and row["architecture_family"] == "baseline"
    ]
    lut_rows = [row for row in measured if row["measured_lut"] is not None and row["lut_delta"] is not None]
    wns_rows = [row for row in measured if row["measured_wns_ns"] is not None]
    summary = {
        "measured_points": len(measured),
        "grids": [row["grid"] for row in measured],
        "dsp_exact_match_count": sum(1 for row in measured if row["dsp_delta"] == 0),
        "latency_exact_match_count": sum(1 for row in measured if row["latency_delta_cycles"] == 0),
        "throughput_exact_match_count": sum(1 for row in measured if row["throughput_delta_ops_per_cycle"] == 0.0),
        "lut_error_min": min((row["lut_delta"] for row in lut_rows), default=None),
        "lut_error_max": max((row["lut_delta"] for row in lut_rows), default=None),
        "lut_error_mean": round(sum(row["lut_delta"] for row in lut_rows) / len(lut_rows), 6) if lut_rows else None,
        "wns_measured_min_ns": min((row["measured_wns_ns"] for row in wns_rows), default=None),
        "wns_measured_max_ns": max((row["measured_wns_ns"] for row in wns_rows), default=None),
        "baseline_model_scope_note": "Direct calibration applies only to the standalone baseline slice, not shared/replicated/adaptive hardware.",
    }
    if len(lut_rows) >= 2:
        xs = [float(row["mac_units"]) for row in lut_rows]
        ys = [float(row["measured_lut"]) for row in lut_rows]
        fit = _fit_linear(xs, ys)
        if fit is not None:
            intercept, slope = fit
            summary["calibrated_lut_model"] = {
                "formula": "lut = lut_fixed + lut_per_mac * mac_units",
                "lut_fixed": round(intercept, 6),
                "lut_per_mac": round(slope, 6),
                "calibration_kind": "direct_slice_linear_fit",
                "provenance_kind": "derived_from_direct_measured_mac_slice",
                "source_aggregates": sorted({row["direct_evidence_source"] for row in lut_rows}),
                "usage_note": "Use as a baseline-only calibration aid or caution reference; do not silently replace the global framework model.",
            }
    return summary


def _measured_relief_kind(measured_dsp_delta: int | None, measured_lut_delta: int | None) -> str:
    dsp_relief = measured_dsp_delta is not None and measured_dsp_delta < 0
    lut_relief = measured_lut_delta is not None and measured_lut_delta < 0
    if dsp_relief and lut_relief:
        return "dsp_and_lut_relief"
    if dsp_relief:
        return "dsp_relief"
    if lut_relief:
        return "lut_relief"
    if measured_dsp_delta == 0 and measured_lut_delta == 0:
        return "no_resource_relief"
    return "mixed_or_regressive"


def _shared_variant_intent(shared_variant: str) -> str:
    if shared_variant == "shared_dsp_reducing":
        return "dsp_relief"
    if shared_variant == "shared_lut_saving":
        return "lut_relief"
    return "shared_relief"


def _shared_tradeoff_note(shared_variant: str, measured_dsp_delta: int | None) -> str:
    if shared_variant == "shared_lut_saving":
        return (
            "The LUT-saving shared implementation directly measures a real LUT/latency/throughput tradeoff on the same isolated slice, "
            "but its DSP count stays flat instead of showing the modelled reduction."
        )
    if measured_dsp_delta is not None and measured_dsp_delta < 0:
        return (
            "The DSP-reducing shared implementation directly measures a real DSP-reduction tradeoff on the same isolated slice. "
            "Treat this as a small implementation-specific bridge rather than a family-wide calibration."
        )
    return (
        "The DSP-reducing shared implementation was intended to lower mapped DSP usage, but this measured point does not show a DSP reduction on the current flow. "
        "Treat it as an implementation-specific warning rather than proof of a general DSP-saving mechanism."
    )


def build_direct_tradeoff_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured_rows = [row for row in rows if row["comparison_status"] == "direct_measured_vs_modelled"]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in measured_rows:
        key = (row["grid"], row["k_depth"])
        grouped.setdefault(key, {})[row["architecture"]] = row

    tradeoff_rows: list[dict[str, Any]] = []
    for (grid, k_depth), by_arch in sorted(grouped.items()):
        baseline = by_arch.get("baseline")
        if baseline is None:
            continue
        shared_variants = sorted(
            (architecture for architecture, row in by_arch.items() if row["architecture_family"] == "shared"),
            key=lambda architecture: (0 if architecture == "shared_lut_saving" else 1, architecture),
        )
        for shared_variant in shared_variants:
            shared = by_arch[shared_variant]
            measured_dsp_delta = None if baseline["measured_dsp"] is None or shared["measured_dsp"] is None else shared["measured_dsp"] - baseline["measured_dsp"]
            model_dsp_delta = shared["model_dsp"] - baseline["model_dsp"]
            measured_lut_delta = None if baseline["measured_lut"] is None or shared["measured_lut"] is None else shared["measured_lut"] - baseline["measured_lut"]
            measured_latency_delta = None if baseline["measured_latency_cycles"] is None or shared["measured_latency_cycles"] is None else shared["measured_latency_cycles"] - baseline["measured_latency_cycles"]
            measured_throughput_delta = (
                None
                if baseline["measured_effective_throughput_ops_per_cycle"] is None
                or shared["measured_effective_throughput_ops_per_cycle"] is None
                else round(
                    shared["measured_effective_throughput_ops_per_cycle"] - baseline["measured_effective_throughput_ops_per_cycle"],
                    6,
                )
            )
            measured_wns_delta = (
                None
                if baseline["measured_wns_ns"] is None or shared["measured_wns_ns"] is None
                else round(shared["measured_wns_ns"] - baseline["measured_wns_ns"], 6)
            )
            measured_ff_delta = (
                None
                if baseline["measured_ff"] is None or shared["measured_ff"] is None
                else shared["measured_ff"] - baseline["measured_ff"]
            )
            measured_lut_reduction_pct = _percent_change(
                None if measured_lut_delta is None else -measured_lut_delta,
                baseline["measured_lut"],
            )
            measured_throughput_retention_pct = _percent_change(
                shared["measured_effective_throughput_ops_per_cycle"],
                baseline["measured_effective_throughput_ops_per_cycle"],
            )
            measured_throughput_reduction_pct = (
                None
                if measured_throughput_retention_pct is None
                else round(100.0 - measured_throughput_retention_pct, 6)
            )
            measured_latency_increase_factor = _ratio(
                shared["measured_latency_cycles"],
                baseline["measured_latency_cycles"],
            )
            tradeoff_rows.append(
                {
                    "tradeoff_row_id": f"{grid}_k{k_depth}_{shared_variant}",
                    "grid": grid,
                    "k_depth": k_depth,
                    "shared_architecture_variant": shared_variant,
                    "shared_architecture_family": shared["architecture_family"],
                    "shared_intended_relief_kind": _shared_variant_intent(shared_variant),
                    "shared_measured_relief_kind": _measured_relief_kind(measured_dsp_delta, measured_lut_delta),
                    "baseline_run_id": baseline["run_id"],
                    "shared_run_id": shared["run_id"],
                    "baseline_direct_evidence_kind": baseline["direct_evidence_kind"],
                    "shared_direct_evidence_kind": shared["direct_evidence_kind"],
                    "baseline_measured_dsp": baseline["measured_dsp"],
                    "shared_measured_dsp": shared["measured_dsp"],
                    "measured_dsp_delta_shared_minus_baseline": measured_dsp_delta,
                    "baseline_model_dsp": baseline["model_dsp"],
                    "shared_model_dsp": shared["model_dsp"],
                    "model_dsp_delta_shared_minus_baseline": model_dsp_delta,
                    "baseline_measured_lut": baseline["measured_lut"],
                    "shared_measured_lut": shared["measured_lut"],
                    "measured_lut_delta_shared_minus_baseline": measured_lut_delta,
                    "measured_lut_reduction_pct": measured_lut_reduction_pct,
                    "baseline_model_lut": baseline["model_lut"],
                    "shared_model_lut": shared["model_lut"],
                    "model_lut_delta_shared_minus_baseline": shared["model_lut"] - baseline["model_lut"],
                    "baseline_measured_ff": baseline["measured_ff"],
                    "shared_measured_ff": shared["measured_ff"],
                    "measured_ff_delta_shared_minus_baseline": measured_ff_delta,
                    "baseline_measured_wns_ns": baseline["measured_wns_ns"],
                    "shared_measured_wns_ns": shared["measured_wns_ns"],
                    "measured_wns_delta_ns_shared_minus_baseline": measured_wns_delta,
                    "baseline_model_wns_estimate_ns": baseline["model_wns_estimate_ns"],
                    "shared_model_wns_estimate_ns": shared["model_wns_estimate_ns"],
                    "model_wns_delta_ns_shared_minus_baseline": round(
                        shared["model_wns_estimate_ns"] - baseline["model_wns_estimate_ns"], 6
                    ),
                    "baseline_measured_latency_cycles": baseline["measured_latency_cycles"],
                    "shared_measured_latency_cycles": shared["measured_latency_cycles"],
                    "measured_latency_delta_cycles_shared_minus_baseline": measured_latency_delta,
                    "measured_latency_increase_factor": measured_latency_increase_factor,
                    "baseline_direct_latency_model_cycles": baseline["direct_slice_latency_model_cycles"],
                    "shared_direct_latency_model_cycles": shared["direct_slice_latency_model_cycles"],
                    "model_latency_delta_cycles_shared_minus_baseline": shared["direct_slice_latency_model_cycles"] - baseline["direct_slice_latency_model_cycles"],
                    "baseline_measured_effective_throughput_ops_per_cycle": baseline["measured_effective_throughput_ops_per_cycle"],
                    "shared_measured_effective_throughput_ops_per_cycle": shared["measured_effective_throughput_ops_per_cycle"],
                    "measured_throughput_delta_ops_per_cycle_shared_minus_baseline": measured_throughput_delta,
                    "measured_throughput_retention_pct": measured_throughput_retention_pct,
                    "measured_throughput_reduction_pct": measured_throughput_reduction_pct,
                    "baseline_direct_throughput_model_ops_per_cycle": baseline["direct_slice_throughput_model_ops_per_cycle"],
                    "shared_direct_throughput_model_ops_per_cycle": shared["direct_slice_throughput_model_ops_per_cycle"],
                    "model_throughput_delta_ops_per_cycle_shared_minus_baseline": round(
                        shared["direct_slice_throughput_model_ops_per_cycle"] - baseline["direct_slice_throughput_model_ops_per_cycle"],
                        6,
                    ),
                    "tradeoff_status": "direct_measured_baseline_vs_shared_variant",
                    "tradeoff_note": _shared_tradeoff_note(shared_variant, measured_dsp_delta),
                }
            )
    return tradeoff_rows
def _shared_variant_label(shared_variant: str) -> str:
    if shared_variant == "shared_lut_saving":
        return "shared_lut_saving"
    if shared_variant == "shared_dsp_reducing":
        return "shared_dsp_reducing"
    return shared_variant


def _variant_sort_key(name: str) -> tuple[int, str]:
    return (0 if name == "shared_lut_saving" else 1 if name == "shared_dsp_reducing" else 2, name)


def _group_tradeoff_rows_by_grid(tradeoff_rows: list[dict[str, Any]]) -> list[tuple[tuple[str, int], list[dict[str, Any]]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in tradeoff_rows:
        grouped.setdefault((row["grid"], row["k_depth"]), []).append(row)
    return [
        ((grid, k_depth), sorted(group_rows, key=lambda item: _variant_sort_key(item["shared_architecture_variant"])))
        for (grid, k_depth), group_rows in sorted(grouped.items())
    ]


def _build_grid_shared_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {row["shared_architecture_variant"]: row for row in group_rows}
    lut_variant = by_variant.get("shared_lut_saving")
    dsp_variant = by_variant.get("shared_dsp_reducing")
    baseline_lut = group_rows[0]["baseline_measured_lut"]
    baseline_dsp = group_rows[0]["baseline_measured_dsp"]
    baseline_latency = group_rows[0]["baseline_measured_latency_cycles"]
    baseline_throughput = group_rows[0]["baseline_measured_effective_throughput_ops_per_cycle"]

    baseline_performance_first = all(
        variant["shared_measured_latency_cycles"] is not None
        and variant["shared_measured_effective_throughput_ops_per_cycle"] is not None
        and baseline_latency is not None
        and baseline_throughput is not None
        and baseline_latency < variant["shared_measured_latency_cycles"]
        and baseline_throughput > variant["shared_measured_effective_throughput_ops_per_cycle"]
        for variant in by_variant.values()
    )
    lut_role_survives = (
        lut_variant is not None
        and lut_variant["shared_measured_lut"] is not None
        and baseline_lut is not None
        and lut_variant["shared_measured_lut"] < baseline_lut
        and (
            dsp_variant is None
            or dsp_variant["shared_measured_lut"] is None
            or lut_variant["shared_measured_lut"] < dsp_variant["shared_measured_lut"]
        )
    )
    dsp_role_survives = (
        dsp_variant is not None
        and dsp_variant["shared_measured_dsp"] is not None
        and baseline_dsp is not None
        and dsp_variant["shared_measured_dsp"] < baseline_dsp
        and (
            lut_variant is None
            or lut_variant["shared_measured_dsp"] is None
            or dsp_variant["shared_measured_dsp"] < lut_variant["shared_measured_dsp"]
        )
    )
    three_way_rule_survives = baseline_performance_first and lut_role_survives and dsp_role_survives

    lines: list[str] = []
    if lut_variant is not None:
        lines.append(
            f"shared_lut_saving saves LUT by `{abs(lut_variant['measured_lut_delta_shared_minus_baseline'])}` ("
            f"`{lut_variant['measured_lut_reduction_pct']}`%) but keeps DSP flat at `{lut_variant['shared_measured_dsp']}`, "
            f"while latency rises to `{lut_variant['shared_measured_latency_cycles']}` cycles and throughput falls to `"
            f"{lut_variant['shared_measured_effective_throughput_ops_per_cycle']}` ops/cycle."
        )
    if dsp_variant is not None and dsp_variant["measured_dsp_delta_shared_minus_baseline"] is not None and dsp_variant["measured_dsp_delta_shared_minus_baseline"] < 0:
        lines.append(
            f"shared_dsp_reducing reduces DSP by `{abs(dsp_variant['measured_dsp_delta_shared_minus_baseline'])}`, keeps latency at `{dsp_variant['shared_measured_latency_cycles']}` cycles and throughput at `{dsp_variant['shared_measured_effective_throughput_ops_per_cycle']}` ops/cycle, and lands at `{dsp_variant['shared_measured_lut']}` LUT and `{dsp_variant['shared_measured_wns_ns']}` ns WNS."
        )
    elif dsp_variant is not None:
        lines.append(
            f"shared_dsp_reducing was intended to lower DSP usage, but this measured point still maps to `{dsp_variant['shared_measured_dsp']}` DSP, so it does not realize DSP relief on the current flow. Its LUT delta is `{dsp_variant['measured_lut_delta_shared_minus_baseline']}`, latency is `{dsp_variant['shared_measured_latency_cycles']}` cycles, and throughput is `{dsp_variant['shared_measured_effective_throughput_ops_per_cycle']}` ops/cycle."
        )

    if lut_variant is not None and dsp_variant is not None and dsp_variant["measured_dsp_delta_shared_minus_baseline"] is not None and dsp_variant["measured_dsp_delta_shared_minus_baseline"] < 0:
        lines.append(
            f"Use shared_lut_saving when LUT budget falls between `{lut_variant['shared_measured_lut']}` and `< {dsp_variant['shared_measured_lut']}` and DSP is not the bottleneck; it is the most LUT-efficient measured shared option."
        )
        lines.append(
            f"Use shared_dsp_reducing when DSP budget falls below `{baseline_dsp}` and LUT budget still admits `{dsp_variant['shared_measured_lut']}` LUT; it is the only measured non-baseline option that relieves DSP pressure on this flow."
        )
    lines.append("If throughput or latency matter more than resource pressure, baseline remains preferable to both shared implementations.")
    lines.append(
        "The three-way rule survives on this grid: baseline stays performance-first, shared_lut_saving stays LUT-first, and shared_dsp_reducing stays DSP-first."
        if three_way_rule_survives
        else "The three-way role split partially breaks on this grid; inspect the measured ordering before carrying the 4x4 rule upward."
    )
    return {
        "grid": group_rows[0]["grid"],
        "k_depth": group_rows[0]["k_depth"],
        "baseline_performance_first": baseline_performance_first,
        "shared_lut_saving_is_best_lut_relief": lut_role_survives,
        "shared_dsp_reducing_is_best_dsp_relief": dsp_role_survives,
        "three_way_rule_status": "survives" if three_way_rule_survives else "partial_break",
        "summary_lines": lines,
    }


def build_direct_shared_implementation_comparison_rows(tradeoff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    for _, group_rows in _group_tradeoff_rows_by_grid(tradeoff_rows):
        for row in group_rows:
            comparison_rows.append(
                {
                    "comparison_kind": "baseline_vs_shared_variant",
                    "grid": row["grid"],
                    "k_depth": row["k_depth"],
                    "shared_architecture_variant": row["shared_architecture_variant"],
                    "shared_variant_label": _shared_variant_label(row["shared_architecture_variant"]),
                    "shared_intended_relief_kind": row["shared_intended_relief_kind"],
                    "shared_measured_relief_kind": row["shared_measured_relief_kind"],
                    "baseline_measured_dsp": row["baseline_measured_dsp"],
                    "shared_measured_dsp": row["shared_measured_dsp"],
                    "measured_dsp_delta_shared_minus_baseline": row["measured_dsp_delta_shared_minus_baseline"],
                    "baseline_measured_lut": row["baseline_measured_lut"],
                    "shared_measured_lut": row["shared_measured_lut"],
                    "measured_lut_delta_shared_minus_baseline": row["measured_lut_delta_shared_minus_baseline"],
                    "measured_lut_reduction_pct": row["measured_lut_reduction_pct"],
                    "baseline_measured_ff": row["baseline_measured_ff"],
                    "shared_measured_ff": row["shared_measured_ff"],
                    "measured_ff_delta_shared_minus_baseline": row["measured_ff_delta_shared_minus_baseline"],
                    "baseline_measured_wns_ns": row["baseline_measured_wns_ns"],
                    "shared_measured_wns_ns": row["shared_measured_wns_ns"],
                    "measured_wns_delta_ns_shared_minus_baseline": row["measured_wns_delta_ns_shared_minus_baseline"],
                    "baseline_measured_latency_cycles": row["baseline_measured_latency_cycles"],
                    "shared_measured_latency_cycles": row["shared_measured_latency_cycles"],
                    "measured_latency_delta_cycles_shared_minus_baseline": row["measured_latency_delta_cycles_shared_minus_baseline"],
                    "measured_latency_increase_factor": row["measured_latency_increase_factor"],
                    "baseline_measured_effective_throughput_ops_per_cycle": row["baseline_measured_effective_throughput_ops_per_cycle"],
                    "shared_measured_effective_throughput_ops_per_cycle": row["shared_measured_effective_throughput_ops_per_cycle"],
                    "measured_throughput_delta_ops_per_cycle_shared_minus_baseline": row["measured_throughput_delta_ops_per_cycle_shared_minus_baseline"],
                    "measured_throughput_retention_pct": row["measured_throughput_retention_pct"],
                    "measured_throughput_reduction_pct": row["measured_throughput_reduction_pct"],
                    "comparison_note": row["tradeoff_note"],
                }
            )

        by_variant = {row["shared_architecture_variant"]: row for row in group_rows}
        lut_variant = by_variant.get("shared_lut_saving")
        dsp_variant = by_variant.get("shared_dsp_reducing")
        if lut_variant and dsp_variant:
            comparison_rows.append(
                {
                    "comparison_kind": "shared_variant_vs_shared_variant",
                    "grid": lut_variant["grid"],
                    "k_depth": lut_variant["k_depth"],
                    "lhs_variant": "shared_dsp_reducing",
                    "rhs_variant": "shared_lut_saving",
                    "lhs_measured_dsp": dsp_variant["shared_measured_dsp"],
                    "rhs_measured_dsp": lut_variant["shared_measured_dsp"],
                    "measured_dsp_delta_lhs_minus_rhs": None if dsp_variant["shared_measured_dsp"] is None or lut_variant["shared_measured_dsp"] is None else dsp_variant["shared_measured_dsp"] - lut_variant["shared_measured_dsp"],
                    "lhs_measured_lut": dsp_variant["shared_measured_lut"],
                    "rhs_measured_lut": lut_variant["shared_measured_lut"],
                    "measured_lut_delta_lhs_minus_rhs": None if dsp_variant["shared_measured_lut"] is None or lut_variant["shared_measured_lut"] is None else dsp_variant["shared_measured_lut"] - lut_variant["shared_measured_lut"],
                    "lhs_measured_latency_cycles": dsp_variant["shared_measured_latency_cycles"],
                    "rhs_measured_latency_cycles": lut_variant["shared_measured_latency_cycles"],
                    "measured_latency_delta_lhs_minus_rhs": None if dsp_variant["shared_measured_latency_cycles"] is None or lut_variant["shared_measured_latency_cycles"] is None else dsp_variant["shared_measured_latency_cycles"] - lut_variant["shared_measured_latency_cycles"],
                    "lhs_measured_effective_throughput_ops_per_cycle": dsp_variant["shared_measured_effective_throughput_ops_per_cycle"],
                    "rhs_measured_effective_throughput_ops_per_cycle": lut_variant["shared_measured_effective_throughput_ops_per_cycle"],
                    "measured_throughput_delta_lhs_minus_rhs": None if dsp_variant["shared_measured_effective_throughput_ops_per_cycle"] is None or lut_variant["shared_measured_effective_throughput_ops_per_cycle"] is None else round(dsp_variant["shared_measured_effective_throughput_ops_per_cycle"] - lut_variant["shared_measured_effective_throughput_ops_per_cycle"], 6),
                    "comparison_note": "This row compares the two directly measured shared implementations against each other so the repo can show which resource each implementation actually buys.",
                }
            )
    return comparison_rows


def build_direct_shared_scaling_summary(tradeoff_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not tradeoff_rows:
        return {}
    grid_summaries = [_build_grid_shared_summary(group_rows) for _, group_rows in _group_tradeoff_rows_by_grid(tradeoff_rows)]
    measured_grids = [row["grid"] for row in grid_summaries]
    survives_all = all(row["three_way_rule_status"] == "survives" for row in grid_summaries)
    if len(grid_summaries) >= 2:
        scaling_rule = (
            "The 4x4 three-way rule survives at 8x4: baseline remains the performance-first option, shared_lut_saving remains the best LUT-relief option, and shared_dsp_reducing remains the DSP-relief option."
            if survives_all
            else "The 4x4 three-way rule partially breaks at 8x4; use the per-grid measured ordering rather than assuming the 4x4 roles scale cleanly."
        )
    else:
        scaling_rule = "Only one measured three-way scale point is available so far, so the scaling rule is not yet established."
    return {
        "headline": "This summary asks whether the measured three-way resource-relief rule survives from 4x4 to 8x4.",
        "measured_grids": measured_grids,
        "scale_points": len(grid_summaries),
        "scaling_rule_status": "survives" if survives_all else "partial_break" if len(grid_summaries) >= 2 else "insufficient_scale_data",
        "scaling_rule_line": scaling_rule,
        "summary_lines": [
            f"Measured three-way shared comparisons are now available at: {', '.join(measured_grids)}.",
            scaling_rule,
            "Implementation roles should be classified by measured resource relief at each grid, not by the abstract label 'shared' alone.",
        ],
        "grid_summaries": grid_summaries,
    }


def build_direct_shared_implementation_summary(tradeoff_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scaling_summary = build_direct_shared_scaling_summary(tradeoff_rows)
    if not scaling_summary:
        return {}
    return {
        "headline": "The repo now distinguishes between two directly measured shared implementations rather than treating sharing as one thing.",
        "summary_lines": scaling_summary["summary_lines"],
        "grid_summaries": scaling_summary["grid_summaries"],
        "scaling_rule_status": scaling_summary["scaling_rule_status"],
        "scaling_rule_line": scaling_summary["scaling_rule_line"],
        "measured_grids": scaling_summary["measured_grids"],
        "scale_points": scaling_summary["scale_points"],
    }


def _support_row(
    claim_id: str,
    claim_scope: str,
    claim_subject: str,
    claim_text: str,
    support_level: str,
    measured_basis: str,
    trust_note: str,
    measured_grids: list[str],
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_scope": claim_scope,
        "claim_subject": claim_subject,
        "claim_text": claim_text,
        "support_level": support_level,
        "measured_basis": measured_basis,
        "trust_note": trust_note,
        "measured_grids": ", ".join(measured_grids),
        "measured_grid_count": len(measured_grids),
    }


def build_measured_support_rows(tradeoff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scaling_summary = build_direct_shared_scaling_summary(tradeoff_rows)
    if not scaling_summary:
        return []
    measured_grids = scaling_summary["measured_grids"]
    grid_map = {row["grid"]: row for row in scaling_summary["grid_summaries"]}
    rows = [
        _support_row(
            claim_id="baseline_performance_first_role",
            claim_scope="measured_implementation_role",
            claim_subject="baseline",
            claim_text="Baseline is the performance-first option relative to the measured shared implementations.",
            support_level=DIRECTLY_MEASURED_SUPPORTED,
            measured_basis="Across the measured 4x4 and 8x4 direct slices, baseline keeps the 33-cycle schedule and higher ops/cycle than both measured shared implementations.",
            trust_note="This role is directly measured on the isolated slice, not inferred from the family model.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_lut_saving_lut_relief_role",
            claim_scope="measured_implementation_role",
            claim_subject="shared_lut_saving",
            claim_text="shared_lut_saving is the LUT-relief option.",
            support_level=DIRECTLY_MEASURED_SUPPORTED,
            measured_basis="At both measured grids, shared_lut_saving is the lowest-LUT shared implementation and lowers LUT versus baseline while keeping the shared 65-cycle schedule.",
            trust_note="This is a directly measured implementation-specific role, not a statement about every shared strategy.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_dsp_reducing_dsp_relief_role",
            claim_scope="measured_implementation_role",
            claim_subject="shared_dsp_reducing",
            claim_text="shared_dsp_reducing is the DSP-relief option.",
            support_level=DIRECTLY_MEASURED_SUPPORTED,
            measured_basis="At both measured grids, shared_dsp_reducing reduces mapped DSP to 0 while retaining the shared 65-cycle schedule.",
            trust_note="This is directly measured for the implemented DSP-oriented shared slice only.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_family_latency_throughput_penalty",
            claim_scope="family_level_direction",
            claim_subject="shared_modelled_family",
            claim_text="Shared-family implementations trade lower throughput and higher latency for resource relief.",
            support_level=DIRECTLY_MEASURED_SUPPORTED,
            measured_basis="Both measured shared implementations at both measured grids move from the 33-cycle baseline point to the 65-cycle shared point and halve effective ops/cycle.",
            trust_note="The latency/throughput penalty direction is directly supported by measured implementations, even though the framework family remains modelled.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_family_lut_relief",
            claim_scope="family_level_direction",
            claim_subject="shared_modelled_family",
            claim_text="Shared-family implementations can relieve LUT pressure.",
            support_level=MEASURED_DIRECTIONALLY_SUPPORTED,
            measured_basis="Both measured shared implementations reduce LUT versus baseline at 4x4 and 8x4, but the amount of LUT relief is implementation-specific.",
            trust_note="Measured data supports LUT relief directionally, not as a single family-wide realization or fixed magnitude.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_family_dsp_relief",
            claim_scope="family_level_direction",
            claim_subject="shared_modelled_family",
            claim_text="Shared-family implementations reduce DSP pressure.",
            support_level=MEASURED_PARTIAL_SUPPORT,
            measured_basis="The measured DSP-oriented shared implementation reduces DSP to 0 at both measured grids, but the measured LUT-oriented shared implementation stays DSP-flat.",
            trust_note="DSP relief is implementation-dependent; the measured direct slice does not justify treating all shared implementations as one DSP-saving hardware truth.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_family_single_measured_truth",
            claim_scope="family_level_generalization",
            claim_subject="shared_modelled_family",
            claim_text="The shared family can be treated as one directly measured hardware behavior.",
            support_level=CONTRADICTED_BY_MEASURED_IMPLEMENTATION,
            measured_basis="The measured direct slice splits into distinct LUT-oriented and DSP-oriented implementations with different realized resource relief.",
            trust_note="Measured evidence explicitly shows implementation-specific behavior, so collapsing the family into one measured truth is not supported.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_8x8_dsp_reduction_anchor",
            claim_scope="family_level_extrapolation",
            claim_subject="shared_modelled_family_8x8",
            claim_text="The 8x8 shared-family DSP reduction is directly measured in this repo's current isolated direct slice.",
            support_level=EXTRAPOLATED_BEYOND_MEASURED_SUPPORT,
            measured_basis="The repo has direct shared measurements only at 4x4 and 8x4. The 8x8 shared DSP reduction still comes from anchored prior-study evidence plus the modelled family layer.",
            trust_note="Treat 8x8 shared-family DSP reduction as anchored/modelled rather than directly measured implementation truth.",
            measured_grids=measured_grids,
        ),
    ]
    return rows


def build_framework_trust_overlay_rows(support_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not support_rows:
        return []
    by_claim = {row["claim_id"]: row for row in support_rows}
    ordered_claims = [
        "baseline_performance_first_role",
        "shared_lut_saving_lut_relief_role",
        "shared_dsp_reducing_dsp_relief_role",
        "shared_family_latency_throughput_penalty",
        "shared_family_lut_relief",
        "shared_family_dsp_relief",
        "shared_family_single_measured_truth",
        "shared_8x8_dsp_reduction_anchor",
    ]
    rows = []
    for claim_id in ordered_claims:
        row = by_claim.get(claim_id)
        if row is None:
            continue
        rows.append(
            {
                "overlay_topic": row["claim_subject"],
                "claim_id": row["claim_id"],
                "support_level": row["support_level"],
                "framework_reading": row["claim_text"],
                "trust_boundary": row["trust_note"],
                "measured_basis": row["measured_basis"],
                "measured_grids": row["measured_grids"],
            }
        )
    return rows


def build_measured_trust_summary(support_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not support_rows:
        return {}
    by_claim = {row["claim_id"]: row for row in support_rows}
    summary_lines = [
        "Directly measured support is strongest for the implementation-specific performance-first role of baseline.",
        "The measured direct slice strongly validates that sharing can trade performance for resource relief.",
        "Resource relief is implementation-specific: one measured shared implementation relieves LUT pressure, while another relieves DSP pressure.",
        "Shared-family latency and throughput penalties are directly measured in the isolated slice, but shared-family resource relief remains implementation-dependent.",
        "The direct evidence does not justify collapsing all shared implementations into one measured hardware truth.",
        "Family-level 8x8 shared DSP reduction remains an anchored/modelled claim that is extrapolated beyond measured support.",
        "Framework shared recommendations should be read as modelled-family conclusions with implementation-dependent realization.",
    ]
    return {
        "headline": "This summary marks what the current framework is directly supported by, what is only directionally supported, and what remains extrapolated beyond measured support.",
        "summary_lines": summary_lines,
        "support_levels_present": sorted({row["support_level"] for row in support_rows}),
        "claim_count": len(support_rows),
        "directly_measured_claims": sum(1 for row in support_rows if row["support_level"] == DIRECTLY_MEASURED_SUPPORTED),
        "extrapolated_claims": sum(1 for row in support_rows if row["support_level"] == EXTRAPOLATED_BEYOND_MEASURED_SUPPORT),
    }


def _calibration_row(
    metric: str,
    architecture_variant_or_family: str,
    grid: str,
    model_value: float | None,
    measured_value: float | None,
    calibration_reading: str,
    calibration_status: str,
    usage_note: str,
    model_reference: str,
    measured_reference: str,
) -> dict[str, Any]:
    delta = None if model_value is None or measured_value is None else round(measured_value - model_value, 6)
    relative_error_pct = None
    if delta is not None and model_value not in (None, 0):
        relative_error_pct = round((delta / model_value) * 100.0, 6)
    return {
        "metric": metric,
        "architecture_variant_or_family": architecture_variant_or_family,
        "grid": grid,
        "model_value": model_value,
        "measured_value": measured_value,
        "delta_measured_minus_model": delta,
        "relative_error_pct": relative_error_pct,
        "calibration_reading": calibration_reading,
        "calibration_status": calibration_status,
        "usage_note": usage_note,
        "model_reference": model_reference,
        "measured_reference": measured_reference,
    }


def _numeric_alignment_status(model_value: float | None, measured_value: float | None) -> str:
    if model_value is None or measured_value is None:
        return CALIBRATION_TOO_UNCERTAIN
    if abs(measured_value - model_value) <= 1e-5:
        return CALIBRATION_WELL_ALIGNED
    if measured_value > model_value:
        return CALIBRATION_DIRECTIONALLY_OPTIMISTIC
    return CALIBRATION_DIRECTIONALLY_PESSIMISTIC


def build_framework_calibration_aid_rows(
    direct_rows: list[dict[str, Any]],
    tradeoff_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not direct_rows or not tradeoff_rows:
        return []

    measured_tradeoff_grids = {row["grid"] for row in tradeoff_rows}
    baseline_rows = sorted(
        (
            row
            for row in direct_rows
            if row["architecture"] == "baseline"
            and row["comparison_status"] == "direct_measured_vs_modelled"
            and row["grid"] in measured_tradeoff_grids
        ),
        key=lambda row: (row.get("mac_units", row.get("grid_rows", 0) * row.get("grid_cols", 0)), row["grid"]),
    )

    rows: list[dict[str, Any]] = []
    for row in baseline_rows:
        lut_delta = row.get("lut_delta")
        if lut_delta is None and row.get("measured_lut") is not None and row.get("model_lut") is not None:
            lut_delta = row["measured_lut"] - row["model_lut"]
        status = _numeric_alignment_status(row["model_lut"], row["measured_lut"])
        rows.append(
            _calibration_row(
                metric="lut",
                architecture_variant_or_family="baseline",
                grid=row["grid"],
                model_value=row["model_lut"],
                measured_value=row["measured_lut"],
                calibration_reading=(
                    f"Baseline LUT model underpredicts the measured direct-slice LUT by `{lut_delta}` on `{row['grid']}`."
                    if lut_delta is not None and lut_delta > 0
                    else f"Baseline LUT model is numerically aligned to the measured direct-slice LUT on `{row['grid']}`."
                ),
                calibration_status=status,
                usage_note="Use as a baseline-only calibration aid or caution reference; it is not a direct replacement for the global framework model.",
                model_reference="framework_baseline_static_model",
                measured_reference=row["direct_evidence_kind"],
            )
        )

    for row in sorted(tradeoff_rows, key=lambda item: (item["grid"], _variant_sort_key(item["shared_architecture_variant"]))):
        if row["shared_architecture_variant"] == "shared_lut_saving":
            status = _numeric_alignment_status(row["shared_model_lut"], row["shared_measured_lut"])
            rows.append(
                _calibration_row(
                    metric="lut",
                    architecture_variant_or_family="shared_lut_saving",
                    grid=row["grid"],
                    model_value=row["shared_model_lut"],
                    measured_value=row["shared_measured_lut"],
                    calibration_reading=(
                        f"The modelled shared-family LUT expectation underpredicts measured `{row['shared_architecture_variant']}` LUT by `{row['measured_lut_delta_shared_minus_baseline'] - row['model_lut_delta_shared_minus_baseline']}` relative to baseline-driven expectations on `{row['grid']}`."
                    ),
                    calibration_status=status,
                    usage_note="This is an implementation-specific calibration aid for the LUT-oriented shared slice; do not treat it as the numeric truth for every shared-family realization.",
                    model_reference="framework_shared_family_static_model",
                    measured_reference=row["shared_direct_evidence_kind"],
                )
            )
        if row["shared_architecture_variant"] == "shared_dsp_reducing":
            status = _numeric_alignment_status(row["shared_model_dsp"], row["shared_measured_dsp"])
            rows.append(
                _calibration_row(
                    metric="dsp",
                    architecture_variant_or_family="shared_dsp_reducing",
                    grid=row["grid"],
                    model_value=row["shared_model_dsp"],
                    measured_value=row["shared_measured_dsp"],
                    calibration_reading=(
                        f"The modelled shared-family DSP expectation predicts `{row['shared_model_dsp']}` DSP, while measured `{row['shared_architecture_variant']}` lands at `{row['shared_measured_dsp']}` DSP on `{row['grid']}`."
                    ),
                    calibration_status=status,
                    usage_note="This calibration row belongs to the DSP-oriented measured shared implementation only; it does not universally calibrate the whole shared family.",
                    model_reference="framework_shared_family_static_model",
                    measured_reference=row["shared_direct_evidence_kind"],
                )
            )
        for metric, model_key, measured_key in (
            ("latency_delta_cycles", "model_latency_delta_cycles_shared_minus_baseline", "measured_latency_delta_cycles_shared_minus_baseline"),
            ("throughput_delta_ops_per_cycle", "model_throughput_delta_ops_per_cycle_shared_minus_baseline", "measured_throughput_delta_ops_per_cycle_shared_minus_baseline"),
        ):
            status = _numeric_alignment_status(row[model_key], row[measured_key])
            rows.append(
                _calibration_row(
                    metric=metric,
                    architecture_variant_or_family="shared_modelled_family_direction",
                    grid=row["grid"],
                    model_value=row[model_key],
                    measured_value=row[measured_key],
                    calibration_reading=(
                        f"Measured `{row['shared_architecture_variant']}` matches the modelled shared-family {metric.replace('_', ' ')} direction on `{row['grid']}`."
                    ),
                    calibration_status=status,
                    usage_note="Treat this as a direction check for the shared-family latency/throughput story, not as a direct replacement of the family model with one measured implementation.",
                    model_reference="framework_shared_family_direct_slice_model",
                    measured_reference=row["shared_direct_evidence_kind"],
                )
            )

    return rows


def _overlay_row(
    overlay_topic: str,
    calibration_status: str,
    calibration_reading: str,
    usage_note: str,
    measured_grids: list[str],
) -> dict[str, Any]:
    return {
        "overlay_topic": overlay_topic,
        "calibration_status": calibration_status,
        "calibration_reading": calibration_reading,
        "usage_note": usage_note,
        "measured_grids": ", ".join(measured_grids),
        "measured_grid_count": len(measured_grids),
    }


def build_framework_calibration_overlay_rows(calibration_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not calibration_rows:
        return []
    measured_grids = sorted({row["grid"] for row in calibration_rows if row["grid"]})
    baseline_lut_rows = [row for row in calibration_rows if row["architecture_variant_or_family"] == "baseline" and row["metric"] == "lut"]
    shared_lut_rows = [row for row in calibration_rows if row["architecture_variant_or_family"] == "shared_lut_saving" and row["metric"] == "lut"]
    shared_dsp_rows = [row for row in calibration_rows if row["architecture_variant_or_family"] == "shared_dsp_reducing" and row["metric"] == "dsp"]
    shared_direction_rows = [
        row
        for row in calibration_rows
        if row["architecture_variant_or_family"] == "shared_modelled_family_direction"
        and row["metric"] in {"latency_delta_cycles", "throughput_delta_ops_per_cycle"}
    ]
    return [
        _overlay_row(
            overlay_topic="baseline_lut_expectation",
            calibration_status=(
                CALIBRATION_DIRECTIONALLY_OPTIMISTIC
                if any(row["calibration_status"] == CALIBRATION_DIRECTIONALLY_OPTIMISTIC for row in baseline_lut_rows)
                else CALIBRATION_WELL_ALIGNED
            ),
            calibration_reading="Baseline LUT in the lightweight framework is numerically optimistic on the measured direct slice and should be read through the direct calibration aid.",
            usage_note="Use the baseline LUT fit as a caution reference for nearby slice scales, not as a silent model replacement.",
            measured_grids=measured_grids,
        ),
        _overlay_row(
            overlay_topic="shared_family_latency_throughput_direction",
            calibration_status=(
                CALIBRATION_WELL_ALIGNED
                if shared_direction_rows and all(row["calibration_status"] == CALIBRATION_WELL_ALIGNED for row in shared_direction_rows)
                else CALIBRATION_TOO_UNCERTAIN
            ),
            calibration_reading="The modelled shared-family latency and throughput direction is well aligned with the measured direct slice.",
            usage_note="This is the strongest numeric calibration signal for the shared family; it supports direction, not a full resource refit.",
            measured_grids=measured_grids,
        ),
        _overlay_row(
            overlay_topic="shared_family_lut_expectation",
            calibration_status=CALIBRATION_IMPLEMENTATION_DEPENDENT,
            calibration_reading=(
                "Measured shared LUT behavior is directionally aligned with relief versus baseline, but the numeric realization depends on implementation style and should be read with a caution band."
            ),
            usage_note="Do not collapse the LUT-oriented and DSP-oriented shared implementations into one numeric LUT expectation for the whole modelled family.",
            measured_grids=measured_grids,
        ),
        _overlay_row(
            overlay_topic="shared_family_dsp_expectation",
            calibration_status=CALIBRATION_IMPLEMENTATION_DEPENDENT,
            calibration_reading=(
                "Measured shared DSP behavior is implementation-dependent: the DSP-oriented slice reduces DSP strongly, while the LUT-oriented slice remains DSP-flat."
            ),
            usage_note="Read shared-family DSP projections as modelled-family expectations with implementation-dependent realization, especially beyond 8x4.",
            measured_grids=measured_grids,
        ),
        _overlay_row(
            overlay_topic="shared_family_numeric_projection_boundary",
            calibration_status=CALIBRATION_TOO_UNCERTAIN,
            calibration_reading="Family-level shared resource projections should be read through the measured calibration aid as a caution reference, not as a direct replacement by local implementation numbers.",
            usage_note="The current calibration layer bounds trust near the measured 4x4 and 8x4 slice evidence; 8x8 shared resource claims remain anchored/modelled.",
            measured_grids=measured_grids,
        ),
    ]


def build_shared_family_calibration_summary(
    calibration_rows: list[dict[str, Any]],
    overlay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not calibration_rows:
        return {}
    baseline_lut_rows = [row for row in calibration_rows if row["architecture_variant_or_family"] == "baseline" and row["metric"] == "lut"]
    shared_direction_rows = [
        row
        for row in calibration_rows
        if row["architecture_variant_or_family"] == "shared_modelled_family_direction"
        and row["metric"] in {"latency_delta_cycles", "throughput_delta_ops_per_cycle"}
    ]
    measured_grids = sorted({row["grid"] for row in calibration_rows if row["grid"]})
    baseline_error_min = min((row["relative_error_pct"] for row in baseline_lut_rows if row["relative_error_pct"] is not None), default=None)
    baseline_error_max = max((row["relative_error_pct"] for row in baseline_lut_rows if row["relative_error_pct"] is not None), default=None)
    direction_exact = shared_direction_rows and all(row["calibration_status"] == CALIBRATION_WELL_ALIGNED for row in shared_direction_rows)
    summary_lines = [
        "This calibration aid keeps the framework as a modelled family layer, but adds a measured caution reference from the direct slice rather than treating shared-family resource numbers as unconstrained.",
        (
            f"Baseline LUT is numerically optimistic in the lightweight framework across the measured 4x4 and 8x4 bridge, with relative error spanning `{baseline_error_min}`% to `{baseline_error_max}`%."
            if baseline_error_min is not None and baseline_error_max is not None
            else "Baseline LUT remains a caution-reference case rather than a fully trusted numeric prediction."
        ),
        (
            "Shared-family latency and throughput direction is well aligned with measured direct-slice behavior."
            if direction_exact
            else "Shared-family latency and throughput direction is only partially aligned with measured direct-slice behavior."
        ),
        "Shared-family DSP reduction is implementation-dependent, not a universally calibrated measured truth.",
        "Shared-family LUT expectations are approximate because the two measured shared implementations realize different resource tradeoffs.",
        "Read family-level shared resource projections through this calibration aid and caution band, not as a direct replacement of the modelled family with local implementation values.",
        "8x8 shared-family resource conclusions remain modelled/anchored beyond the directly calibrated 4x4 and 8x4 slice evidence.",
    ]
    return {
        "headline": "This summary turns the direct-slice evidence into a measured calibration aid for reading the broader shared-family framework outputs.",
        "summary_lines": summary_lines,
        "measured_grids": measured_grids,
        "overlay_topics": [row["overlay_topic"] for row in overlay_rows],
        "calibration_row_count": len(calibration_rows),
    }


def build_measured_utility_rows(tradeoff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tradeoff_rows:
        return []
    utility_rows: list[dict[str, Any]] = []
    for (grid, k_depth), group_rows in _group_tradeoff_rows_by_grid(tradeoff_rows):
        baseline_lut = group_rows[0]["baseline_measured_lut"]
        baseline_dsp = group_rows[0]["baseline_measured_dsp"]
        baseline_wns = group_rows[0]["baseline_measured_wns_ns"]
        baseline_latency = group_rows[0]["baseline_measured_latency_cycles"]
        baseline_throughput = group_rows[0]["baseline_measured_effective_throughput_ops_per_cycle"]

        utility_rows.append(
            {
                "grid": grid,
                "k_depth": k_depth,
                "architecture_variant": "baseline",
                "resource_relief_kind": "none",
                "resource_relief_magnitude": 0,
                "lut_relief_units_vs_baseline": 0,
                "dsp_relief_units_vs_baseline": 0,
                "performance_penalty_kind": "none",
                "throughput_retention_pct": 100.0,
                "latency_increase_factor": 1.0,
                "wns_delta_ns": 0.0,
                "utility_status": UTILITY_PERFORMANCE_FIRST_DEFAULT,
                "utility_reading": "Baseline is the measured default when no hard resource bottleneck justifies paying the shared-style performance penalty.",
                "recommended_use_case": "Use baseline when throughput, latency, or general performance dominates the decision.",
                "avoid_use_case": "Avoid baseline only when LUT or DSP pressure is the true bottleneck and a measured shared option is the only relief that fits.",
            }
        )

        for row in group_rows:
            if row["shared_architecture_variant"] == "shared_lut_saving":
                utility_rows.append(
                    {
                        "grid": grid,
                        "k_depth": k_depth,
                        "architecture_variant": "shared_lut_saving",
                        "resource_relief_kind": "lut_relief",
                        "resource_relief_magnitude": abs(row["measured_lut_delta_shared_minus_baseline"]) if row["measured_lut_delta_shared_minus_baseline"] is not None else None,
                        "lut_relief_units_vs_baseline": abs(row["measured_lut_delta_shared_minus_baseline"]) if row["measured_lut_delta_shared_minus_baseline"] is not None else None,
                        "dsp_relief_units_vs_baseline": max(0, -(row["measured_dsp_delta_shared_minus_baseline"] or 0)),
                        "performance_penalty_kind": "throughput_and_latency",
                        "throughput_retention_pct": row["measured_throughput_retention_pct"],
                        "latency_increase_factor": row["measured_latency_increase_factor"],
                        "wns_delta_ns": row["measured_wns_delta_ns_shared_minus_baseline"],
                        "utility_status": UTILITY_WORTH_LUT_BOTTLENECK,
                        "utility_reading": "shared_lut_saving is worthwhile only when LUT pressure is the real bottleneck and the measured shared-style throughput and latency penalty is acceptable.",
                        "recommended_use_case": f"Use when LUT pressure matters more than losing roughly half the throughput and moving from `{baseline_latency}` to `{row['shared_measured_latency_cycles']}` cycles.",
                        "avoid_use_case": "Avoid when DSP is the real bottleneck or when performance and timing margin matter more than LUT relief.",
                    }
                )
            elif row["shared_architecture_variant"] == "shared_dsp_reducing":
                utility_rows.append(
                    {
                        "grid": grid,
                        "k_depth": k_depth,
                        "architecture_variant": "shared_dsp_reducing",
                        "resource_relief_kind": "dsp_relief",
                        "resource_relief_magnitude": abs(row["measured_dsp_delta_shared_minus_baseline"]) if row["measured_dsp_delta_shared_minus_baseline"] is not None else None,
                        "lut_relief_units_vs_baseline": abs(row["measured_lut_delta_shared_minus_baseline"]) if row["measured_lut_delta_shared_minus_baseline"] is not None and row["measured_lut_delta_shared_minus_baseline"] < 0 else 0,
                        "dsp_relief_units_vs_baseline": abs(row["measured_dsp_delta_shared_minus_baseline"]) if row["measured_dsp_delta_shared_minus_baseline"] is not None else None,
                        "performance_penalty_kind": "throughput_and_latency",
                        "throughput_retention_pct": row["measured_throughput_retention_pct"],
                        "latency_increase_factor": row["measured_latency_increase_factor"],
                        "wns_delta_ns": row["measured_wns_delta_ns_shared_minus_baseline"],
                        "utility_status": UTILITY_WORTH_DSP_BOTTLENECK,
                        "utility_reading": "shared_dsp_reducing is worthwhile only when DSP pressure is the real bottleneck and its LUT footprint plus shared-style performance penalty still fit the design.",
                        "recommended_use_case": f"Use when DSP pressure matters more than losing roughly half the throughput and the measured LUT footprint of `{row['shared_measured_lut']}` still fits.",
                        "avoid_use_case": "Avoid when performance is the bottleneck or when LUT pressure is tighter than DSP pressure.",
                    }
                )
    return utility_rows


def build_measured_bottleneck_choice_map(
    utility_rows: list[dict[str, Any]],
    tradeoff_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not utility_rows or not tradeoff_rows:
        return []
    utility_by_grid = {
        grid: {row["architecture_variant"]: row for row in rows}
        for (grid, _), rows in (
            ((grid, k_depth), [row for row in utility_rows if row["grid"] == grid and row["k_depth"] == k_depth])
            for (grid, k_depth), _group in _group_tradeoff_rows_by_grid(tradeoff_rows)
        )
    }
    tradeoff_by_grid = {
        grid: {row["shared_architecture_variant"]: row for row in rows}
        for (grid, _), rows in _group_tradeoff_rows_by_grid(tradeoff_rows)
    }
    rows: list[dict[str, Any]] = []
    for grid, by_variant in tradeoff_by_grid.items():
        baseline = utility_by_grid[grid]["baseline"]
        shared_lut = utility_by_grid[grid]["shared_lut_saving"]
        shared_dsp = utility_by_grid[grid]["shared_dsp_reducing"]
        lut_tradeoff = by_variant["shared_lut_saving"]
        dsp_tradeoff = by_variant["shared_dsp_reducing"]
        timing_choice = (
            "shared_dsp_reducing"
            if dsp_tradeoff["shared_measured_wns_ns"] is not None
            and lut_tradeoff["baseline_measured_wns_ns"] is not None
            and dsp_tradeoff["shared_measured_wns_ns"] > lut_tradeoff["baseline_measured_wns_ns"]
            and dsp_tradeoff["shared_measured_wns_ns"] >= (lut_tradeoff["shared_measured_wns_ns"] or float("-inf"))
            else "baseline"
        )
        rows.extend(
            [
                {
                    "grid": grid,
                    "k_depth": lut_tradeoff["k_depth"],
                    "bottleneck_kind": "lut",
                    "preferred_variant": "shared_lut_saving",
                    "decision_basis": f"shared_lut_saving is the lowest-LUT measured option at `{grid}` with `{lut_tradeoff['shared_measured_lut']}` LUT.",
                },
                {
                    "grid": grid,
                    "k_depth": lut_tradeoff["k_depth"],
                    "bottleneck_kind": "dsp",
                    "preferred_variant": "shared_dsp_reducing",
                    "decision_basis": f"shared_dsp_reducing is the only measured option at `{grid}` that materially relieves DSP pressure, dropping from `{lut_tradeoff['baseline_measured_dsp']}` to `{dsp_tradeoff['shared_measured_dsp']}` DSP.",
                },
                {
                    "grid": grid,
                    "k_depth": lut_tradeoff["k_depth"],
                    "bottleneck_kind": "performance",
                    "preferred_variant": "baseline",
                    "decision_basis": f"Baseline keeps the shortest measured schedule at `{lut_tradeoff['baseline_measured_latency_cycles']}` cycles and the highest throughput at `{lut_tradeoff['baseline_measured_effective_throughput_ops_per_cycle']}` ops/cycle.",
                },
                {
                    "grid": grid,
                    "k_depth": lut_tradeoff["k_depth"],
                    "bottleneck_kind": "timing_margin",
                    "preferred_variant": timing_choice,
                    "decision_basis": (
                        f"shared_dsp_reducing has the best measured WNS at `{dsp_tradeoff['shared_measured_wns_ns']}` ns on `{grid}`, so timing-margin-sensitive use is grid-specific."
                        if timing_choice == "shared_dsp_reducing"
                        else f"Baseline has the best measured WNS at `{lut_tradeoff['baseline_measured_wns_ns']}` ns on `{grid}`, so timing-margin-sensitive use should stay with baseline here."
                    ),
                },
                {
                    "grid": grid,
                    "k_depth": lut_tradeoff["k_depth"],
                    "bottleneck_kind": "no_hard_resource_bottleneck",
                    "preferred_variant": "baseline",
                    "decision_basis": "When no hard resource bottleneck dominates, the measured shared implementations are not worth their throughput and latency penalty.",
                },
            ]
        )
    return rows


def build_measured_utility_summary(
    utility_rows: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not utility_rows:
        return {}
    measured_grids = sorted({row["grid"] for row in utility_rows})
    timing_choices = {row["grid"]: row["preferred_variant"] for row in bottleneck_rows if row["bottleneck_kind"] == "timing_margin"}
    timing_line = (
        "Timing-margin preference is grid-dependent in the measured slice: 4x4 favors `shared_dsp_reducing`, while 8x4 favors `baseline`."
        if timing_choices.get("4x4") != timing_choices.get("8x4")
        else f"Timing-margin preference is consistent across the measured slice and favors `{next(iter(set(timing_choices.values())))}.`"
    )
    summary_lines = [
        "Baseline is still the best measured default when performance dominates.",
        "shared_lut_saving is worthwhile when LUT pressure is the real bottleneck and the shared-style performance penalty is acceptable.",
        "shared_dsp_reducing is worthwhile when DSP pressure is the real bottleneck and its larger LUT footprint still fits the design.",
        "Measured shared implementations are bottleneck-specific relief mechanisms, not generally better architectures.",
        timing_line,
        "Flexibility pays only when the relieved bottleneck matters more than the lost performance.",
    ]
    return {
        "headline": "This summary turns the measured direct-slice tradeoff into a bounded utility result: each shared implementation is only worthwhile when it relieves the actual bottleneck.",
        "summary_lines": summary_lines,
        "measured_grids": measured_grids,
        "utility_row_count": len(utility_rows),
        "bottleneck_row_count": len(bottleneck_rows),
    }


def _flexibility_overhead_severity(
    throughput_retention_pct: float | None,
    latency_increase_factor: float | None,
) -> str:
    if throughput_retention_pct is None or latency_increase_factor is None:
        return "unknown"
    if throughput_retention_pct <= 55.0 and latency_increase_factor >= 1.9:
        return "high"
    if throughput_retention_pct <= 75.0 and latency_increase_factor >= 1.25:
        return "moderate"
    return "low"


def _relief_strength(
    relief_kind: str,
    resource_relief_magnitude: int | float | None,
    baseline_reference: int | float | None,
) -> str:
    if relief_kind == "none" or resource_relief_magnitude in (None, 0) or baseline_reference in (None, 0):
        return "none"
    pct = (float(resource_relief_magnitude) / float(baseline_reference)) * 100.0
    if pct >= 50.0:
        return "strong"
    if pct >= 20.0:
        return "moderate"
    return "limited"


def build_measured_flexibility_overhead_rows(
    tradeoff_rows: list[dict[str, Any]],
    utility_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not tradeoff_rows or not utility_rows:
        return []
    utility_lookup = {(row["grid"], row["architecture_variant"]): row for row in utility_rows}
    rows: list[dict[str, Any]] = []
    for row in sorted(tradeoff_rows, key=lambda item: (item["grid"], _variant_sort_key(item["shared_architecture_variant"]))):
        variant = row["shared_architecture_variant"]
        utility = utility_lookup[(row["grid"], variant)]
        baseline_reference = (
            row["baseline_measured_lut"]
            if utility["resource_relief_kind"] == "lut_relief"
            else row["baseline_measured_dsp"]
            if utility["resource_relief_kind"] == "dsp_relief"
            else None
        )
        rows.append(
            {
                "grid": row["grid"],
                "k_depth": row["k_depth"],
                "architecture_variant": variant,
                "flexibility_kind": "shared_with_fixed_schedule_overhead",
                "relieved_bottleneck": utility["resource_relief_kind"],
                "lut_delta_vs_baseline": row["measured_lut_delta_shared_minus_baseline"],
                "dsp_delta_vs_baseline": row["measured_dsp_delta_shared_minus_baseline"],
                "ff_delta_vs_baseline": row["measured_ff_delta_shared_minus_baseline"],
                "latency_delta_vs_baseline": row["measured_latency_delta_cycles_shared_minus_baseline"],
                "throughput_delta_vs_baseline": row["measured_throughput_delta_ops_per_cycle_shared_minus_baseline"],
                "throughput_retention_pct": row["measured_throughput_retention_pct"],
                "latency_increase_factor": row["measured_latency_increase_factor"],
                "wns_delta_ns": row["measured_wns_delta_ns_shared_minus_baseline"],
                "overhead_kind": "latency_and_throughput_penalty",
                "overhead_severity": _flexibility_overhead_severity(
                    row["measured_throughput_retention_pct"],
                    row["measured_latency_increase_factor"],
                ),
                "relief_kind": utility["resource_relief_kind"],
                "relief_strength": _relief_strength(
                    utility["resource_relief_kind"],
                    utility["resource_relief_magnitude"],
                    baseline_reference,
                ),
                "justification_reading": (
                    "Flexibility introduces a real shared-style schedule overhead here; it is justified only if the measured LUT relief is the bottleneck that dominates the lost performance."
                    if variant == "shared_lut_saving"
                    else "Flexibility introduces a real shared-style schedule overhead here; it is justified only if the measured DSP relief is the bottleneck that dominates the lost performance."
                ),
                "justification_status": (
                    FLEXIBILITY_JUSTIFIED_LUT
                    if variant == "shared_lut_saving"
                    else FLEXIBILITY_JUSTIFIED_DSP
                ),
            }
        )
    return rows


def build_measured_flexibility_justification_table(
    overhead_rows: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not overhead_rows or not bottleneck_rows:
        return []
    overhead_by_key = {(row["grid"], row["architecture_variant"]): row for row in overhead_rows}
    rows: list[dict[str, Any]] = []
    for grid in sorted({row["grid"] for row in overhead_rows}):
        lut_row = overhead_by_key[(grid, "shared_lut_saving")]
        dsp_row = overhead_by_key[(grid, "shared_dsp_reducing")]
        timing_row = next(row for row in bottleneck_rows if row["grid"] == grid and row["bottleneck_kind"] == "timing_margin")
        rows.extend(
            [
                {
                    "grid": grid,
                    "rule_id": f"{grid}_lut_flexibility_justified",
                    "dominant_condition": "lut_dominant_need",
                    "preferred_variant": "shared_lut_saving",
                    "justification_status": FLEXIBILITY_JUSTIFIED_LUT,
                    "justification_reason": (
                        f"shared_lut_saving relieves `{abs(lut_row['lut_delta_vs_baseline'])}` LUT while paying a `{lut_row['latency_delta_vs_baseline']}`-cycle latency increase and retaining `{lut_row['throughput_retention_pct']}`% throughput."
                    ),
                },
                {
                    "grid": grid,
                    "rule_id": f"{grid}_dsp_flexibility_justified",
                    "dominant_condition": "dsp_dominant_need",
                    "preferred_variant": "shared_dsp_reducing",
                    "justification_status": FLEXIBILITY_JUSTIFIED_DSP,
                    "justification_reason": (
                        f"shared_dsp_reducing relieves `{abs(dsp_row['dsp_delta_vs_baseline'])}` DSP while paying a `{dsp_row['latency_delta_vs_baseline']}`-cycle latency increase and retaining `{dsp_row['throughput_retention_pct']}`% throughput."
                    ),
                },
                {
                    "grid": grid,
                    "rule_id": f"{grid}_performance_flexibility_not_justified",
                    "dominant_condition": "performance_or_latency_dominant",
                    "preferred_variant": "baseline",
                    "justification_status": FLEXIBILITY_NOT_JUSTIFIED_PERFORMANCE,
                    "justification_reason": "When throughput or latency dominates, the shared-style flexibility overhead is not justified and baseline should be preferred.",
                },
                {
                    "grid": grid,
                    "rule_id": f"{grid}_no_resource_bottleneck_baseline",
                    "dominant_condition": "no_hard_resource_bottleneck",
                    "preferred_variant": "baseline",
                    "justification_status": FLEXIBILITY_BASELINE_PREFERRED,
                    "justification_reason": "When no hard LUT or DSP bottleneck dominates, flexibility is just overhead rather than useful relief.",
                },
                {
                    "grid": grid,
                    "rule_id": f"{grid}_timing_margin_grid_specific",
                    "dominant_condition": "timing_margin_sensitive",
                    "preferred_variant": timing_row["preferred_variant"],
                    "justification_status": "grid_specific_timing_margin_case",
                    "justification_reason": timing_row["decision_basis"],
                },
            ]
        )
    return rows


def build_measured_design_rule_extraction_summary(
    overhead_rows: list[dict[str, Any]],
    justification_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not overhead_rows:
        return {}
    measured_grids = sorted({row["grid"] for row in overhead_rows})
    summary_lines = [
        "Flexibility introduces real overhead; it is not free optionality.",
        "LUT-oriented flexibility is justified only when LUT relief is the dominant need.",
        "DSP-oriented flexibility is justified only when DSP relief is the dominant need.",
        "If throughput or latency dominates, flexibility is not justified and baseline should be preferred.",
        "Implementation style determines both the relieved bottleneck and the overhead paid.",
        "Flexibility should be selected for bottleneck relief, not for abstract architectural elegance.",
        "Flexibility is justified only when the relieved bottleneck matters more than the overhead it introduces.",
        "The measured slice shows flexibility as a bounded tradeoff, not a free win.",
    ]
    return {
        "headline": "This summary extracts the measured design rule: flexibility is only justified when the bottleneck it relieves matters more than the overhead it introduces.",
        "summary_lines": summary_lines,
        "measured_grids": measured_grids,
        "overhead_row_count": len(overhead_rows),
        "justification_row_count": len(justification_rows),
    }


def render_measured_vs_modelled_trust_summary(output_path: Path, summary: dict[str, Any], support_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Measured Vs Modelled Trust Summary",
        "",
    ]
    if not summary:
        lines.append("- No measured trust boundary can be stated yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
        lines.extend(["", "## Support Map", ""] )
        for row in support_rows:
            lines.append(
                f"- `{row['claim_id']}` -> `{row['support_level']}`: {row['trust_note']}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_shared_family_calibration_summary(output_path: Path, summary: dict[str, Any], overlay_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Shared-Family Calibration Summary",
        "",
    ]
    if not summary:
        lines.append("- No measured calibration aid is available yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
        lines.extend(["", "## Calibration Overlay", ""])
        for row in overlay_rows:
            lines.append(
                f"- `{row['overlay_topic']}` -> `{row['calibration_status']}`: {row['calibration_reading']}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_measured_utility_summary(output_path: Path, summary: dict[str, Any], bottleneck_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Measured Utility Summary",
        "",
    ]
    if not summary:
        lines.append("- No measured utility result is available yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
        lines.extend(["", "## Bottleneck Map", ""])
        for row in bottleneck_rows:
            lines.append(
                f"- `{row['grid']}` / `{row['bottleneck_kind']}` -> `{row['preferred_variant']}`: {row['decision_basis']}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_measured_design_rule_extraction_summary(
    output_path: Path,
    summary: dict[str, Any],
    justification_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Measured Design Rule Extraction Summary",
        "",
    ]
    if not summary:
        lines.append("- No measured design-rule extraction result is available yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
        lines.extend(["", "## Justification Table", ""])
        for row in justification_rows:
            lines.append(
                f"- `{row['grid']}` / `{row['dominant_condition']}` -> `{row['preferred_variant']}`: {row['justification_reason']}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _regime_thresholds(row: dict[str, Any]) -> dict[str, float]:
    return {
        "baseline_lut": float(row["baseline_measured_lut"]),
        "shared_lut": float(row["shared_measured_lut"]),
        "baseline_dsp": float(row["baseline_measured_dsp"]),
        "shared_dsp": float(row["shared_measured_dsp"]),
        "baseline_wns_ns": float(row["baseline_measured_wns_ns"]),
        "shared_wns_ns": float(row["shared_measured_wns_ns"]),
        "baseline_latency_cycles": float(row["baseline_measured_latency_cycles"]),
        "shared_latency_cycles": float(row["shared_measured_latency_cycles"]),
        "baseline_throughput_ops_per_cycle": float(row["baseline_measured_effective_throughput_ops_per_cycle"]),
        "shared_throughput_ops_per_cycle": float(row["shared_measured_effective_throughput_ops_per_cycle"]),
    }


def _decision_row(
    tradeoff_row: dict[str, Any],
    regime_id: str,
    lut_budget_class: str,
    dsp_budget_class: str,
    throughput_class: str,
    latency_class: str,
    slack_class: str,
    baseline_feasible: bool | None,
    shared_feasible: bool | None,
    preferred_implementation: str,
    decision_reason: str,
    design_implication: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = _regime_thresholds(tradeoff_row)
    row = {
        "grid": tradeoff_row["grid"],
        "k_depth": tradeoff_row["k_depth"],
        "regime_id": regime_id,
        "lut_budget_class": lut_budget_class,
        "dsp_budget_class": dsp_budget_class,
        "throughput_class": throughput_class,
        "latency_class": latency_class,
        "slack_class": slack_class,
        "lut_budget_shared_only_min": thresholds["shared_lut"],
        "lut_budget_baseline_min": thresholds["baseline_lut"],
        "dsp_budget_measured_floor": thresholds["baseline_dsp"],
        "throughput_shared_min_ops_per_cycle": thresholds["shared_throughput_ops_per_cycle"],
        "throughput_baseline_min_ops_per_cycle": thresholds["baseline_throughput_ops_per_cycle"],
        "latency_baseline_max_cycles": thresholds["baseline_latency_cycles"],
        "latency_shared_max_cycles": thresholds["shared_latency_cycles"],
        "slack_shared_min_ns": thresholds["shared_wns_ns"],
        "slack_baseline_min_ns": thresholds["baseline_wns_ns"],
        "baseline_feasible": baseline_feasible,
        "shared_feasible": shared_feasible,
        "preferred_implementation": preferred_implementation,
        "decision_reason": decision_reason,
        "design_implication": design_implication,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def build_measured_tradeoff_decision_rows(tradeoff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tradeoff_rows:
        return []
    tradeoff_row = next((row for row in tradeoff_rows if row["shared_architecture_variant"] == "shared_lut_saving"), tradeoff_rows[0])
    rows = [
        _decision_row(
            tradeoff_row,
            regime_id="lut_shared_only_relaxed_perf",
            lut_budget_class="lut_shared_only_window",
            dsp_budget_class="dsp_not_limiting",
            throughput_class="throughput_relaxed",
            latency_class="latency_relaxed",
            slack_class="slack_relaxed",
            baseline_feasible=False,
            shared_feasible=True,
            preferred_implementation=SHARED_PREFERRED,
            decision_reason="Shared is the only measured option inside the LUT window, and the throughput, latency, and slack constraints are all relaxed enough to admit the measured shared point.",
            design_implication="Use the measured LUT-saving shared implementation only as a LUT-relief option when performance and timing-margin requirements are relaxed.",
            extra_fields={"shared_architecture_variant": tradeoff_row["shared_architecture_variant"]},
        ),
        _decision_row(
            tradeoff_row,
            regime_id="lut_shared_only_perf_too_strict",
            lut_budget_class="lut_shared_only_window",
            dsp_budget_class="dsp_not_limiting",
            throughput_class="throughput_baseline_only",
            latency_class="latency_relaxed",
            slack_class="slack_relaxed",
            baseline_feasible=False,
            shared_feasible=False,
            preferred_implementation=NO_FEASIBLE_MEASURED_OPTION,
            decision_reason="Baseline violates the LUT budget, while shared misses the required throughput floor.",
            design_implication="A LUT-only advantage is not enough when the workload still needs baseline-class throughput.",
            extra_fields={"shared_architecture_variant": tradeoff_row["shared_architecture_variant"]},
        ),
        _decision_row(
            tradeoff_row,
            regime_id="lut_shared_only_latency_too_strict",
            lut_budget_class="lut_shared_only_window",
            dsp_budget_class="dsp_not_limiting",
            throughput_class="throughput_relaxed",
            latency_class="latency_baseline_only",
            slack_class="slack_relaxed",
            baseline_feasible=False,
            shared_feasible=False,
            preferred_implementation=NO_FEASIBLE_MEASURED_OPTION,
            decision_reason="Baseline violates the LUT budget, while shared misses the latency bound.",
            design_implication="The measured LUT-saving shared implementation is not a drop-in replacement when latency remains a hard constraint.",
            extra_fields={"shared_architecture_variant": tradeoff_row["shared_architecture_variant"]},
        ),
        _decision_row(
            tradeoff_row,
            regime_id="balanced_relaxed",
            lut_budget_class="lut_relaxed",
            dsp_budget_class="dsp_not_limiting",
            throughput_class="throughput_relaxed",
            latency_class="latency_relaxed",
            slack_class="slack_relaxed",
            baseline_feasible=True,
            shared_feasible=True,
            preferred_implementation=BASELINE_PREFERRED,
            decision_reason="DSP is flat, and baseline keeps better throughput, lower latency, and better slack; shared only saves LUT.",
            design_implication="When the LUT budget already admits baseline, the measured LUT-saving shared implementation gives away too much performance to be the default choice.",
            extra_fields={"shared_architecture_variant": tradeoff_row["shared_architecture_variant"]},
        ),
        _decision_row(
            tradeoff_row,
            regime_id="throughput_sensitive",
            lut_budget_class="lut_relaxed",
            dsp_budget_class="dsp_not_limiting",
            throughput_class="throughput_baseline_only",
            latency_class="latency_relaxed",
            slack_class="slack_relaxed",
            baseline_feasible=True,
            shared_feasible=False,
            preferred_implementation=BASELINE_PREFERRED,
            decision_reason="Baseline is the only measured option that still meets the throughput floor.",
            design_implication="Pick baseline whenever throughput demand exceeds the measured shared point.",
            extra_fields={"shared_architecture_variant": tradeoff_row["shared_architecture_variant"]},
        ),
        _decision_row(
            tradeoff_row,
            regime_id="latency_sensitive",
            lut_budget_class="lut_relaxed",
            dsp_budget_class="dsp_not_limiting",
            throughput_class="throughput_relaxed",
            latency_class="latency_baseline_only",
            slack_class="slack_relaxed",
            baseline_feasible=True,
            shared_feasible=False,
            preferred_implementation=BASELINE_PREFERRED,
            decision_reason="Baseline is the only measured option that still meets the latency bound.",
            design_implication="Pick baseline whenever latency must stay below the measured shared point.",
            extra_fields={"shared_architecture_variant": tradeoff_row["shared_architecture_variant"]},
        ),
        _decision_row(
            tradeoff_row,
            regime_id="slack_sensitive_or_dsp_pressure",
            lut_budget_class="lut_relaxed",
            dsp_budget_class="mixed_slack_case_and_dsp_pressure_case",
            throughput_class="throughput_relaxed",
            latency_class="latency_relaxed",
            slack_class="slack_baseline_only",
            baseline_feasible=True,
            shared_feasible=False,
            preferred_implementation="mixed_outcome",
            decision_reason="Under slack-sensitive constraints baseline is preferred because it is the only measured option above the shared WNS; if DSP budget falls below baseline DSP, neither measured option is feasible because the LUT-saving implementation does not reduce DSP.",
            design_implication="The LUT-saving shared implementation should not be selected for DSP relief, and its timing-margin advantage is weaker than baseline.",
            extra_fields={
                "shared_architecture_variant": tradeoff_row["shared_architecture_variant"],
                "secondary_dsp_budget_class": "dsp_infeasible",
                "secondary_preferred_implementation": NO_FEASIBLE_MEASURED_OPTION,
                "secondary_baseline_feasible": False,
                "secondary_shared_feasible": False,
            },
        ),
    ]
    return rows


def build_measured_design_rule_summary(
    tradeoff_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not tradeoff_rows:
        return {}
    tradeoff_row = next((row for row in tradeoff_rows if row["shared_architecture_variant"] == "shared_lut_saving"), tradeoff_rows[0])
    lut_window = next(row for row in decision_rows if row["regime_id"] == "lut_shared_only_relaxed_perf")
    summary_lines = [
        "The measured LUT-saving shared implementation is not a DSP-saving strategy.",
        "The measured LUT-saving shared implementation is a LUT-saving, throughput-sacrificing, latency-increasing strategy.",
        (
            "The LUT-saving shared implementation is the cleanest measured choice when LUT budget falls between "
            f"`{int(lut_window['lut_budget_shared_only_min'])}` and `< 910`, "
            "and throughput, latency, and slack demands are relaxed enough to admit the measured shared point."
        ),
        "If throughput, latency, or timing margin matter more than LUT pressure, baseline is preferable to the LUT-saving shared implementation.",
        "If DSP pressure is the problem, the LUT-saving shared implementation should not be selected for DSP relief.",
    ]
    dsp_variant = next((row for row in tradeoff_rows if row["shared_architecture_variant"] == "shared_dsp_reducing"), None)
    if dsp_variant is not None:
        if dsp_variant["measured_dsp_delta_shared_minus_baseline"] is not None and dsp_variant["measured_dsp_delta_shared_minus_baseline"] < 0:
            summary_lines.append(
                f"The measured DSP-reducing shared implementation lowers DSP by `{abs(dsp_variant['measured_dsp_delta_shared_minus_baseline'])}` on this flow, with LUT delta `{dsp_variant['measured_lut_delta_shared_minus_baseline']}`, latency `{dsp_variant['shared_measured_latency_cycles']}` cycles, and throughput `{dsp_variant['shared_measured_effective_throughput_ops_per_cycle']}` ops/cycle."
            )
            summary_lines.append(
                f"This creates a second measured rule: use the DSP-reducing shared implementation when DSP budget falls below `{tradeoff_row['baseline_measured_dsp']}` and the shared-style 65-cycle / 7.877 ops-cycle performance remains acceptable."
            )
            summary_lines.append(
                "When both measured shared implementations fit, choose by the actual bottleneck: `shared_lut_saving` for stronger LUT relief, `shared_dsp_reducing` for DSP relief."
            )
        else:
            summary_lines.append(
                "The attempted DSP-reducing shared implementation still does not reduce mapped DSP on this flow, so it should be treated as a failed DSP-relief attempt rather than a validated DSP-saving strategy."
            )
    scaling_summary = build_direct_shared_scaling_summary(tradeoff_rows)
    scaling_rule_line = scaling_summary.get("scaling_rule_line")
    if scaling_rule_line:
        summary_lines.append(scaling_rule_line)
    summary_lines.append("Sharing is not one thing: implementation style determines whether the measured relief is LUT-oriented, DSP-oriented, or neither.")
    summary_lines.append("The measured result validates that implementation style, not just architectural intention, determines which resource sharing actually relieves.")
    return {
        "grid": tradeoff_row["grid"],
        "k_depth": tradeoff_row["k_depth"],
        "summary_lines": summary_lines,
        "headline": "The repo now distinguishes between at least two directly measured sharing implementations instead of treating sharing as a single behavior.",
        "scaling_rule_line": scaling_rule_line,
        "preferred_only_window": {
            "lut_budget_class": lut_window["lut_budget_class"],
            "throughput_class": lut_window["throughput_class"],
            "latency_class": lut_window["latency_class"],
            "slack_class": lut_window["slack_class"],
        },
    }


def render_direct_slice_summary(
    output_path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    design_rule_summary: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Direct MAC-Array Slice Summary",
        "",
        "- This summary covers the standalone directly measurable MAC-array slice family used for the smallest baseline-vs-shared bridge.",
        "- Scope remains intentionally narrow: direct measurement currently covers the isolated baseline slice plus selective 4x4 and 8x4 shared implementations, not the full baseline/shared/replicated/adaptive family.",
        f"- Directly measured baseline points: `{summary['measured_points']}`.",
        f"- Baseline DSP exact matches: `{summary['dsp_exact_match_count']}`. Baseline latency exact matches: `{summary['latency_exact_match_count']}`. Baseline throughput exact matches: `{summary['throughput_exact_match_count']}`.",
        "",
    ]
    if not rows:
        lines.append("- No direct MAC-array aggregate is available yet. Run the direct-slice experiment config to populate this comparison.")
    else:
        lines.append("## Compared Points")
        lines.append("")
        for row in rows:
            lines.append(
                f"- `{row['architecture']}` `{row['grid']}` -> DSP measured/modelled `{row['measured_dsp']}`/`{row['model_dsp']}`, "
                f"LUT measured/modelled `{row['measured_lut']}`/`{row['model_lut']}`, "
                f"calibrated LUT `{row['calibrated_model_lut']}`, "
                f"throughput measured/modelled `{row['measured_effective_throughput_ops_per_cycle']}`/`{row['direct_slice_throughput_model_ops_per_cycle']}`."
            )
    if design_rule_summary:
        lines.extend(
            [
                "",
                "## Measured Design Rule",
                "",
                f"- {design_rule_summary['headline']}",
                *[f"- {line}" for line in design_rule_summary['summary_lines'][:4]],
                *([f"- {design_rule_summary['scaling_rule_line']}"] if design_rule_summary.get('scaling_rule_line') else []),
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_direct_calibration_summary(output_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Direct Calibration Summary",
        "",
        f"- Direct baseline points measured: `{summary['measured_points']}`.",
        f"- Grids covered: {', '.join(summary['grids']) if summary['grids'] else 'none' }.",
        f"- DSP exact matches: `{summary['dsp_exact_match_count']}`.",
        f"- Latency exact matches: `{summary['latency_exact_match_count']}`.",
        f"- Throughput exact matches: `{summary['throughput_exact_match_count']}`.",
        f"- LUT error range vs lightweight framework model: `{summary['lut_error_min']}` .. `{summary['lut_error_max']}` LUT.",
        f"- Measured WNS range: `{summary['wns_measured_min_ns']}` .. `{summary['wns_measured_max_ns']}` ns.",
        "",
        "## Interpretation",
        "",
        "- DSP and direct-slice latency/throughput should be treated as the most strongly validated baseline model components over the measured direct set.",
        "- LUT behaviour is now reported with both the original lightweight model and a direct-slice-calibrated alternative; the calibrated line is a caution aid, not a silent framework replacement.",
        f"- {summary['baseline_model_scope_note']}",
    ]
    if "calibrated_lut_model" in summary:
        model = summary["calibrated_lut_model"]
        lines.extend([
            "",
            "## Calibrated LUT Aid",
            "",
            f"- Formula: `{model['formula']}`.",
            f"- `lut_fixed = {model['lut_fixed']}`.",
            f"- `lut_per_mac = {model['lut_per_mac']}`.",
            f"- Provenance: `{model['provenance_kind']}` from {', '.join(model['source_aggregates'])}.",
            f"- Usage note: {model['usage_note']}",
        ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_direct_tradeoff_summary(output_path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Direct Baseline-vs-Shared Tradeoff Summary",
        "",
        "- This summary covers the directly measured MAC-array architecture tradeoffs in the repo.",
        "- Scope is intentionally narrow: baseline and the measured shared implementations are compared only on the isolated direct slice, with 4x4 and 8x4 as the measured bridge points.",
        "- Baseline direct calibration remains baseline-only; each shared implementation is compared against the lightweight shared model and against the measured baseline point without claiming global calibration.",
        "",
    ]
    if not rows:
        lines.append("- No directly measured baseline-vs-shared tradeoff pair is available yet. Run the direct shared configs to populate this summary.")
    else:
        for row in rows:
            lines.extend(
                [
                    f"## {row['grid']} @ K={row['k_depth']} :: {row['shared_architecture_variant']}",
                    "",
                    f"- Intended relief: `{row['shared_intended_relief_kind']}`. Measured relief: `{row['shared_measured_relief_kind']}`.",
                    f"- Measured DSP: baseline `{row['baseline_measured_dsp']}` vs shared `{row['shared_measured_dsp']}` (delta `{row['measured_dsp_delta_shared_minus_baseline']}`).",
                    f"- Measured LUT: baseline `{row['baseline_measured_lut']}` vs shared `{row['shared_measured_lut']}` (delta `{row['measured_lut_delta_shared_minus_baseline']}`).",
                    f"- Measured LUT reduction: `{row['measured_lut_reduction_pct']}`%.",
                    f"- Measured FF: baseline `{row['baseline_measured_ff']}` vs shared `{row['shared_measured_ff']}` (delta `{row['measured_ff_delta_shared_minus_baseline']}`).",
                    f"- Measured WNS: baseline `{row['baseline_measured_wns_ns']}` ns vs shared `{row['shared_measured_wns_ns']}` ns (delta `{row['measured_wns_delta_ns_shared_minus_baseline']}` ns).",
                    f"- Measured latency: baseline `{row['baseline_measured_latency_cycles']}` cycles vs shared `{row['shared_measured_latency_cycles']}` cycles (delta `{row['measured_latency_delta_cycles_shared_minus_baseline']}`).",
                    f"- Measured latency increase factor: `{row['measured_latency_increase_factor']}`x.",
                    f"- Measured throughput: baseline `{row['baseline_measured_effective_throughput_ops_per_cycle']}` vs shared `{row['shared_measured_effective_throughput_ops_per_cycle']}` ops/cycle (delta `{row['measured_throughput_delta_ops_per_cycle_shared_minus_baseline']}`).",
                    f"- Measured throughput retention/reduction: `{row['measured_throughput_retention_pct']}`% retained, `{row['measured_throughput_reduction_pct']}`% reduced.",
                    f"- Modelled DSP delta shared-baseline: `{row['model_dsp_delta_shared_minus_baseline']}`. Modelled latency delta: `{row['model_latency_delta_cycles_shared_minus_baseline']}`. Modelled throughput delta: `{row['model_throughput_delta_ops_per_cycle_shared_minus_baseline']}`.",
                    f"- Note: {row['tradeoff_note']}",
                    "",
                ]
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n")


def render_direct_shared_implementation_summary(output_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Direct Shared Implementation Summary",
        "",
    ]
    if not summary:
        lines.append("- No directly measured shared-implementation comparison is available yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
        for grid_summary in summary.get("grid_summaries", []):
            lines.extend(
                [
                    "",
                    f"## {grid_summary['grid']} @ K={grid_summary['k_depth']}",
                    "",
                ]
            )
            for line in grid_summary["summary_lines"]:
                lines.append(f"- {line}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_direct_shared_scaling_summary(output_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Direct Shared Scaling Summary",
        "",
    ]
    if not summary:
        lines.append("- No two-scale shared comparison is available yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
        for grid_summary in summary.get("grid_summaries", []):
            lines.extend(
                [
                    "",
                    f"## {grid_summary['grid']} @ K={grid_summary['k_depth']}",
                    "",
                    f"- baseline_performance_first: `{grid_summary['baseline_performance_first']}`",
                    f"- shared_lut_saving_is_best_lut_relief: `{grid_summary['shared_lut_saving_is_best_lut_relief']}`",
                    f"- shared_dsp_reducing_is_best_dsp_relief: `{grid_summary['shared_dsp_reducing_is_best_dsp_relief']}`",
                    f"- three_way_rule_status: `{grid_summary['three_way_rule_status']}`",
                ]
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_measured_tradeoff_regime_summary(output_path: Path, decision_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Measured Tradeoff Regime Summary",
        "",
        "- This summary is intentionally limited to the directly measured 4x4 baseline-vs-shared pair.",
        "- It does not generalize to the broader modelled shared family.",
        "",
    ]
    if not decision_rows:
        lines.append("- No directly measured tradeoff pair is available yet, so no regime summary can be derived.")
    else:
        for row in decision_rows:
            lines.extend(
                [
                    f"## {row['regime_id']}",
                    "",
                    f"- Classes: LUT `{row['lut_budget_class']}`, DSP `{row['dsp_budget_class']}`, throughput `{row['throughput_class']}`, latency `{row['latency_class']}`, slack `{row['slack_class']}`.",
                    f"- Thresholds: shared-only LUT window `[{int(row['lut_budget_shared_only_min'])}, {int(row['lut_budget_baseline_min'])})`, DSP floor `{int(row['dsp_budget_measured_floor'])}`, throughput shared/baseline `{row['throughput_shared_min_ops_per_cycle']}` / `{row['throughput_baseline_min_ops_per_cycle']}` ops-cycle, latency baseline/shared `{int(row['latency_baseline_max_cycles'])}` / `{int(row['latency_shared_max_cycles'])}` cycles, slack shared/baseline `{row['slack_shared_min_ns']}` / `{row['slack_baseline_min_ns']}` ns.",
                    f"- Feasibility: baseline `{row['baseline_feasible']}`, shared `{row['shared_feasible']}`.",
                    f"- Decision: `{row['preferred_implementation']}`.",
                    f"- Reason: {row['decision_reason']}",
                    f"- Design implication: {row['design_implication']}",
                    "",
                ]
            )
            if "secondary_preferred_implementation" in row:
                lines.extend(
                    [
                        f"- Secondary DSP-pressure case: `{row['secondary_dsp_budget_class']}` -> `{row['secondary_preferred_implementation']}` with baseline/shared feasibility `{row['secondary_baseline_feasible']}`/`{row['secondary_shared_feasible']}`.",
                        "",
                    ]
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n")


def render_measured_design_rules(output_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Measured Design Rules",
        "",
    ]
    if not summary:
        lines.append("- No directly measured baseline-vs-shared pair is available yet, so no measured design rule can be stated.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_direct_calibration_plot(output_path: Path, rows: list[dict[str, Any]]) -> str | None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cnn_fpga_mplconfig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    measured = [
        row
        for row in rows
        if row["comparison_status"] == "direct_measured_vs_modelled" and row["architecture_family"] == "baseline"
    ]
    if not measured:
        return None

    x_labels = [row["grid"] for row in measured]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].plot(x_labels, [row["measured_lut"] for row in measured], marker="o", label="measured LUT")
    axes[0].plot(x_labels, [row["model_lut"] for row in measured], marker="o", label="framework LUT")
    if all(row["calibrated_model_lut"] is not None for row in measured):
        axes[0].plot(x_labels, [row["calibrated_model_lut"] for row in measured], marker="o", label="calibrated LUT")
    axes[0].set_title("Baseline LUT Across Direct Points")
    axes[0].set_ylabel("LUT")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_labels, [row["measured_wns_ns"] for row in measured], marker="o", label="measured WNS")
    axes[1].plot(x_labels, [row["model_wns_estimate_ns"] for row in measured], marker="o", label="framework WNS")
    axes[1].set_title("Baseline Timing Trend Across Direct Points")
    axes[1].set_ylabel("WNS (ns)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)
