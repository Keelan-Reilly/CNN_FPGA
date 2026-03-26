#!/usr/bin/env python3
"""Direct MAC-array slice comparison, calibration, and tradeoff helpers."""

from __future__ import annotations

import json
import os
from math import sqrt
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
    "study_mac_array_direct_shared_lut_8x8.json",
    "study_mac_array_direct_shared_dsp_8x8.json",
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
PREDICTOR_EXACT = "exact_validated_formula"
PREDICTOR_LINEAR = "local_linear_fit"
PREDICTOR_CONSTANT = "measured_constant_fit"
PREDICTOR_UNSTABLE = "too_unstable_for_trusted_prediction"
MEASURED_LATTICE_POINT = "measured_lattice_point"
INTERPOLATED_WITHIN_MEASURED_DOMAIN = "interpolated_within_measured_domain"
UNSUPPORTED_EXTRAPOLATION = "unsupported_extrapolation"


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


def _fit_constant(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _variant_predictor_domain(rows: list[dict[str, Any]]) -> tuple[int, int]:
    mac_units = sorted(int(row["mac_units"]) for row in rows)
    return mac_units[0], mac_units[-1]


def _predictor_confidence_status(metric: str, max_abs_residual: float, max_abs_relative_residual_pct: float | None) -> str:
    if metric in {"dsp", "latency_cycles", "effective_throughput_ops_per_cycle"} and max_abs_residual <= 1e-6:
        return "high_within_measured_domain"
    if metric in {"lut", "ff"}:
        if max_abs_relative_residual_pct is not None and max_abs_relative_residual_pct <= 2.0:
            return "moderate_local_predictor"
        if max_abs_relative_residual_pct is not None and max_abs_relative_residual_pct <= 5.0:
            return "cautionary_local_predictor"
        return "low_confidence_local_predictor"
    if metric == "wns_ns":
        if max_abs_residual <= 0.15:
            return "cautionary_local_predictor"
        return "low_confidence_local_predictor"
    return "cautionary_local_predictor"


def _build_predictor_row(
    architecture_variant: str,
    metric: str,
    predictor_kind: str,
    predictor_formula: str,
    measured_rows: list[dict[str, Any]],
    predict_fn: Any,
    fit_status: str | None = None,
    confidence_status: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    domain_min, domain_max = _variant_predictor_domain(measured_rows)
    residual_rows: list[dict[str, Any]] = []
    abs_residuals: list[float] = []
    relative_residuals: list[float] = []
    for row in measured_rows:
        measured_value = float(row[f"measured_{metric}"])
        predicted_value = float(predict_fn(row))
        residual = round(measured_value - predicted_value, 6)
        abs_residual = abs(residual)
        abs_residuals.append(abs_residual)
        rel = None
        if measured_value != 0:
            rel = round((abs_residual / abs(measured_value)) * 100.0, 6)
            relative_residuals.append(rel)
        residual_rows.append(
            {
                "architecture_variant": architecture_variant,
                "metric": metric,
                "grid": row["grid"],
                "mac_units": row["mac_units"],
                "k_depth": row["k_depth"],
                "measured_value": measured_value,
                "predicted_value": round(predicted_value, 6),
                "residual": residual,
                "abs_residual": round(abs_residual, 6),
                "abs_relative_residual_pct": rel,
            }
        )

    max_abs_residual = round(max(abs_residuals), 6) if abs_residuals else None
    rms_residual = round(sqrt(sum(value * value for value in abs_residuals) / len(abs_residuals)), 6) if abs_residuals else None
    max_abs_relative_residual_pct = round(max(relative_residuals), 6) if relative_residuals else None
    fit_status = fit_status or (PREDICTOR_EXACT if max_abs_residual == 0 else predictor_kind)
    if metric == "wns_ns" and max_abs_residual is not None and max_abs_residual > 0.15:
        fit_status = PREDICTOR_UNSTABLE
    confidence_status = confidence_status or _predictor_confidence_status(metric, max_abs_residual or 0.0, max_abs_relative_residual_pct)

    predictor_row = {
        "architecture_variant": architecture_variant,
        "metric": metric,
        "predictor_kind": predictor_kind,
        "predictor_formula": predictor_formula,
        "fit_status": fit_status,
        "confidence_status": confidence_status,
        "measured_point_count": len(measured_rows),
        "interpolation_domain_mac_units_min": domain_min,
        "interpolation_domain_mac_units_max": domain_max,
        "interpolation_domain_k_depth_min": min(int(row["k_depth"]) for row in measured_rows),
        "interpolation_domain_k_depth_max": max(int(row["k_depth"]) for row in measured_rows),
        "max_abs_residual": max_abs_residual,
        "rms_residual": rms_residual,
        "max_abs_relative_residual_pct": max_abs_relative_residual_pct,
        "extrapolation_warning": (
            f"Use only for interpolation over {domain_min} <= mac_units <= {domain_max} at the currently measured k-depth range; outside that domain treat predictions as unvalidated extrapolation."
        ),
    }
    return predictor_row, residual_rows


def build_measured_predictor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured_rows = [row for row in rows if row["comparison_status"] == "direct_measured_vs_modelled"]
    if not measured_rows:
        return []

    predictor_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in measured_rows:
        by_variant.setdefault(row["architecture"], []).append(row)

    for architecture_variant, variant_rows in sorted(by_variant.items(), key=lambda item: (0 if item[0] == "baseline" else 1 if item[0] == "shared_lut_saving" else 2, item[0])):
        variant_rows = sorted(variant_rows, key=lambda item: item["mac_units"])
        linear_metrics = {
            "lut": [float(row["measured_lut"]) for row in variant_rows if row["measured_lut"] is not None],
            "ff": [float(row["measured_ff"]) for row in variant_rows if row["measured_ff"] is not None],
            "wns_ns": [float(row["measured_wns_ns"]) for row in variant_rows if row["measured_wns_ns"] is not None],
        }
        mac_units = [float(row["mac_units"]) for row in variant_rows]

        if architecture_variant == "baseline":
            exact_specs = [
                ("dsp", PREDICTOR_LINEAR, "dsp = mac_units", lambda row: float(row["mac_units"])),
                ("latency_cycles", PREDICTOR_EXACT, "latency_cycles = k_depth + 1", lambda row: float(row["k_depth"] + 1)),
                ("effective_throughput_ops_per_cycle", PREDICTOR_EXACT, "throughput = mac_units * k_depth / (k_depth + 1)", lambda row: float(row["mac_units"] * row["k_depth"]) / float(row["k_depth"] + 1)),
            ]
        elif architecture_variant == "shared_lut_saving":
            exact_specs = [
                ("dsp", PREDICTOR_LINEAR, "dsp = mac_units", lambda row: float(row["mac_units"])),
                ("latency_cycles", PREDICTOR_EXACT, "latency_cycles = 2 * k_depth + 1", lambda row: float((2 * row["k_depth"]) + 1)),
                ("effective_throughput_ops_per_cycle", PREDICTOR_EXACT, "throughput = mac_units * k_depth / (2 * k_depth + 1)", lambda row: float(row["mac_units"] * row["k_depth"]) / float((2 * row["k_depth"]) + 1)),
            ]
        else:
            exact_specs = [
                ("dsp", PREDICTOR_CONSTANT, "dsp = 0", lambda row: 0.0),
                ("latency_cycles", PREDICTOR_EXACT, "latency_cycles = 2 * k_depth + 1", lambda row: float((2 * row["k_depth"]) + 1)),
                ("effective_throughput_ops_per_cycle", PREDICTOR_EXACT, "throughput = mac_units * k_depth / (2 * k_depth + 1)", lambda row: float(row["mac_units"] * row["k_depth"]) / float((2 * row["k_depth"]) + 1)),
            ]
        for metric, predictor_kind, formula, predict_fn in exact_specs:
            predictor_row, current_residuals = _build_predictor_row(architecture_variant, metric, predictor_kind, formula, variant_rows, predict_fn)
            predictor_rows.append(predictor_row)
            residual_rows.extend(current_residuals)

        for metric, ys in linear_metrics.items():
            fit = _fit_linear(mac_units, ys)
            if fit is None:
                continue
            intercept, slope = fit
            formula = f"{metric} = {round(intercept, 6)} + {round(slope, 6)} * mac_units"
            predictor_row, current_residuals = _build_predictor_row(
                architecture_variant,
                metric,
                PREDICTOR_LINEAR,
                formula,
                variant_rows,
                lambda row, _intercept=intercept, _slope=slope: _intercept + (_slope * float(row["mac_units"])),
            )
            predictor_rows.append(predictor_row)
            residual_rows.extend(current_residuals)

    predictor_rows.sort(key=lambda row: (row["architecture_variant"], row["metric"]))
    return predictor_rows


def build_measured_fit_residual_rows(rows: list[dict[str, Any]], predictor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured_rows = [row for row in rows if row["comparison_status"] == "direct_measured_vs_modelled"]
    if not measured_rows or not predictor_rows:
        return []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in measured_rows:
        by_variant.setdefault(row["architecture"], []).append(row)

    residual_rows: list[dict[str, Any]] = []
    for predictor_row in predictor_rows:
        architecture_variant = predictor_row["architecture_variant"]
        metric = predictor_row["metric"]
        variant_rows = sorted(by_variant[architecture_variant], key=lambda item: item["mac_units"])
        formula = predictor_row["predictor_formula"]
        if formula == "dsp = mac_units":
            predict_fn = lambda row: float(row["mac_units"])
        elif formula == "dsp = 0":
            predict_fn = lambda row: 0.0
        elif formula == "latency_cycles = k_depth + 1":
            predict_fn = lambda row: float(row["k_depth"] + 1)
        elif formula == "latency_cycles = 2 * k_depth + 1":
            predict_fn = lambda row: float((2 * row["k_depth"]) + 1)
        elif formula == "throughput = mac_units * k_depth / (k_depth + 1)":
            predict_fn = lambda row: float(row["mac_units"] * row["k_depth"]) / float(row["k_depth"] + 1)
        elif formula == "throughput = mac_units * k_depth / (2 * k_depth + 1)":
            predict_fn = lambda row: float(row["mac_units"] * row["k_depth"]) / float((2 * row["k_depth"]) + 1)
        else:
            _, rhs = formula.split("=", 1)
            intercept_str, slope_part = rhs.strip().split("+", 1)
            slope_str = slope_part.strip().split("*", 1)[0].strip()
            intercept = float(intercept_str.strip())
            slope = float(slope_str)
            predict_fn = lambda row, _intercept=intercept, _slope=slope: _intercept + (_slope * float(row["mac_units"]))
        _, current_residuals = _build_predictor_row(
            architecture_variant,
            metric,
            predictor_row["predictor_kind"],
            predictor_row["predictor_formula"],
            variant_rows,
            predict_fn,
            fit_status=predictor_row["fit_status"],
            confidence_status=predictor_row["confidence_status"],
        )
        residual_rows.extend(current_residuals)
    residual_rows.sort(key=lambda row: (row["architecture_variant"], row["metric"], row["mac_units"]))
    return residual_rows


def build_measured_predictor_summary(predictor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictor_rows:
        return {}
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in predictor_rows:
        by_metric.setdefault(row["metric"], []).append(row)
    exact_metrics = sorted(
        metric for metric, rows in by_metric.items() if all(item["fit_status"] == PREDICTOR_EXACT or (metric == "dsp" and item["max_abs_residual"] == 0) for item in rows)
    )
    unstable_metrics = sorted(
        metric for metric, rows in by_metric.items() if any(item["fit_status"] == PREDICTOR_UNSTABLE for item in rows)
    )
    domain_min = min(row["interpolation_domain_mac_units_min"] for row in predictor_rows)
    domain_max = max(row["interpolation_domain_mac_units_max"] for row in predictor_rows)
    summary_lines = [
        f"Within the measured direct-slice domain {domain_min} <= mac_units <= {domain_max} at k_depth=32, DSP, latency, and throughput are well fit by simple transparent local predictors.",
        "LUT and FF admit compact local linear predictors over the measured range, with residuals that should still be treated as interpolation-only aids rather than global architecture models.",
        "WNS remains the least stable metric; use the local predictor only as a cautionary interpolation aid and avoid trusting it for extrapolation.",
        "The measured direct slice shows shared flexibility overhead as approximately fixed in schedule terms: shared variants keep the 65-cycle schedule while baseline keeps the 33-cycle schedule.",
        "Within the measured domain, the direct-slice predictor can replace lightweight direct-slice expectations for local architecture-variant metrics, but it does not replace broader shared-family modelled expectations.",
    ]
    if unstable_metrics:
        summary_lines.append(f"The least trustworthy local predictor metric is: {', '.join(unstable_metrics)}.")
    return {
        "headline": "This summary turns the measured direct-slice lattice into a compact local predictor with residuals and explicit trust boundaries.",
        "summary_lines": summary_lines,
        "exact_metrics": exact_metrics,
        "unstable_metrics": unstable_metrics,
        "predictor_row_count": len(predictor_rows),
        "interpolation_domain_mac_units_min": domain_min,
        "interpolation_domain_mac_units_max": domain_max,
    }


def build_measured_extrapolation_boundary_summary(predictor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictor_rows:
        return {}
    domain_min = min(row["interpolation_domain_mac_units_min"] for row in predictor_rows)
    domain_max = max(row["interpolation_domain_mac_units_max"] for row in predictor_rows)
    return {
        "headline": "This summary marks where local measured prediction ends and extrapolation begins for the isolated direct-slice architecture family.",
        "summary_lines": [
            f"Interpolation is bounded to the measured direct-slice lattice over {domain_min} <= mac_units <= {domain_max} at k_depth=32.",
            "Predictions inside that domain are local direct-slice interpolation aids only; they should not be promoted to full shared-family architecture truths.",
            "Any use outside the measured mac-unit domain should be treated as unvalidated extrapolation with low trust, especially for WNS and implementation-dependent resource metrics.",
            "The measured local predictor is appropriate for the isolated baseline/shared_lut_saving/shared_dsp_reducing slice family only.",
        ],
        "interpolation_domain_mac_units_min": domain_min,
        "interpolation_domain_mac_units_max": domain_max,
        "supported_variants": sorted({row["architecture_variant"] for row in predictor_rows}),
    }


def _predict_from_formula(formula: str, mac_units: int, k_depth: int) -> float:
    if formula == "dsp = mac_units":
        return float(mac_units)
    if formula == "dsp = 0":
        return 0.0
    if formula == "latency_cycles = k_depth + 1":
        return float(k_depth + 1)
    if formula == "latency_cycles = 2 * k_depth + 1":
        return float((2 * k_depth) + 1)
    if formula == "throughput = mac_units * k_depth / (k_depth + 1)":
        return float(mac_units * k_depth) / float(k_depth + 1)
    if formula == "throughput = mac_units * k_depth / (2 * k_depth + 1)":
        return float(mac_units * k_depth) / float((2 * k_depth) + 1)
    _, rhs = formula.split("=", 1)
    intercept_str, slope_part = rhs.strip().split("+", 1)
    slope_str = slope_part.strip().split("*", 1)[0].strip()
    intercept = float(intercept_str.strip())
    slope = float(slope_str)
    return intercept + (slope * float(mac_units))


def _trust_status_for_mac_units(mac_units: int, domain_min: int, domain_max: int, measured_mac_units: set[int]) -> str:
    if mac_units < domain_min or mac_units > domain_max:
        return UNSUPPORTED_EXTRAPOLATION
    if mac_units in measured_mac_units:
        return MEASURED_LATTICE_POINT
    return INTERPOLATED_WITHIN_MEASURED_DOMAIN


def _predict_metric(
    predictor_lookup: dict[tuple[str, str], dict[str, Any]],
    architecture_variant: str,
    metric: str,
    mac_units: int,
    k_depth: int,
) -> float:
    row = predictor_lookup[(architecture_variant, metric)]
    return _predict_from_formula(row["predictor_formula"], mac_units, k_depth)


def build_measured_budget_boundary_rows(predictor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not predictor_rows:
        return []
    predictor_lookup = {(row["architecture_variant"], row["metric"]): row for row in predictor_rows}
    domain_min = min(row["interpolation_domain_mac_units_min"] for row in predictor_rows)
    domain_max = max(row["interpolation_domain_mac_units_max"] for row in predictor_rows)
    k_depth = min(row["interpolation_domain_k_depth_min"] for row in predictor_rows)
    measured_mac_units = sorted(
        {
            row["interpolation_domain_mac_units_min"]
            for row in predictor_rows
            if row["interpolation_domain_mac_units_min"] == row["interpolation_domain_mac_units_max"]
        }
    )
    if not measured_mac_units:
        measured_mac_units = [16, 32, 64]

    rows: list[dict[str, Any]] = []
    for mac_units in range(domain_min, domain_max + 1):
        trust_status = _trust_status_for_mac_units(mac_units, domain_min, domain_max, set(measured_mac_units))
        baseline_lut = _predict_metric(predictor_lookup, "baseline", "lut", mac_units, k_depth)
        shared_lut = _predict_metric(predictor_lookup, "shared_lut_saving", "lut", mac_units, k_depth)
        baseline_dsp = _predict_metric(predictor_lookup, "baseline", "dsp", mac_units, k_depth)
        shared_dsp = _predict_metric(predictor_lookup, "shared_dsp_reducing", "dsp", mac_units, k_depth)
        baseline_latency = _predict_metric(predictor_lookup, "baseline", "latency_cycles", mac_units, k_depth)
        shared_latency = _predict_metric(predictor_lookup, "shared_lut_saving", "latency_cycles", mac_units, k_depth)
        baseline_throughput = _predict_metric(
            predictor_lookup, "baseline", "effective_throughput_ops_per_cycle", mac_units, k_depth
        )
        shared_throughput = _predict_metric(
            predictor_lookup, "shared_lut_saving", "effective_throughput_ops_per_cycle", mac_units, k_depth
        )
        shared_dsp_lut = _predict_metric(predictor_lookup, "shared_dsp_reducing", "lut", mac_units, k_depth)
        rows.append(
            {
                "mac_units": mac_units,
                "k_depth": k_depth,
                "trust_status": trust_status,
                "lut_budget_shared_lut_min": round(shared_lut, 6),
                "lut_budget_baseline_min": round(baseline_lut, 6),
                "lut_budget_window_kind": "shared_lut_saving_justified_when_budget_is_between_shared_and_baseline",
                "dsp_budget_shared_dsp_min": round(shared_dsp, 6),
                "dsp_budget_baseline_min": round(baseline_dsp, 6),
                "dsp_budget_window_kind": "shared_dsp_reducing_justified_when_budget_is_below_baseline_dsp_floor",
                "shared_dsp_lut_floor": round(shared_dsp_lut, 6),
                "shared_throughput_floor_ops_per_cycle": round(shared_throughput, 6),
                "baseline_throughput_floor_ops_per_cycle": round(baseline_throughput, 6),
                "shared_latency_ceiling_cycles": round(shared_latency, 6),
                "baseline_latency_ceiling_cycles": round(baseline_latency, 6),
                "boundary_reading": (
                    f"At {mac_units} mac units, shared_lut_saving becomes worthwhile only if LUT budget falls below `{round(baseline_lut, 3)}` but still admits `{round(shared_lut, 3)}`, while shared_dsp_reducing only matters if DSP budget falls below `{round(baseline_dsp, 3)}` and the design can still absorb about `{round(shared_dsp_lut, 3)}` LUT."
                ),
                "extrapolation_warning": (
                    "This boundary row is safe only inside the measured direct-slice interpolation domain."
                    if trust_status != UNSUPPORTED_EXTRAPOLATION
                    else "This row is outside the measured domain and should not be used for architecture choice."
                ),
            }
        )
    for mac_units, boundary_side in ((domain_min - 1, "below_measured_domain"), (domain_max + 1, "above_measured_domain")):
        rows.append(
            {
                "mac_units": mac_units,
                "k_depth": k_depth,
                "trust_status": UNSUPPORTED_EXTRAPOLATION,
                "boundary_side": boundary_side,
                "boundary_reading": "Outside the measured direct-slice domain, the repo should refuse to claim a predictor-backed choice boundary.",
                "extrapolation_warning": "Unsupported extrapolation beyond the validated direct-slice predictor domain.",
            }
        )
    return rows


def build_measured_decision_surface(
    predictor_rows: list[dict[str, Any]],
    utility_rows: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not predictor_rows or not utility_rows or not bottleneck_rows:
        return []
    predictor_lookup = {(row["architecture_variant"], row["metric"]): row for row in predictor_rows}
    domain_min = min(row["interpolation_domain_mac_units_min"] for row in predictor_rows)
    domain_max = max(row["interpolation_domain_mac_units_max"] for row in predictor_rows)
    k_depth = min(row["interpolation_domain_k_depth_min"] for row in predictor_rows)
    measured_mac_units = {16, 32, 64}

    rows: list[dict[str, Any]] = []
    for mac_units in range(domain_min, domain_max + 1):
        trust_status = _trust_status_for_mac_units(mac_units, domain_min, domain_max, measured_mac_units)
        baseline_lut = _predict_metric(predictor_lookup, "baseline", "lut", mac_units, k_depth)
        shared_lut = _predict_metric(predictor_lookup, "shared_lut_saving", "lut", mac_units, k_depth)
        baseline_dsp = _predict_metric(predictor_lookup, "baseline", "dsp", mac_units, k_depth)
        shared_dsp = _predict_metric(predictor_lookup, "shared_dsp_reducing", "dsp", mac_units, k_depth)
        baseline_latency = _predict_metric(predictor_lookup, "baseline", "latency_cycles", mac_units, k_depth)
        shared_latency = _predict_metric(predictor_lookup, "shared_lut_saving", "latency_cycles", mac_units, k_depth)
        baseline_throughput = _predict_metric(
            predictor_lookup, "baseline", "effective_throughput_ops_per_cycle", mac_units, k_depth
        )
        shared_throughput = _predict_metric(
            predictor_lookup, "shared_lut_saving", "effective_throughput_ops_per_cycle", mac_units, k_depth
        )
        shared_dsp_lut = _predict_metric(predictor_lookup, "shared_dsp_reducing", "lut", mac_units, k_depth)
        rows.extend(
            [
                {
                    "mac_units": mac_units,
                    "k_depth": k_depth,
                    "regime_id": "lut_budget_tight_relaxed_performance",
                    "trust_status": trust_status,
                    "preferred_variant": "shared_lut_saving",
                    "decision_status": "shared_worth_overhead",
                    "selection_condition": (
                        f"LUT budget in [`{round(shared_lut, 3)}`, `{round(baseline_lut, 3)}`), DSP budget >= `{round(baseline_dsp, 3)}`, min throughput <= `{round(shared_throughput, 6)}`, max latency >= `{round(shared_latency, 3)}`."
                    ),
                    "why_preferred": "The LUT-oriented shared implementation is the lowest-LUT measured option and is only worthwhile when that LUT relief matters more than the fixed shared schedule penalty.",
                    "why_not_others": "Baseline violates the LUT budget in this window; shared_dsp_reducing spends more LUT than shared_lut_saving while offering the wrong relief kind.",
                },
                {
                    "mac_units": mac_units,
                    "k_depth": k_depth,
                    "regime_id": "dsp_budget_tight_relaxed_performance",
                    "trust_status": trust_status,
                    "preferred_variant": "shared_dsp_reducing",
                    "decision_status": "shared_worth_overhead",
                    "selection_condition": (
                        f"DSP budget in [`{round(shared_dsp, 3)}`, `{round(baseline_dsp, 3)}`), LUT budget >= `{round(shared_dsp_lut, 3)}`, min throughput <= `{round(shared_throughput, 6)}`, max latency >= `{round(shared_latency, 3)}`."
                    ),
                    "why_preferred": "The DSP-oriented shared implementation is the only measured option that materially relieves DSP pressure inside the validated direct-slice family.",
                    "why_not_others": "Baseline and shared_lut_saving both retain the baseline DSP floor, so they do not solve DSP pressure.",
                },
                {
                    "mac_units": mac_units,
                    "k_depth": k_depth,
                    "regime_id": "performance_or_latency_dominant",
                    "trust_status": trust_status,
                    "preferred_variant": "baseline",
                    "decision_status": "baseline_preferred",
                    "selection_condition": (
                        f"Min throughput > `{round(shared_throughput, 6)}` or max latency < `{round(shared_latency, 3)}`."
                    ),
                    "why_preferred": "Baseline keeps the shortest schedule and highest throughput, so the shared overhead is not worth paying when performance dominates.",
                    "why_not_others": "Both shared variants give away roughly half the throughput and move to the 65-cycle schedule.",
                },
                {
                    "mac_units": mac_units,
                    "k_depth": k_depth,
                    "regime_id": "no_hard_resource_bottleneck",
                    "trust_status": trust_status,
                    "preferred_variant": "baseline",
                    "decision_status": "no_shared_option_worth_overhead",
                    "selection_condition": (
                        f"LUT budget >= `{round(baseline_lut, 3)}` and DSP budget >= `{round(baseline_dsp, 3)}`."
                    ),
                    "why_preferred": "When baseline already fits, the measured shared variants become bottleneck-specific overhead rather than a general improvement.",
                    "why_not_others": "Shared variants only make sense when their relieved bottleneck dominates the lost performance.",
                },
                {
                    "mac_units": mac_units,
                    "k_depth": k_depth,
                    "regime_id": "timing_margin_sensitive",
                    "trust_status": trust_status,
                    "preferred_variant": "unsupported_within_surface",
                    "decision_status": "unsupported_due_to_wns_instability",
                    "selection_condition": "Timing-margin-sensitive choice is intentionally not interpolated by this surface.",
                    "why_preferred": "The direct-slice predictor marks WNS as too unstable for trusted decision interpolation, so timing-sensitive choice remains a measured-grid-specific lookup.",
                    "why_not_others": "Promoting a smooth timing boundary would over-claim beyond the measured WNS evidence.",
                },
            ]
        )
    for mac_units, boundary_side in ((domain_min - 1, "below_measured_domain"), (domain_max + 1, "above_measured_domain")):
        rows.append(
            {
                "mac_units": mac_units,
                "k_depth": k_depth,
                "regime_id": "unsupported_extrapolation_boundary",
                "trust_status": UNSUPPORTED_EXTRAPOLATION,
                "boundary_side": boundary_side,
                "preferred_variant": "refuse_to_claim",
                "decision_status": "unsupported_extrapolation",
                "selection_condition": "Outside the measured direct-slice predictor domain.",
                "why_preferred": "The repo should refuse a predictor-backed architecture choice outside the validated 16..64 mac-unit domain at k_depth=32.",
                "why_not_others": "This surface is intentionally bounded to the measured lattice and its local interpolation region.",
            }
        )
    return rows


def build_measured_supported_region_map(
    decision_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not decision_rows or not boundary_rows:
        return {}
    supported_rows = [row for row in decision_rows if row["trust_status"] != UNSUPPORTED_EXTRAPOLATION]
    mac_units_values = sorted({row["mac_units"] for row in supported_rows})
    domain_min = mac_units_values[0]
    domain_max = mac_units_values[-1]
    return {
        "headline": "This map marks the bounded measured regions where the direct-slice predictor supports an architecture choice and where the repo should explicitly refuse to claim one.",
        "summary_lines": [
            f"Supported interpolation is bounded to {domain_min} <= mac_units <= {domain_max} at k_depth=32 for the isolated baseline/shared_lut_saving/shared_dsp_reducing slice family.",
            "Within that domain, baseline remains the default outside hard LUT- or DSP-pressure windows, shared_lut_saving owns the LUT-only window, and shared_dsp_reducing owns the DSP-only window.",
            "Timing-margin-sensitive choice is kept out of the smooth surface because WNS is too unstable for trusted interpolation.",
            "Below 16 mac units or above 64 mac units, the repo should refuse to claim a predictor-backed choice and mark the result as unsupported extrapolation.",
        ],
        "supported_region_rows": [
            {
                "region_id": "baseline_supported_default_region",
                "trust_status": INTERPOLATED_WITHIN_MEASURED_DOMAIN,
                "mac_units_min": domain_min,
                "mac_units_max": domain_max,
                "selection_condition": "No hard LUT/DSP bottleneck or performance/latency dominates.",
            },
            {
                "region_id": "shared_lut_supported_region",
                "trust_status": INTERPOLATED_WITHIN_MEASURED_DOMAIN,
                "mac_units_min": domain_min,
                "mac_units_max": domain_max,
                "selection_condition": "LUT budget falls below the baseline LUT floor but still admits shared_lut_saving with relaxed performance bounds.",
            },
            {
                "region_id": "shared_dsp_supported_region",
                "trust_status": INTERPOLATED_WITHIN_MEASURED_DOMAIN,
                "mac_units_min": domain_min,
                "mac_units_max": domain_max,
                "selection_condition": "DSP budget falls below the baseline DSP floor while LUT budget still admits shared_dsp_reducing and performance bounds remain relaxed.",
            },
            {
                "region_id": "timing_sensitive_unsupported_region",
                "trust_status": UNSUPPORTED_EXTRAPOLATION,
                "mac_units_min": domain_min,
                "mac_units_max": domain_max,
                "selection_condition": "Timing-sensitive interpolation is intentionally refused because WNS is unstable.",
            },
            {
                "region_id": "outside_domain_unsupported_region",
                "trust_status": UNSUPPORTED_EXTRAPOLATION,
                "mac_units_min": None,
                "mac_units_max": None,
                "selection_condition": "Any architecture choice outside the validated direct-slice domain.",
            },
        ],
        "boundary_row_count": len(boundary_rows),
        "decision_row_count": len(decision_rows),
    }


def build_measured_regime_transfer_summary(
    decision_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    predictor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not decision_rows or not boundary_rows or not predictor_rows:
        return {}
    supported_rows = [row for row in decision_rows if row["trust_status"] != UNSUPPORTED_EXTRAPOLATION]
    domain_min = min(row["mac_units"] for row in supported_rows)
    domain_max = max(row["mac_units"] for row in supported_rows)
    lut_rows = [row for row in predictor_rows if row["metric"] == "lut"]
    dsp_rows = [row for row in predictor_rows if row["metric"] == "dsp"]
    throughput_rows = [row for row in predictor_rows if row["metric"] == "effective_throughput_ops_per_cycle"]
    smooth_lut = all((row["max_abs_relative_residual_pct"] or 100.0) <= 2.5 for row in lut_rows)
    exact_dsp = all(row["max_abs_residual"] is not None and row["max_abs_residual"] <= 1e-6 for row in dsp_rows)
    exact_throughput = all(
        row["max_abs_residual"] is not None and row["max_abs_residual"] <= 1e-6 for row in throughput_rows
    )
    summary_lines = [
        f"The measured decision surface is bounded to the validated direct-slice interpolation domain {domain_min} <= mac_units <= {domain_max} at k_depth=32.",
        (
            "The LUT and DSP decision boundaries move smoothly across the measured domain: DSP scales exactly with mac_units, and the LUT boundaries follow low-residual local linear fits."
            if smooth_lut and exact_dsp
            else "The decision boundaries should be treated cautiously because one or more resource predictors are not smooth enough for trusted interpolation."
        ),
        (
            "Baseline remains dominant whenever the design is performance-first or when no hard LUT/DSP bottleneck forces a shared choice."
        ),
        "shared_lut_saving is only worth its overhead inside the predictor-backed LUT window where baseline no longer fits but the LUT-oriented shared point still does.",
        "shared_dsp_reducing is only worth its overhead inside the predictor-backed DSP window where baseline DSP is no longer admissible and the larger LUT footprint still fits.",
        "Timing-sensitive transfer is intentionally excluded from the smooth decision surface because WNS remains too unstable for trusted interpolation.",
        "Outside the validated direct-slice domain, the repo should refuse to claim a predictor-backed architecture choice.",
    ]
    return {
        "headline": "This summary turns the measured predictor into a bounded architecture-choice surface for the isolated direct-slice family.",
        "summary_lines": summary_lines,
        "interpolation_domain_mac_units_min": domain_min,
        "interpolation_domain_mac_units_max": domain_max,
        "dsp_boundary_status": "exact" if exact_dsp else "caution",
        "lut_boundary_status": "smooth_local_fit" if smooth_lut else "caution",
        "throughput_boundary_status": "exact" if exact_throughput else "caution",
        "decision_row_count": len(decision_rows),
        "boundary_row_count": len(boundary_rows),
    }


def _variant_label(architecture_variant: str) -> str:
    return {
        "baseline": "baseline",
        "shared_lut_saving": "shared_lut_saving",
        "shared_dsp_reducing": "shared_dsp_reducing",
    }.get(architecture_variant, architecture_variant)


def _variant_color(architecture_variant: str) -> str:
    return {
        "baseline": "#1f4e79",
        "shared_lut_saving": "#2e8b57",
        "shared_dsp_reducing": "#c75b12",
    }.get(architecture_variant, "#444444")


def _trust_marker(trust_status: str) -> str:
    return {
        MEASURED_LATTICE_POINT: "o",
        INTERPOLATED_WITHIN_MEASURED_DOMAIN: "s",
        UNSUPPORTED_EXTRAPOLATION: "x",
    }.get(trust_status, "o")


def build_final_design_rule_table(
    tradeoff_rows: list[dict[str, Any]],
    flexibility_justification_rows: list[dict[str, Any]],
    scaling_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not tradeoff_rows or not flexibility_justification_rows:
        return []
    lut_rows = [row for row in tradeoff_rows if row["shared_architecture_variant"] == "shared_lut_saving"]
    dsp_rows = [row for row in tradeoff_rows if row["shared_architecture_variant"] == "shared_dsp_reducing"]
    lut_relief_min = min(abs(row["measured_lut_delta_shared_minus_baseline"]) for row in lut_rows if row["measured_lut_delta_shared_minus_baseline"] is not None)
    lut_relief_max = max(abs(row["measured_lut_delta_shared_minus_baseline"]) for row in lut_rows if row["measured_lut_delta_shared_minus_baseline"] is not None)
    dsp_relief_min = min(abs(row["measured_dsp_delta_shared_minus_baseline"]) for row in dsp_rows if row["measured_dsp_delta_shared_minus_baseline"] is not None)
    dsp_relief_max = max(abs(row["measured_dsp_delta_shared_minus_baseline"]) for row in dsp_rows if row["measured_dsp_delta_shared_minus_baseline"] is not None)
    throughput_retention = lut_rows[0]["measured_throughput_retention_pct"]
    latency_factor = lut_rows[0]["measured_latency_increase_factor"]
    return [
        {
            "rule_id": "baseline_performance_first_default",
            "preferred_variant": "baseline",
            "measured_domain": "4x4, 8x4, 8x8 direct slice at k_depth=32",
            "when_justified": "Throughput or latency dominates, or no hard LUT/DSP bottleneck exists.",
            "when_not_justified": "A real LUT-only or DSP-only bottleneck dominates and the corresponding shared implementation is the only relief that fits.",
            "measured_cost_or_benefit": f"Retains the shortest measured schedule at 33 cycles and the highest measured throughput across the direct-slice lattice.",
            "trust_status": DIRECTLY_MEASURED_SUPPORTED,
        },
        {
            "rule_id": "shared_lut_saving_lut_only_rule",
            "preferred_variant": "shared_lut_saving",
            "measured_domain": "4x4, 8x4, 8x8 direct slice at k_depth=32",
            "when_justified": "LUT pressure is the dominant bottleneck and performance bounds are relaxed enough for the shared schedule.",
            "when_not_justified": "DSP pressure, throughput, or latency dominates.",
            "measured_cost_or_benefit": f"Relieves `{lut_relief_min}`..`{lut_relief_max}` LUT versus baseline while retaining `{throughput_retention}`% throughput and increasing latency by `{latency_factor}`x.",
            "trust_status": DIRECTLY_MEASURED_SUPPORTED,
        },
        {
            "rule_id": "shared_dsp_reducing_dsp_only_rule",
            "preferred_variant": "shared_dsp_reducing",
            "measured_domain": "4x4, 8x4, 8x8 direct slice at k_depth=32",
            "when_justified": "DSP pressure is the dominant bottleneck and the larger LUT footprint still fits.",
            "when_not_justified": "LUT pressure, throughput, or latency dominates.",
            "measured_cost_or_benefit": f"Relieves `{dsp_relief_min}`..`{dsp_relief_max}` DSP versus baseline while retaining `{throughput_retention}`% throughput and increasing latency by `{latency_factor}`x.",
            "trust_status": DIRECTLY_MEASURED_SUPPORTED,
        },
        {
            "rule_id": "flexibility_not_a_free_win",
            "preferred_variant": "baseline",
            "measured_domain": "4x4, 8x4, 8x8 direct slice at k_depth=32",
            "when_justified": "No hard resource bottleneck dominates or the relieved bottleneck matters less than the lost performance.",
            "when_not_justified": "A hard LUT-only or DSP-only window forces a shared choice.",
            "measured_cost_or_benefit": "The measured three-way rule survives through 8x8: implementation style determines whether sharing buys LUT relief, DSP relief, or only overhead.",
            "trust_status": scaling_summary.get("overall_rule_status", DIRECTLY_MEASURED_SUPPORTED),
        },
    ]


def build_final_trust_calibration_table(
    support_rows: list[dict[str, Any]],
    calibration_overlay_rows: list[dict[str, Any]],
    predictor_rows: list[dict[str, Any]],
    extrapolation_boundary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not support_rows or not calibration_overlay_rows or not predictor_rows:
        return []
    predictor_by_metric = {}
    for row in predictor_rows:
        predictor_by_metric.setdefault(row["metric"], []).append(row)
    support_lookup = {row["claim_id"]: row for row in support_rows}
    overlay_lookup = {row["overlay_topic"]: row for row in calibration_overlay_rows}
    domain_min = extrapolation_boundary.get("interpolation_domain_mac_units_min")
    domain_max = extrapolation_boundary.get("interpolation_domain_mac_units_max")
    return [
        {
            "topic": "baseline_role",
            "measured_support": support_lookup["baseline_performance_first_role"]["support_level"],
            "calibration_status": overlay_lookup["baseline_lut_expectation"]["calibration_status"],
            "predictor_status": "exact_for_dsp_latency_throughput; local_linear_for_lut_ff",
            "bounded_domain": f"{domain_min} <= mac_units <= {domain_max} at k_depth=32",
            "usage_note": "Baseline role is directly measured supported on the isolated slice; use the predictor locally within the validated domain.",
        },
        {
            "topic": "shared_lut_saving_role",
            "measured_support": support_lookup["shared_lut_saving_lut_relief_role"]["support_level"],
            "calibration_status": overlay_lookup["shared_family_lut_expectation"]["calibration_status"],
            "predictor_status": "exact_for_dsp_latency_throughput; local_linear_for_lut_ff",
            "bounded_domain": f"{domain_min} <= mac_units <= {domain_max} at k_depth=32",
            "usage_note": "Treat as an implementation-specific LUT-relief mechanism, not a universal shared-family numeric truth.",
        },
        {
            "topic": "shared_dsp_reducing_role",
            "measured_support": support_lookup["shared_dsp_reducing_dsp_relief_role"]["support_level"],
            "calibration_status": overlay_lookup["shared_family_dsp_expectation"]["calibration_status"],
            "predictor_status": "exact_for_dsp_latency_throughput; local_linear_for_lut_ff",
            "bounded_domain": f"{domain_min} <= mac_units <= {domain_max} at k_depth=32",
            "usage_note": "Treat as an implementation-specific DSP-relief mechanism, not a universal shared-family numeric truth.",
        },
        {
            "topic": "shared_family_numeric_projection",
            "measured_support": support_lookup["shared_family_dsp_relief"]["support_level"],
            "calibration_status": overlay_lookup["shared_family_numeric_projection_boundary"]["calibration_status"],
            "predictor_status": "modelled_family_with_measured_caution_band",
            "bounded_domain": "outside the isolated direct-slice predictor scope",
            "usage_note": "Shared-family framework numbers remain modelled-family expectations read through trust and calibration overlays, not replaced by the local predictor.",
        },
        {
            "topic": "wns_numeric_use",
            "measured_support": "measured_caution_only",
            "calibration_status": "too_unstable_for_numeric_trust",
            "predictor_status": "caution_only_local_fit",
            "bounded_domain": f"{domain_min} <= mac_units <= {domain_max} at k_depth=32",
            "usage_note": "WNS is measured on the lattice, but remains too unstable for trusted timing-sensitive interpolation.",
        },
        {
            "topic": "outside_domain_architecture_choice",
            "measured_support": UNSUPPORTED_EXTRAPOLATION,
            "calibration_status": "not_applicable",
            "predictor_status": "unsupported_extrapolation",
            "bounded_domain": f"mac_units < {domain_min} or mac_units > {domain_max}",
            "usage_note": "The repo should explicitly refuse a predictor-backed architecture choice outside the validated direct-slice domain.",
        },
    ]


def build_final_architecture_choice_boundary_table(
    boundary_rows: list[dict[str, Any]],
    regime_transfer_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not boundary_rows:
        return []
    measured_rows = [row for row in boundary_rows if row["trust_status"] == MEASURED_LATTICE_POINT]
    rows: list[dict[str, Any]] = []
    for row in measured_rows:
        rows.extend(
            [
                {
                    "boundary_id": f"lut_window_{row['mac_units']}",
                    "mac_units": row["mac_units"],
                    "trust_status": row["trust_status"],
                    "boundary_kind": "lut_pressure",
                    "preferred_variant": "shared_lut_saving",
                    "selection_boundary": f"LUT budget in [`{round(row['lut_budget_shared_lut_min'], 3)}`, `{round(row['lut_budget_baseline_min'], 3)}`) with relaxed throughput/latency constraints.",
                    "usage_note": "Inside this window, shared_lut_saving is the measured lowest-LUT option and is the only shared variant worth the schedule overhead.",
                },
                {
                    "boundary_id": f"dsp_window_{row['mac_units']}",
                    "mac_units": row["mac_units"],
                    "trust_status": row["trust_status"],
                    "boundary_kind": "dsp_pressure",
                    "preferred_variant": "shared_dsp_reducing",
                    "selection_boundary": f"DSP budget in [`{round(row['dsp_budget_shared_dsp_min'], 3)}`, `{round(row['dsp_budget_baseline_min'], 3)}`) with LUT budget >= `{round(row['shared_dsp_lut_floor'], 3)}` and relaxed throughput/latency constraints.",
                    "usage_note": "Inside this window, shared_dsp_reducing is the only measured option that actually relieves DSP pressure.",
                },
                {
                    "boundary_id": f"baseline_default_{row['mac_units']}",
                    "mac_units": row["mac_units"],
                    "trust_status": row["trust_status"],
                    "boundary_kind": "baseline_default",
                    "preferred_variant": "baseline",
                    "selection_boundary": f"LUT budget >= `{round(row['lut_budget_baseline_min'], 3)}` and DSP budget >= `{round(row['dsp_budget_baseline_min'], 3)}`, or throughput > `{round(row['shared_throughput_floor_ops_per_cycle'], 6)}`.",
                    "usage_note": "Once baseline fits and performance matters, the shared schedule penalty is not justified.",
                },
            ]
        )
    rows.append(
        {
            "boundary_id": "timing_sensitive_interpolation_refused",
            "mac_units": None,
            "trust_status": UNSUPPORTED_EXTRAPOLATION,
            "boundary_kind": "timing_margin",
            "preferred_variant": "measured_grid_lookup_only",
            "selection_boundary": "No smooth timing-sensitive boundary is claimed.",
            "usage_note": "Timing-sensitive choice remains excluded from the smooth surface because WNS is caution-only.",
        }
    )
    rows.append(
        {
            "boundary_id": "outside_domain_refusal",
            "mac_units": None,
            "trust_status": UNSUPPORTED_EXTRAPOLATION,
            "boundary_kind": "outside_domain",
            "preferred_variant": "refuse_to_claim",
            "selection_boundary": f"Outside {regime_transfer_summary.get('interpolation_domain_mac_units_min')} <= mac_units <= {regime_transfer_summary.get('interpolation_domain_mac_units_max')} at k_depth=32.",
            "usage_note": "The repo explicitly refuses a predictor-backed architecture choice outside the validated direct-slice domain.",
        }
    )
    return rows


def build_final_results_summary(
    design_rule_table: list[dict[str, Any]],
    trust_calibration_table: list[dict[str, Any]],
    boundary_table: list[dict[str, Any]],
    regime_transfer_summary: dict[str, Any],
) -> dict[str, Any]:
    if not design_rule_table or not trust_calibration_table or not boundary_table:
        return {}
    return {
        "headline": "This final direct-slice results pack turns the measured lattice into a bounded thesis-ready conclusion for what each architecture buys, what it costs, and when it should be chosen.",
        "summary_lines": [
            "Across the measured 4x4, 8x4, and 8x8 direct-slice lattice, baseline remains the performance-first option, shared_lut_saving remains the LUT-relief option, and shared_dsp_reducing remains the DSP-relief option.",
            "The measured shared implementations are not generally better architectures; they are bottleneck-specific relief mechanisms whose utility disappears when the relieved bottleneck does not dominate the lost performance.",
            "Within the validated domain 16 <= mac_units <= 64 at k_depth=32, local interpolation is acceptable for DSP, latency, throughput, LUT, and FF under the measured predictor, while WNS remains caution-only.",
            "The final decision surface is bounded: it supports predictor-backed LUT and DSP budget windows inside the measured domain and explicitly refuses unsupported extrapolation outside it.",
            "These conclusions are specific to the isolated baseline/shared_lut_saving/shared_dsp_reducing direct-slice family and should not be silently promoted to the broader shared-family framework model.",
        ],
        "design_rule_row_count": len(design_rule_table),
        "trust_calibration_row_count": len(trust_calibration_table),
        "boundary_row_count": len(boundary_table),
        "interpolation_domain_mac_units_min": regime_transfer_summary.get("interpolation_domain_mac_units_min"),
        "interpolation_domain_mac_units_max": regime_transfer_summary.get("interpolation_domain_mac_units_max"),
    }


def build_final_artifact_index(output_dir: Path, final_results_summary: dict[str, Any]) -> list[dict[str, Any]]:
    domain_min = final_results_summary.get("interpolation_domain_mac_units_min")
    domain_max = final_results_summary.get("interpolation_domain_mac_units_max")
    domain_note = f"{domain_min} <= mac_units <= {domain_max} at k_depth=32"
    rows = [
        {
            "filename": "final_tradeoff_figures/lut_vs_mac_units.png",
            "purpose": "Final architecture tradeoff figure for LUT scaling across the direct-slice family.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}",
            "thesis_use_note": f"Use in Results to show that shared_lut_saving is the LUT-relief option within the validated domain {domain_note}.",
        },
        {
            "filename": "final_tradeoff_figures/dsp_vs_mac_units.png",
            "purpose": "Final architecture tradeoff figure for DSP scaling across the direct-slice family.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}",
            "thesis_use_note": "Use in Results to show that shared_dsp_reducing is the DSP-relief option while shared_lut_saving stays DSP-flat.",
        },
        {
            "filename": "final_tradeoff_figures/throughput_vs_mac_units.png",
            "purpose": "Final architecture tradeoff figure for throughput scaling and shared overhead.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}",
            "thesis_use_note": "Use in Results to show that baseline remains the performance-first option and that shared overhead is schedule-driven.",
        },
        {
            "filename": "final_tradeoff_figures/latency_vs_mac_units.png",
            "purpose": "Final architecture tradeoff figure for latency scaling and fixed schedule overhead.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}",
            "thesis_use_note": "Use in Results or Discussion to show that both shared implementations pay the 65-cycle schedule while baseline stays at 33 cycles.",
        },
        {
            "filename": "final_tradeoff_figures/wns_vs_mac_units_caution.png",
            "purpose": "Caution-only WNS trend figure across the direct-slice family.",
            "measured_interpolation_status": "caution_only_wns",
            "thesis_use_note": "Use only as a cautionary timing trend figure; do not use it to claim a smooth timing-sensitive decision boundary.",
        },
        {
            "filename": "final_predictor_validation_figures/measured_vs_fitted_lut.png",
            "purpose": "Predictor validation figure for measured versus fitted LUT scaling.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}",
            "thesis_use_note": "Use in Methods/Results to show that LUT admits a compact local linear fit over the validated direct-slice domain.",
        },
        {
            "filename": "final_predictor_validation_figures/measured_vs_fitted_ff.png",
            "purpose": "Predictor validation figure for measured versus fitted FF scaling.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}",
            "thesis_use_note": "Use in Methods/Results to show that FF is predictable locally but remains part of the bounded direct-slice predictor only.",
        },
        {
            "filename": "final_predictor_validation_figures/residuals_lut.png",
            "purpose": "Residual plot for the LUT local fit.",
            "measured_interpolation_status": MEASURED_LATTICE_POINT,
            "thesis_use_note": "Use in Results appendix or validation subsection to justify interpolation trust for LUT.",
        },
        {
            "filename": "final_predictor_validation_figures/residuals_ff.png",
            "purpose": "Residual plot for the FF local fit.",
            "measured_interpolation_status": MEASURED_LATTICE_POINT,
            "thesis_use_note": "Use in Results appendix or validation subsection to justify interpolation trust for FF.",
        },
        {
            "filename": "final_predictor_validation_figures/residuals_wns_caution.png",
            "purpose": "Residual plot for the caution-only WNS fit.",
            "measured_interpolation_status": "caution_only_wns",
            "thesis_use_note": "Use only to justify why timing-sensitive interpolation is explicitly unsupported.",
        },
        {
            "filename": "final_decision_surface_figures/supported_choice_regions.png",
            "purpose": "Final measured decision-surface figure over supported architecture-choice regions.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}, {UNSUPPORTED_EXTRAPOLATION}",
            "thesis_use_note": "Use in Results/Discussion to show where baseline, shared_lut_saving, and shared_dsp_reducing are preferred inside the validated direct-slice domain.",
        },
        {
            "filename": "final_decision_surface_figures/lut_pressure_decision_boundary.png",
            "purpose": "Final LUT-pressure boundary figure for when shared_lut_saving is worth its overhead.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}",
            "thesis_use_note": "Use in Discussion to show the bounded LUT-only window where shared_lut_saving is justified.",
        },
        {
            "filename": "final_decision_surface_figures/dsp_pressure_decision_boundary.png",
            "purpose": "Final DSP-pressure boundary figure for when shared_dsp_reducing is worth its overhead.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}",
            "thesis_use_note": "Use in Discussion to show the bounded DSP-only window and its coupled LUT admission requirement.",
        },
        {
            "filename": "final_decision_surface_figures/timing_sensitive_unsupported.png",
            "purpose": "Explicit unsupported figure for timing-sensitive interpolation.",
            "measured_interpolation_status": UNSUPPORTED_EXTRAPOLATION,
            "thesis_use_note": "Use to defend the explicit refusal to claim a smooth timing-sensitive architecture-choice surface.",
        },
        {
            "filename": "final_design_rule_table.md",
            "purpose": "Compact final table of measured design rules for what each architecture buys and costs.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}",
            "thesis_use_note": "Use directly in the Results or Discussion chapter as the compact architecture-rule table.",
        },
        {
            "filename": "final_trust_calibration_table.md",
            "purpose": "Compact final table of trust, calibration, and bounded predictor status.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}, {UNSUPPORTED_EXTRAPOLATION}",
            "thesis_use_note": "Use in Methods/Discussion to explain what is directly measured, locally interpolated, caution-only, or explicitly refused.",
        },
        {
            "filename": "final_architecture_choice_boundary_table.md",
            "purpose": "Compact final table of predictor-backed architecture-choice boundaries.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}, {UNSUPPORTED_EXTRAPOLATION}",
            "thesis_use_note": "Use in Discussion to show the bounded LUT-pressure and DSP-pressure windows together with the explicit refusal outside the validated domain.",
        },
        {
            "filename": "final_results_summary.md",
            "purpose": "Concise thesis-style final subsection text for the direct-slice family.",
            "measured_interpolation_status": f"{MEASURED_LATTICE_POINT}, {INTERPOLATED_WITHIN_MEASURED_DOMAIN}, {UNSUPPORTED_EXTRAPOLATION}",
            "thesis_use_note": "Use directly as the draft nucleus for the final Results/Discussion subsection.",
        },
    ]
    for row in rows:
        row["absolute_path"] = str((output_dir / row["filename"]).resolve())
    return rows


def build_final_reproducibility_guide(final_results_summary: dict[str, Any]) -> dict[str, Any]:
    domain_min = final_results_summary.get("interpolation_domain_mac_units_min")
    domain_max = final_results_summary.get("interpolation_domain_mac_units_max")
    return {
        "headline": "This guide freezes the direct-slice family results pack into a single reproducible thesis-ready pipeline without changing the measured conclusions.",
        "regeneration_command": "make fpga_mac_direct_final_pack",
        "direct_runner_command": "python3 analysis/run_mac_array_direct_slice.py",
        "framework_runner_command": "python3 analysis/run_mac_array_framework.py --config experiments/configs/mac_array_framework_v2.json",
        "validated_domain": f"{domain_min} <= mac_units <= {domain_max} at k_depth=32 for the isolated baseline/shared_lut_saving/shared_dsp_reducing direct-slice family.",
        "summary_lines": [
            "Directly measured lattice points are the measured 4x4, 8x4, and 8x8 direct-slice runs for baseline, shared_lut_saving, and shared_dsp_reducing.",
            "Interpolated within measured domain applies only to the local direct-slice predictor and the bounded decision surface inside the validated domain.",
            "Unsupported extrapolation applies outside the validated mac-unit domain and to any smooth timing-sensitive interpolation claim.",
            "WNS remains caution-only: it may be plotted and reported at measured points, but it is not promoted to a trusted timing-sensitive interpolation surface.",
            "The final figures and tables are thesis-use artifacts for the isolated direct-slice family only and do not silently replace the broader shared-family framework model.",
        ],
    }


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
    if len(grid_summaries) >= 3:
        scaling_rule = (
            "The measured three-way rule survives through 8x8: baseline remains the performance-first option, shared_lut_saving remains the best LUT-relief option, and shared_dsp_reducing remains the DSP-relief option."
            if survives_all
            else "The measured three-way rule partially breaks by 8x8; use the per-grid measured ordering rather than assuming the smaller-grid roles scale cleanly."
        )
    elif len(grid_summaries) >= 2:
        scaling_rule = (
            "The 4x4 three-way rule survives at 8x4: baseline remains the performance-first option, shared_lut_saving remains the best LUT-relief option, and shared_dsp_reducing remains the DSP-relief option."
            if survives_all
            else "The 4x4 three-way rule partially breaks at 8x4; use the per-grid measured ordering rather than assuming the 4x4 roles scale cleanly."
        )
    else:
        scaling_rule = "Only one measured three-way scale point is available so far, so the scaling rule is not yet established."
    return {
        "headline": "This summary asks whether the measured three-way resource-relief rule survives across the currently measured grids.",
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
            measured_basis=f"Across the measured {', '.join(measured_grids)} direct slices, baseline keeps the 33-cycle schedule and higher ops/cycle than both measured shared implementations.",
            trust_note="This role is directly measured on the isolated slice, not inferred from the family model.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_lut_saving_lut_relief_role",
            claim_scope="measured_implementation_role",
            claim_subject="shared_lut_saving",
            claim_text="shared_lut_saving is the LUT-relief option.",
            support_level=DIRECTLY_MEASURED_SUPPORTED,
            measured_basis=f"Across the measured {', '.join(measured_grids)} grids, shared_lut_saving is the lowest-LUT shared implementation and lowers LUT versus baseline while keeping the shared 65-cycle schedule.",
            trust_note="This is a directly measured implementation-specific role, not a statement about every shared strategy.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_dsp_reducing_dsp_relief_role",
            claim_scope="measured_implementation_role",
            claim_subject="shared_dsp_reducing",
            claim_text="shared_dsp_reducing is the DSP-relief option.",
            support_level=DIRECTLY_MEASURED_SUPPORTED,
            measured_basis=f"Across the measured {', '.join(measured_grids)} grids, shared_dsp_reducing reduces mapped DSP to 0 while retaining the shared 65-cycle schedule.",
            trust_note="This is directly measured for the implemented DSP-oriented shared slice only.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_family_latency_throughput_penalty",
            claim_scope="family_level_direction",
            claim_subject="shared_modelled_family",
            claim_text="Shared-family implementations trade lower throughput and higher latency for resource relief.",
            support_level=DIRECTLY_MEASURED_SUPPORTED,
            measured_basis=f"Both measured shared implementations across {', '.join(measured_grids)} move from the 33-cycle baseline point to the 65-cycle shared point and halve effective ops/cycle.",
            trust_note="The latency/throughput penalty direction is directly supported by measured implementations, even though the framework family remains modelled.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_family_lut_relief",
            claim_scope="family_level_direction",
            claim_subject="shared_modelled_family",
            claim_text="Shared-family implementations can relieve LUT pressure.",
            support_level=MEASURED_DIRECTIONALLY_SUPPORTED,
            measured_basis=f"Both measured shared implementations reduce LUT versus baseline at {', '.join(measured_grids)}, but the amount of LUT relief is implementation-specific.",
            trust_note="Measured data supports LUT relief directionally, not as a single family-wide realization or fixed magnitude.",
            measured_grids=measured_grids,
        ),
        _support_row(
            claim_id="shared_family_dsp_relief",
            claim_scope="family_level_direction",
            claim_subject="shared_modelled_family",
            claim_text="Shared-family implementations reduce DSP pressure.",
            support_level=MEASURED_PARTIAL_SUPPORT,
            measured_basis=f"The measured DSP-oriented shared implementation reduces DSP to 0 across {', '.join(measured_grids)}, but the measured LUT-oriented shared implementation stays DSP-flat.",
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
            claim_text="The 8x8 shared-family DSP reduction is directly measured as one family-wide shared hardware truth in this repo's current isolated direct slice.",
            support_level=MEASURED_PARTIAL_SUPPORT if "8x8" in measured_grids else EXTRAPOLATED_BEYOND_MEASURED_SUPPORT,
            measured_basis=(
                "The repo now has direct 8x8 shared implementation measurements for both shared_lut_saving and shared_dsp_reducing, but the broader family-level 8x8 shared row in the framework remains a modelled-family conclusion rather than one directly measured shared-family truth."
                if "8x8" in measured_grids
                else "The repo has direct shared measurements only at 4x4 and 8x4. The 8x8 shared DSP reduction still comes from anchored prior-study evidence plus the modelled family layer."
            ),
            trust_note=(
                "8x8 shared implementations are now directly measured on the isolated slice, but the framework's single 8x8 shared-family row should still be read as a modelled-family conclusion with implementation-dependent realization."
                if "8x8" in measured_grids
                else "Treat 8x8 shared-family DSP reduction as anchored/modelled rather than directly measured implementation truth."
            ),
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
        (
            "8x8 shared implementations are now directly measured on the isolated slice, but the single 8x8 shared-family framework row remains a modelled-family conclusion rather than one measured shared hardware truth."
            if any(row["claim_id"] == "shared_8x8_dsp_reduction_anchor" and row["support_level"] == MEASURED_PARTIAL_SUPPORT for row in support_rows)
            else "Family-level 8x8 shared DSP reduction remains an anchored/modelled claim that is extrapolated beyond measured support."
        ),
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
            usage_note=(
                "The current calibration layer now reaches through the measured 4x4, 8x4, and 8x8 slice evidence, but the framework's single shared-family resource row still remains modelled-family rather than one directly measured shared hardware truth."
                if "8x8" in measured_grids
                else "The current calibration layer bounds trust near the measured 4x4 and 8x4 slice evidence; 8x8 shared resource claims remain anchored/modelled."
            ),
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
            f"Baseline LUT is numerically optimistic in the lightweight framework across the measured {', '.join(measured_grids)} bridge, with relative error spanning `{baseline_error_min}`% to `{baseline_error_max}`%."
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
        (
            "The new 8x8 direct shared measurements remove the isolated direct-slice measurement gap at 8x8, but the framework's single shared-family 8x8 resource row still remains a modelled/anchored family-level reading rather than one measured shared hardware truth."
            if "8x8" in measured_grids
            else "8x8 shared-family resource conclusions remain modelled/anchored beyond the directly calibrated 4x4 and 8x4 slice evidence."
        ),
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
    if len(set(timing_choices.values())) == 1:
        timing_line = f"Timing-margin preference is consistent across the measured slice and favors `{next(iter(set(timing_choices.values())))}.`"
    else:
        timing_line = "Timing-margin preference is grid-dependent in the measured slice: " + ", ".join(
            f"{grid} favors `{variant}`" for grid, variant in sorted(timing_choices.items())
        ) + "."
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


def render_measured_regime_transfer_summary(
    output_path: Path,
    summary: dict[str, Any],
    decision_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Measured Regime Transfer Summary",
        "",
    ]
    if not summary:
        lines.append("- No measured regime-transfer surface is available yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
        lines.extend(["", "## Decision Surface", ""])
        for row in decision_rows[:12]:
            lines.append(
                f"- `mac_units={row['mac_units']}` / `{row['regime_id']}` / `{row['trust_status']}` -> `{row['preferred_variant']}`: {row['selection_condition']}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_measured_supported_region_map(output_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Measured Supported Region Map",
        "",
    ]
    if not summary:
        lines.append("- No supported-region map is available yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
        lines.extend(["", "## Supported Regions", ""])
        for row in summary.get("supported_region_rows", []):
            lines.append(
                f"- `{row['region_id']}` / `{row['trust_status']}`: {row['selection_condition']}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_measured_predictor_summary(output_path: Path, summary: dict[str, Any], predictor_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Measured Predictor Summary",
        "",
    ]
    if not summary:
        lines.append("- No measured predictor can be stated yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
        lines.extend(["", "## Predictor Table", ""])
        for row in predictor_rows:
            lines.append(
                f"- `{row['architecture_variant']}` / `{row['metric']}` -> `{row['fit_status']}` with formula `{row['predictor_formula']}` and max residual `{row['max_abs_residual']}`."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_measured_extrapolation_boundary(output_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Measured Extrapolation Boundary",
        "",
    ]
    if not summary:
        lines.append("- No extrapolation boundary can be stated yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
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
        "- Scope remains intentionally narrow: direct measurement currently covers the isolated baseline slice plus selective shared implementations on the direct slice, not the full baseline/shared/replicated/adaptive family.",
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
        "- Scope is intentionally narrow: baseline and the measured shared implementations are compared only on the isolated direct slice at the currently measured bridge points.",
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
        lines.append("- No multi-scale shared comparison is available yet.")
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


def render_final_results_summary(output_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Final Results Summary",
        "",
    ]
    if not summary:
        lines.append("- No final direct-slice results summary is available yet.")
    else:
        lines.append(f"- {summary['headline']}")
        for line in summary["summary_lines"]:
            lines.append(f"- {line}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_final_reproducibility_guide(
    output_path: Path,
    guide: dict[str, Any],
    artifact_index_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Final Direct-Slice Family Reproducibility Guide",
        "",
    ]
    if not guide:
        lines.append("- No reproducibility guide is available yet.")
    else:
        lines.append(f"- {guide['headline']}")
        lines.append(f"- Regeneration target: `{guide['regeneration_command']}`")
        lines.append(f"- Direct-slice runner: `{guide['direct_runner_command']}`")
        lines.append(f"- Framework pack refresh: `{guide['framework_runner_command']}`")
        lines.append(f"- Validated domain: {guide['validated_domain']}")
        for line in guide["summary_lines"]:
            lines.append(f"- {line}")
        lines.extend(["", "## Thesis Artifact Roles", ""])
        for row in artifact_index_rows:
            lines.append(f"- `{row['filename']}`: {row['thesis_use_note']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _render_simple_markdown_table(output_path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("- No rows available.")
    else:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def render_final_table_markdown(output_path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    _render_simple_markdown_table(output_path, title, rows)


def _direct_slice_plot_setup():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cnn_fpga_mplconfig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def render_final_tradeoff_figures(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    plt = _direct_slice_plot_setup()
    if plt is None:
        return []
    measured_rows = sorted(
        [row for row in rows if row["comparison_status"] == "direct_measured_vs_modelled"],
        key=lambda row: (row["architecture"], row["mac_units"]),
    )
    if not measured_rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    metric_specs = [
        ("measured_lut", "LUT", "Measured LUT vs MAC Units", "lut_vs_mac_units.png"),
        ("measured_dsp", "DSP", "Measured DSP vs MAC Units", "dsp_vs_mac_units.png"),
        (
            "measured_effective_throughput_ops_per_cycle",
            "Throughput (ops/cycle)",
            "Measured Throughput vs MAC Units",
            "throughput_vs_mac_units.png",
        ),
        ("measured_latency_cycles", "Latency (cycles)", "Measured Latency vs MAC Units", "latency_vs_mac_units.png"),
        ("measured_wns_ns", "WNS (ns)", "Measured WNS vs MAC Units (Caution-Only)", "wns_vs_mac_units_caution.png"),
    ]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in measured_rows:
        by_variant.setdefault(row["architecture"], []).append(row)
    for metric_key, ylabel, title, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for architecture_variant in ("baseline", "shared_lut_saving", "shared_dsp_reducing"):
            variant_rows = sorted(by_variant.get(architecture_variant, []), key=lambda item: item["mac_units"])
            if not variant_rows:
                continue
            ax.plot(
                [row["mac_units"] for row in variant_rows],
                [row[metric_key] for row in variant_rows],
                marker="o",
                linewidth=2.0,
                color=_variant_color(architecture_variant),
                label=_variant_label(architecture_variant),
            )
        ax.set_xlabel("MAC units")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))
    return generated


def render_final_predictor_validation_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
    predictor_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
) -> list[str]:
    plt = _direct_slice_plot_setup()
    if plt is None:
        return []
    measured_rows = sorted(
        [row for row in rows if row["comparison_status"] == "direct_measured_vs_modelled"],
        key=lambda row: (row["architecture"], row["mac_units"]),
    )
    if not measured_rows or not predictor_rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    predictor_lookup = {(row["architecture_variant"], row["metric"]): row for row in predictor_rows}
    for metric, filename, title in (
        ("lut", "measured_vs_fitted_lut.png", "Measured vs Fitted LUT (Local Linear Fit)"),
        ("ff", "measured_vs_fitted_ff.png", "Measured vs Fitted FF (Local Linear Fit)"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for architecture_variant in ("baseline", "shared_lut_saving", "shared_dsp_reducing"):
            variant_rows = sorted(
                [row for row in measured_rows if row["architecture"] == architecture_variant],
                key=lambda item: item["mac_units"],
            )
            if not variant_rows:
                continue
            predictor = predictor_lookup[(architecture_variant, metric)]
            xs = [row["mac_units"] for row in variant_rows]
            measured_values = [row[f"measured_{metric}"] for row in variant_rows]
            fitted_values = [
                _predict_from_formula(predictor["predictor_formula"], row["mac_units"], row["k_depth"])
                for row in variant_rows
            ]
            ax.scatter(xs, measured_values, color=_variant_color(architecture_variant), marker="o", label=f"{architecture_variant} measured")
            ax.plot(xs, fitted_values, color=_variant_color(architecture_variant), linestyle="--", linewidth=2.0, label=f"{architecture_variant} fit")
        ax.set_xlabel("MAC units")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))
    for metric, filename, title in (
        ("lut", "residuals_lut.png", "LUT Fit Residuals"),
        ("ff", "residuals_ff.png", "FF Fit Residuals"),
        ("wns_ns", "residuals_wns_caution.png", "WNS Fit Residuals (Caution-Only)"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle=":")
        for architecture_variant in ("baseline", "shared_lut_saving", "shared_dsp_reducing"):
            metric_rows = sorted(
                [row for row in residual_rows if row["architecture_variant"] == architecture_variant and row["metric"] == metric],
                key=lambda item: item["mac_units"],
            )
            if not metric_rows:
                continue
            ax.plot(
                [row["mac_units"] for row in metric_rows],
                [row["residual"] for row in metric_rows],
                marker="o",
                linewidth=1.8,
                color=_variant_color(architecture_variant),
                label=_variant_label(architecture_variant),
            )
        ax.set_xlabel("MAC units")
        ax.set_ylabel("Residual")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))
    return generated


def render_final_decision_surface_figures(
    output_dir: Path,
    decision_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> list[str]:
    plt = _direct_slice_plot_setup()
    if plt is None:
        return []
    if not decision_rows or not boundary_rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    supported_decisions = sorted(
        [row for row in decision_rows if row["regime_id"] != "unsupported_extrapolation_boundary"],
        key=lambda row: (row["regime_id"], row["mac_units"]),
    )
    regime_order = [
        "lut_budget_tight_relaxed_performance",
        "dsp_budget_tight_relaxed_performance",
        "performance_or_latency_dominant",
        "no_hard_resource_bottleneck",
        "timing_margin_sensitive",
    ]
    regime_y = {regime: index for index, regime in enumerate(reversed(regime_order))}
    color_by_preference = {
        "baseline": _variant_color("baseline"),
        "shared_lut_saving": _variant_color("shared_lut_saving"),
        "shared_dsp_reducing": _variant_color("shared_dsp_reducing"),
        "unsupported_within_surface": "#999999",
        "refuse_to_claim": "#bbbbbb",
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for row in supported_decisions:
        ax.scatter(
            row["mac_units"],
            regime_y[row["regime_id"]],
            color=color_by_preference.get(row["preferred_variant"], "#666666"),
            marker=_trust_marker(row["trust_status"]),
            s=48,
        )
    ax.set_xlabel("MAC units")
    ax.set_yticks([regime_y[regime] for regime in reversed(regime_order)])
    ax.set_yticklabels(
        [
            "LUT-pressure",
            "DSP-pressure",
            "Performance/latency",
            "No bottleneck",
            "Timing-sensitive\nunsupported",
        ]
    )
    ax.set_title("Supported Architecture-Choice Regions")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / "supported_choice_regions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    supported_boundary_rows = sorted(
        [row for row in boundary_rows if row["trust_status"] != UNSUPPORTED_EXTRAPOLATION and "lut_budget_shared_lut_min" in row],
        key=lambda row: row["mac_units"],
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = [row["mac_units"] for row in supported_boundary_rows]
    shared_lut = [row["lut_budget_shared_lut_min"] for row in supported_boundary_rows]
    baseline_lut = [row["lut_budget_baseline_min"] for row in supported_boundary_rows]
    ax.plot(xs, shared_lut, color=_variant_color("shared_lut_saving"), linewidth=2.0, label="shared_lut_saving LUT floor")
    ax.plot(xs, baseline_lut, color=_variant_color("baseline"), linewidth=2.0, label="baseline LUT floor")
    ax.fill_between(xs, shared_lut, baseline_lut, color="#9fd8b7", alpha=0.35, label="shared_lut_saving justified window")
    ax.set_xlabel("MAC units")
    ax.set_ylabel("LUT budget")
    ax.set_title("LUT-Pressure Decision Boundary")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "lut_pressure_decision_boundary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.0), sharex=True)
    baseline_dsp = [row["dsp_budget_baseline_min"] for row in supported_boundary_rows]
    shared_dsp = [row["dsp_budget_shared_dsp_min"] for row in supported_boundary_rows]
    shared_dsp_lut_floor = [row["shared_dsp_lut_floor"] for row in supported_boundary_rows]
    axes[0].plot(xs, shared_dsp, color=_variant_color("shared_dsp_reducing"), linewidth=2.0, label="shared_dsp_reducing DSP floor")
    axes[0].plot(xs, baseline_dsp, color=_variant_color("baseline"), linewidth=2.0, label="baseline DSP floor")
    axes[0].fill_between(xs, shared_dsp, baseline_dsp, color="#f2c49b", alpha=0.35, label="shared_dsp_reducing justified window")
    axes[0].set_ylabel("DSP budget")
    axes[0].set_title("DSP-Pressure Decision Boundary")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(xs, shared_dsp_lut_floor, color=_variant_color("shared_dsp_reducing"), linewidth=2.0)
    axes[1].set_xlabel("MAC units")
    axes[1].set_ylabel("Required LUT budget")
    axes[1].set_title("LUT Floor Needed to Admit shared_dsp_reducing")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "dsp_pressure_decision_boundary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.axvspan(min(xs), max(xs), color="#d8d8d8", alpha=0.65)
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("MAC units")
    ax.set_title("Timing-Sensitive Architecture Choice: Unsupported for Interpolation")
    ax.text((min(xs) + max(xs)) / 2.0, 0.5, "WNS is caution-only;\nuse measured grid lookup, not a smooth boundary.", ha="center", va="center")
    fig.tight_layout()
    path = output_dir / "timing_sensitive_unsupported.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))
    return generated


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
