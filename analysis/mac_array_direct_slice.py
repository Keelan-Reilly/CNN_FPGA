#!/usr/bin/env python3
"""Direct MAC-array baseline slice comparison and calibration helpers."""

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
]
FRAMEWORK_CONFIG = REPO_ROOT / "experiments" / "configs" / "mac_array_framework_v2.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _parse_params(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def direct_latency_model(k_depth: int) -> int:
    return k_depth + 1


def direct_throughput_model(rows: int, cols: int, k_depth: int) -> float:
    return float(rows * cols * k_depth) / float(direct_latency_model(k_depth))


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
    baseline_arch = evidence.architectures["baseline"]
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
            if grid_rows <= 0 or grid_cols <= 0:
                continue
            grid = GridSpec(label=f"{grid_rows}x{grid_cols}", rows=grid_rows, cols=grid_cols)
            mac_units = grid_rows * grid_cols
            static_model = derive_static_row(grid, baseline_arch, evidence)
            measured_eff_tput = measured.get("effective_throughput_ops_per_cycle")
            measured_latency = measured.get("latency_cycles")
            model_latency = direct_latency_model(k_depth)
            model_eff_tput = direct_throughput_model(grid_rows, grid_cols, k_depth)
            rows.append(
                {
                    "run_id": measured.get("run_id"),
                    "grid": grid.label,
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "mac_units": mac_units,
                    "k_depth": k_depth,
                    "architecture": "baseline",
                    "status": measured.get("status"),
                    "direct_evidence_kind": (
                        "direct_measured_mac_array_slice"
                        if measured.get("status") == "succeeded"
                        else "direct_measurement_missing_or_failed"
                    ),
                    "direct_evidence_source": source_path,
                    "framework_model_kind": "framework_baseline_static_model",
                    "framework_model_source": str((REPO_ROOT / cfg["evidence_path"]).relative_to(REPO_ROOT)),
                    "measured_dsp": measured.get("dsp"),
                    "model_dsp": static_model["dsp"],
                    "dsp_delta": None if measured.get("dsp") is None else measured.get("dsp") - static_model["dsp"],
                    "measured_lut": measured.get("lut"),
                    "model_lut": static_model["lut"],
                    "lut_delta": None if measured.get("lut") is None else measured.get("lut") - static_model["lut"],
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
                        "Direct MAC-array slice measurement is available for this baseline grid."
                        if measured.get("status") == "succeeded"
                        else "Direct slice exists but this grid point has not been measured successfully yet."
                    ),
                }
            )
    rows.sort(key=lambda row: (row["mac_units"], row["grid_rows"], row["grid_cols"], row["run_id"] or ""))

    measured_rows = [row for row in rows if row["comparison_status"] == "direct_measured_vs_modelled" and row["measured_lut"] is not None]
    lut_fit = _fit_linear([float(row["mac_units"]) for row in measured_rows], [float(row["measured_lut"]) for row in measured_rows])
    for row in rows:
        if lut_fit is None:
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
    measured = [row for row in rows if row["comparison_status"] == "direct_measured_vs_modelled"]
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


def render_direct_slice_summary(output_path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Direct MAC-Array Slice Summary",
        "",
        "- This summary covers the standalone directly measurable baseline MAC-array slice.",
        "- Scope remains intentionally narrow: it is a baseline-only spatial slice, not the full baseline/shared/replicated/adaptive family.",
        f"- Directly measured points: `{summary['measured_points']}`.",
        f"- DSP exact matches: `{summary['dsp_exact_match_count']}`. Latency exact matches: `{summary['latency_exact_match_count']}`. Throughput exact matches: `{summary['throughput_exact_match_count']}`.",
        "",
    ]
    if not rows:
        lines.append("- No direct MAC-array aggregate is available yet. Run the direct-slice experiment config to populate this comparison.")
    else:
        lines.append("## Compared Points")
        lines.append("")
        for row in rows:
            lines.append(
                f"- `{row['grid']}` -> DSP measured/modelled `{row['measured_dsp']}`/`{row['model_dsp']}`, "
                f"LUT measured/modelled `{row['measured_lut']}`/`{row['model_lut']}`, "
                f"calibrated LUT `{row['calibrated_model_lut']}`, "
                f"throughput measured/modelled `{row['measured_effective_throughput_ops_per_cycle']}`/`{row['direct_slice_throughput_model_ops_per_cycle']}`."
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


def render_direct_calibration_plot(output_path: Path, rows: list[dict[str, Any]]) -> str | None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cnn_fpga_mplconfig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    measured = [row for row in rows if row["comparison_status"] == "direct_measured_vs_modelled"]
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
