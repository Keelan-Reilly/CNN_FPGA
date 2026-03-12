#!/usr/bin/env python3
"""Generate architecture-study plots from FPGA aggregate datasets."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from fpga_results import REPO_ROOT, load_aggregate, resolve_aggregate_path


METRIC_MAP = {
    "fmax": ("fmax_mhz_est", "Fmax (MHz)", "fmax_vs"),
    "lut": ("lut", "LUT", "lut_vs"),
    "ff": ("ff", "FF", "ff_vs"),
    "dsp": ("dsp", "DSP", "dsp_vs"),
    "bram": ("bram", "BRAM (tile)", "bram_vs"),
    "wns": ("wns_ns", "WNS (ns)", "wns_vs"),
    "latency_cycles": ("latency_cycles", "Latency (cycles)", "latency_cycles_vs"),
    "latency_time_ms": ("latency_time_ms", "Latency (ms)", "latency_time_ms_vs"),
    "throughput": ("throughput_inferences_per_sec", "Throughput (inf/s)", "throughput_vs"),
    "speedup": ("latency_speedup_vs_ref", "Speedup vs Reference", "speedup_vs"),
    "throughput_per_lut": ("throughput_per_lut", "Throughput / LUT", "throughput_per_lut_vs"),
    "throughput_per_dsp": ("throughput_per_dsp", "Throughput / DSP", "throughput_per_dsp_vs"),
}

STAGE_KEYS = [
    ("stage_cycles_conv", "Conv"),
    ("stage_cycles_relu", "ReLU"),
    ("stage_cycles_pool", "MaxPool"),
    ("stage_cycles_dense", "Dense"),
    ("stage_cycles_argmax", "Argmax"),
    ("bubble_cycles", "Bubble"),
]


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    try:
        return float(x)
    except Exception:
        return None


def _is_numeric_series(vals: list[Any]) -> bool:
    return all(_to_float(v) is not None for v in vals)


def _apply_filters(rows: list[dict[str, Any]], filters: list[str]) -> list[dict[str, Any]]:
    pairs = []
    for f in filters:
        if "=" not in f:
            raise ValueError(f"Invalid filter '{f}', expected KEY=VALUE")
        k, v = f.split("=", 1)
        pairs.append((k.strip(), v.strip()))

    out = []
    for r in rows:
        params = r.get("params", {})
        keep = True
        for k, v in pairs:
            if str(params.get(k)) != v:
                keep = False
                break
        if keep:
            out.append(r)
    return out


def _detect_x_param(rows: list[dict[str, Any]]) -> str:
    values: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        for k, v in r.get("params", {}).items():
            values[k].add(str(v))
    varying = sorted(k for k, v in values.items() if len(v) > 1)
    if not varying:
        raise ValueError("No varying parameter found. Pass --x-param explicitly.")
    return varying[0]


def _group_rows(rows: list[dict[str, Any]], group_by: str | None) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if group_by:
            label = str(r.get("params", {}).get(group_by, "(missing)"))
        else:
            label = "all"
        groups[label].append(r)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def _x_series(rows: list[dict[str, Any]], x_param: str) -> tuple[list[Any], bool] | None:
    pts = []
    for r in rows:
        x = r.get("params", {}).get(x_param)
        if x is None:
            continue
        pts.append(x)
    if not pts:
        return None
    return pts, _is_numeric_series(pts)


def _plot_metric_vs_x(
    rows: list[dict[str, Any]],
    x_param: str,
    group_by: str | None,
    metric_key: str,
    y_label: str,
    output_path: Path,
    plt: Any,
) -> bool:
    groups = _group_rows(rows, group_by)

    any_points = False
    plt.figure(figsize=(8, 4.8))

    for group_name, grows in groups.items():
        pts = []
        for r in grows:
            x = r.get("params", {}).get(x_param)
            y = _to_float(r.get(metric_key))
            if x is None or y is None:
                continue
            pts.append((x, y))

        if not pts:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        if _is_numeric_series(xs):
            pairs = sorted((float(x), y) for x, y in zip(xs, ys))
            xvals = [p[0] for p in pairs]
            yvals = [p[1] for p in pairs]
            plt.plot(xvals, yvals, marker="o", label=group_name)
        else:
            categories = sorted({str(x) for x in xs})
            idx = {c: i for i, c in enumerate(categories)}
            pairs = sorted((idx[str(x)], y) for x, y in zip(xs, ys))
            xvals = [p[0] for p in pairs]
            yvals = [p[1] for p in pairs]
            plt.plot(xvals, yvals, marker="o", label=group_name)
            plt.xticks(range(len(categories)), categories, rotation=20)

        any_points = True

    if not any_points:
        plt.close()
        return False

    plt.xlabel(x_param)
    plt.ylabel(y_label)
    title = f"{y_label} vs {x_param}"
    if group_by:
        title += f" (grouped by {group_by})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(groups) > 1:
        plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _plot_area_perf_scatter(
    rows: list[dict[str, Any]],
    area_key: str,
    area_label: str,
    group_by: str | None,
    out_path: Path,
    plt: Any,
) -> bool:
    groups = _group_rows(rows, group_by)
    any_points = False

    plt.figure(figsize=(6.8, 5.2))
    for group_name, grows in groups.items():
        xs = []
        ys = []
        for r in grows:
            x = _to_float(r.get(area_key))
            y = _to_float(r.get("fmax_mhz_est"))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        if not xs:
            continue
        plt.scatter(xs, ys, s=46, alpha=0.85, label=group_name)
        any_points = True

    if not any_points:
        plt.close()
        return False

    plt.xlabel(area_label)
    plt.ylabel("Fmax (MHz)")
    title = f"Area-Performance: Fmax vs {area_label}"
    if group_by:
        title += f" (grouped by {group_by})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(groups) > 1:
        plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _augment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    succeeded = [r for r in rows if _to_float(r.get("latency_cycles")) is not None]
    ref_latency = None
    if succeeded:
        ref_row = None
        for row in succeeded:
            if row.get("params", {}).get("DENSE_OUT_PAR") == 1:
                ref_row = row
                break
        if ref_row is None:
            ref_row = min(succeeded, key=lambda r: _to_float(r.get("latency_cycles")) or float("inf"))
        ref_latency = _to_float(ref_row.get("latency_cycles"))

    out = []
    for row in rows:
        r = dict(row)
        throughput = _to_float(r.get("throughput_inferences_per_sec"))
        lut = _to_float(r.get("lut"))
        dsp = _to_float(r.get("dsp"))
        latency = _to_float(r.get("latency_cycles"))
        if throughput is not None and lut not in (None, 0):
            r["throughput_per_lut"] = throughput / lut
        if throughput is not None and dsp not in (None, 0):
            r["throughput_per_dsp"] = throughput / dsp
        if ref_latency not in (None, 0) and latency not in (None, 0):
            r["latency_speedup_vs_ref"] = ref_latency / latency
        out.append(r)
    return out


def _plot_measured_vs_predicted_vs_x(
    rows: list[dict[str, Any]],
    x_param: str,
    measured_key: str,
    predicted_key: str,
    y_label: str,
    output_path: Path,
    plt: Any,
) -> bool:
    pts = []
    for r in rows:
        x = r.get("params", {}).get(x_param)
        y_meas = _to_float(r.get(measured_key))
        y_pred = _to_float(r.get(predicted_key))
        if x is None or y_meas is None or y_pred is None:
            continue
        pts.append((x, y_meas, y_pred))
    if not pts:
        return False

    xs = [p[0] for p in pts]
    numeric_x = _is_numeric_series(xs)
    if numeric_x:
        pts = sorted((float(x), y_meas, y_pred) for x, y_meas, y_pred in pts)
        xvals = [p[0] for p in pts]
    else:
        categories = sorted({str(x) for x in xs})
        idx = {c: i for i, c in enumerate(categories)}
        pts = sorted((idx[str(x)], y_meas, y_pred) for x, y_meas, y_pred in pts)
        xvals = [p[0] for p in pts]

    plt.figure(figsize=(8, 4.8))
    plt.plot(xvals, [p[1] for p in pts], marker="o", label="Measured")
    plt.plot(xvals, [p[2] for p in pts], marker="s", linestyle="--", label="Predicted")
    if not numeric_x:
        plt.xticks(range(len(categories)), categories, rotation=20)
    plt.xlabel(x_param)
    plt.ylabel(y_label)
    plt.title(f"Measured vs Predicted {y_label} vs {x_param}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _plot_stage_breakdown(
    rows: list[dict[str, Any]],
    x_param: str,
    output_path: Path,
    plt: Any,
) -> bool:
    pts = []
    for r in rows:
        x = r.get("params", {}).get(x_param)
        if x is None:
            continue
        vals = [_to_float(r.get(key)) for key, _ in STAGE_KEYS]
        if any(v is None for v in vals):
            continue
        pts.append((x, vals))
    if not pts:
        return False

    xs = [p[0] for p in pts]
    numeric_x = _is_numeric_series(xs)
    if numeric_x:
        pts = sorted((float(x), vals) for x, vals in pts)
        xvals = [p[0] for p in pts]
        xticklabels = None
    else:
        categories = sorted({str(x) for x in xs})
        idx = {c: i for i, c in enumerate(categories)}
        pts = sorted((idx[str(x)], vals) for x, vals in pts)
        xvals = [p[0] for p in pts]
        xticklabels = categories

    plt.figure(figsize=(8.4, 5.0))
    bottom = [0.0] * len(pts)
    width = 0.7 if not numeric_x else 0.8
    for stage_idx, (_, label) in enumerate(STAGE_KEYS):
        heights = [p[1][stage_idx] for p in pts]
        plt.bar(xvals, heights, bottom=bottom, label=label, width=width)
        bottom = [b + h for b, h in zip(bottom, heights)]

    if xticklabels is not None:
        plt.xticks(range(len(xticklabels)), xticklabels, rotation=20)
    plt.xlabel(x_param)
    plt.ylabel("Cycles")
    plt.title(f"Stage Cycle Breakdown vs {x_param}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _plot_model_decomposition(
    rows: list[dict[str, Any]],
    x_param: str,
    output_path: Path,
    plt: Any,
) -> bool:
    pts = []
    for r in rows:
        x = r.get("params", {}).get(x_param)
        fixed = _to_float(r.get("model_fixed_cycles"))
        dense = _to_float(r.get("model_dense_cycles"))
        if x is None or fixed is None or dense is None:
            continue
        pts.append((x, fixed, dense))
    if not pts:
        return False

    xs = [p[0] for p in pts]
    numeric_x = _is_numeric_series(xs)
    if numeric_x:
        pts = sorted((float(x), fixed, dense) for x, fixed, dense in pts)
        xvals = [p[0] for p in pts]
        xticklabels = None
    else:
        categories = sorted({str(x) for x in xs})
        idx = {c: i for i, c in enumerate(categories)}
        pts = sorted((idx[str(x)], fixed, dense) for x, fixed, dense in pts)
        xvals = [p[0] for p in pts]
        xticklabels = categories

    fixed_vals = [p[1] for p in pts]
    dense_vals = [p[2] for p in pts]

    plt.figure(figsize=(8.0, 4.8))
    plt.bar(xvals, fixed_vals, label="Fixed term")
    plt.bar(xvals, dense_vals, bottom=fixed_vals, label="Dense term")
    if xticklabels is not None:
        plt.xticks(range(len(xticklabels)), xticklabels, rotation=20)
    plt.xlabel(x_param)
    plt.ylabel("Predicted cycles")
    plt.title(f"Model Latency Decomposition vs {x_param}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _plot_speedup_vs_area(
    rows: list[dict[str, Any]],
    area_key: str,
    area_label: str,
    out_path: Path,
    plt: Any,
) -> bool:
    xs = []
    ys = []
    for r in rows:
        area = _to_float(r.get(area_key))
        speedup = _to_float(r.get("latency_speedup_vs_ref"))
        if area is None or speedup is None:
            continue
        xs.append(area)
        ys.append(speedup)
    if not xs:
        return False

    plt.figure(figsize=(6.8, 5.2))
    plt.scatter(xs, ys, s=50, alpha=0.85)
    plt.xlabel(area_label)
    plt.ylabel("Speedup vs Reference")
    plt.title(f"Speedup vs {area_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate FPGA architecture-study plots")
    ap.add_argument("--experiment-id", default="baseline_fpga", help="Experiment id (default: baseline_fpga)")
    ap.add_argument("--aggregate", default=None, help="Path to aggregate .json/.csv")
    ap.add_argument("--x-param", default=None, help="Parameter name on x-axis")
    ap.add_argument("--group-by", default=None, help="Optional grouping parameter")
    ap.add_argument("--filter", action="append", default=[], help="Filter by param KEY=VALUE")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in this environment, then rerun."
        )

    try:
        aggregate_path = resolve_aggregate_path(args.experiment_id, args.aggregate)
        experiment_id, rows = load_aggregate(aggregate_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Could not load aggregate dataset: {exc}")

    if not args.include_failed:
        rows = [r for r in rows if r.get("status") == "succeeded"]

    try:
        rows = _apply_filters(rows, args.filter)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not rows:
        raise SystemExit("No rows left after status/filter selection.")

    rows = _augment_rows(rows)

    x_param = args.x_param or _detect_x_param(rows)

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "results" / "fpga" / "plots" / experiment_id
    )

    generated: list[Path] = []

    for _, (metric_key, metric_label, prefix) in METRIC_MAP.items():
        out = out_dir / f"{prefix}_{x_param}.png"
        if _plot_metric_vs_x(rows, x_param, args.group_by, metric_key, metric_label, out, plt):
            generated.append(out)

    for area_key, area_label in [("lut", "LUT"), ("ff", "FF"), ("dsp", "DSP"), ("bram", "BRAM (tile)")]:
        out = out_dir / f"area_perf_fmax_vs_{area_key}.png"
        if _plot_area_perf_scatter(rows, area_key, area_label, args.group_by, out, plt):
            generated.append(out)

    out = out_dir / f"stage_cycles_breakdown_vs_{x_param}.png"
    if _plot_stage_breakdown(rows, x_param, out, plt):
        generated.append(out)

    out = out_dir / f"measured_vs_predicted_latency_vs_{x_param}.png"
    if _plot_measured_vs_predicted_vs_x(
        rows, x_param, "latency_cycles", "predicted_latency_cycles", "Latency (cycles)", out, plt
    ):
        generated.append(out)

    out = out_dir / f"measured_vs_predicted_throughput_vs_{x_param}.png"
    if _plot_measured_vs_predicted_vs_x(
        rows,
        x_param,
        "throughput_inferences_per_sec",
        "predicted_throughput_inferences_per_sec",
        "Throughput (inf/s)",
        out,
        plt,
    ):
        generated.append(out)

    out = out_dir / f"model_latency_decomposition_vs_{x_param}.png"
    if _plot_model_decomposition(rows, x_param, out, plt):
        generated.append(out)

    for area_key, area_label in [("lut", "LUT"), ("dsp", "DSP")]:
        out = out_dir / f"speedup_vs_{area_key}.png"
        if _plot_speedup_vs_area(rows, area_key, area_label, out, plt):
            generated.append(out)

    if not generated:
        raise SystemExit("No plots generated (missing numeric metrics after filtering).")

    print(f"experiment_id: {experiment_id}")
    print(f"aggregate: {aggregate_path}")
    print(f"x_param: {x_param}")
    print(f"rows_used: {len(rows)}")
    print(f"plots_dir: {out_dir}")
    for p in generated:
        print(f"- {p}")

    print("Note: throughput/latency/model-vs-measurement plots are not generated because those metrics are not in current aggregates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
