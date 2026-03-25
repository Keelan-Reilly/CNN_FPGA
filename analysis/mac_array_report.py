#!/usr/bin/env python3
"""Report-generation helpers for the MAC-array evaluation framework."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

try:
    from .fpga_results import REPO_ROOT, load_aggregate, resolve_aggregate_path
except ImportError:
    from fpga_results import REPO_ROOT, load_aggregate, resolve_aggregate_path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_single_run(experiment_id: str) -> dict[str, Any]:
    _, rows = load_aggregate(resolve_aggregate_path(experiment_id, None))
    succeeded = [row for row in rows if row.get("status") == "succeeded"]
    if not succeeded:
        raise ValueError(f"No succeeded rows found in {experiment_id}")
    return succeeded[0]


def _extract_dense_scaling_rows() -> list[dict[str, Any]]:
    _, rows = load_aggregate(resolve_aggregate_path("study_dense_parallel_scaling", None))
    return [row for row in rows if row.get("status") == "succeeded"]


def current_measured_summary() -> dict[str, Any]:
    baseline = _extract_single_run("study_baseline_characterization")
    dense_rows = _extract_dense_scaling_rows()
    best_dense = max(dense_rows, key=lambda row: row.get("throughput_inferences_per_sec") or 0.0)

    direct_tradeoff_rows = []
    direct_tradeoff_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "direct_tradeoff_measured_vs_modelled.json"
    if direct_tradeoff_path.exists():
        direct_tradeoff_rows = json.loads(direct_tradeoff_path.read_text())

    direct_tradeoff_summary = None
    if direct_tradeoff_rows:
        direct_tradeoff_summary = direct_tradeoff_rows[0]

    direct_shared_summary_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "direct_shared_implementation_summary.json"
    direct_shared_summary = None
    if direct_shared_summary_path.exists():
        direct_shared_summary = json.loads(direct_shared_summary_path.read_text())

    measured_support_map_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "measured_support_map.json"
    measured_support_map = []
    if measured_support_map_path.exists():
        measured_support_map = json.loads(measured_support_map_path.read_text())

    measured_trust_summary_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "measured_trust_summary.json"
    measured_trust_summary = None
    if measured_trust_summary_path.exists():
        measured_trust_summary = json.loads(measured_trust_summary_path.read_text())

    framework_calibration_aid_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "framework_calibration_aid.json"
    framework_calibration_aid = []
    if framework_calibration_aid_path.exists():
        framework_calibration_aid = json.loads(framework_calibration_aid_path.read_text())

    framework_calibration_overlay_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "framework_calibration_overlay.json"
    framework_calibration_overlay = []
    if framework_calibration_overlay_path.exists():
        framework_calibration_overlay = json.loads(framework_calibration_overlay_path.read_text())

    shared_family_calibration_summary_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "shared_family_calibration_summary.json"
    shared_family_calibration_summary = None
    if shared_family_calibration_summary_path.exists():
        shared_family_calibration_summary = json.loads(shared_family_calibration_summary_path.read_text())

    measured_utility_summary_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "measured_utility_summary.json"
    measured_utility_summary = None
    if measured_utility_summary_path.exists():
        measured_utility_summary = json.loads(measured_utility_summary_path.read_text())

    measured_bottleneck_choice_map_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "measured_bottleneck_choice_map.json"
    measured_bottleneck_choice_map = []
    if measured_bottleneck_choice_map_path.exists():
        measured_bottleneck_choice_map = json.loads(measured_bottleneck_choice_map_path.read_text())

    measured_design_rule_extraction_summary_path = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice" / "measured_design_rule_extraction_summary.json"
    measured_design_rule_extraction_summary = None
    if measured_design_rule_extraction_summary_path.exists():
        measured_design_rule_extraction_summary = json.loads(measured_design_rule_extraction_summary_path.read_text())

    return {
        "source_datasets": [
            "results/fpga/aggregates/study_baseline_characterization.json",
            "results/fpga/aggregates/study_dense_parallel_scaling.json",
        ],
        "baseline_characterization": {
            "latency_cycles": baseline.get("latency_cycles"),
            "throughput_inferences_per_sec": baseline.get("throughput_inferences_per_sec"),
            "lut": baseline.get("lut"),
            "dsp": baseline.get("dsp"),
            "wns_ns": baseline.get("wns_ns"),
            "stage_cycles_conv": baseline.get("stage_cycles_conv"),
            "stage_cycles_dense": baseline.get("stage_cycles_dense"),
        },
        "dense_parallel_scaling": {
            "best_throughput_inferences_per_sec": best_dense.get("throughput_inferences_per_sec"),
            "best_run_id": best_dense.get("run_id"),
            "runs": [
                {
                    "run_id": row.get("run_id"),
                    "dense_out_par": (row.get("params") or {}).get("DENSE_OUT_PAR"),
                    "latency_cycles": row.get("latency_cycles"),
                    "throughput_inferences_per_sec": row.get("throughput_inferences_per_sec"),
                    "lut": row.get("lut"),
                    "dsp": row.get("dsp"),
                    "wns_ns": row.get("wns_ns"),
                }
                for row in dense_rows
            ],
        },
        "direct_tradeoff": {
            "source_path": str(direct_tradeoff_path.relative_to(REPO_ROOT)) if direct_tradeoff_path.exists() else None,
            "measured_pairs": len(direct_tradeoff_rows),
            "first_pair": direct_tradeoff_summary,
        },
        "direct_shared_summary": {
            "source_path": str(direct_shared_summary_path.relative_to(REPO_ROOT)) if direct_shared_summary_path.exists() else None,
            "summary": direct_shared_summary,
        },
        "measured_support_map": {
            "source_path": str(measured_support_map_path.relative_to(REPO_ROOT)) if measured_support_map_path.exists() else None,
            "rows": measured_support_map,
        },
        "measured_trust_summary": {
            "source_path": str(measured_trust_summary_path.relative_to(REPO_ROOT)) if measured_trust_summary_path.exists() else None,
            "summary": measured_trust_summary,
        },
        "framework_calibration_aid": {
            "source_path": str(framework_calibration_aid_path.relative_to(REPO_ROOT)) if framework_calibration_aid_path.exists() else None,
            "rows": framework_calibration_aid,
        },
        "framework_calibration_overlay": {
            "source_path": str(framework_calibration_overlay_path.relative_to(REPO_ROOT)) if framework_calibration_overlay_path.exists() else None,
            "rows": framework_calibration_overlay,
        },
        "shared_family_calibration_summary": {
            "source_path": str(shared_family_calibration_summary_path.relative_to(REPO_ROOT)) if shared_family_calibration_summary_path.exists() else None,
            "summary": shared_family_calibration_summary,
        },
        "measured_utility_summary": {
            "source_path": str(measured_utility_summary_path.relative_to(REPO_ROOT)) if measured_utility_summary_path.exists() else None,
            "summary": measured_utility_summary,
        },
        "measured_bottleneck_choice_map": {
            "source_path": str(measured_bottleneck_choice_map_path.relative_to(REPO_ROOT)) if measured_bottleneck_choice_map_path.exists() else None,
            "rows": measured_bottleneck_choice_map,
        },
        "measured_design_rule_extraction_summary": {
            "source_path": str(measured_design_rule_extraction_summary_path.relative_to(REPO_ROOT)) if measured_design_rule_extraction_summary_path.exists() else None,
            "summary": measured_design_rule_extraction_summary,
        },
    }


def render_markdown_report(
    output_path: Path,
    config: dict[str, Any],
    measured_summary: dict[str, Any],
    provenance_rows: list[dict[str, Any]],
    static_rows: list[dict[str, Any]],
    workload_rows: list[dict[str, Any]],
    adaptive_rows: list[dict[str, Any]],
    break_even_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    policy_diagnostics: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    regime_summary_rows: list[dict[str, Any]],
    regime_meta: dict[str, Any],
    regime_rejection_rows: list[dict[str, Any]],
    rejection_surface_rows: list[dict[str, Any]],
    regime_insight_rows: list[dict[str, Any]],
) -> None:
    ensure_dir(output_path.parent)

    strongest_policy = []
    seen_recommendations: set[str] = set()
    for row in policy_rows:
        rec = row["recommendation"]
        if rec not in seen_recommendations:
            strongest_policy.append(row)
            seen_recommendations.add(rec)
    if len(strongest_policy) < 4:
        for row in policy_rows:
            if row not in strongest_policy:
                strongest_policy.append(row)
            if len(strongest_policy) >= 4:
                break

    adaptive_wins = [row for row in policy_rows if row["recommendation"] == "adaptive_mode_switching"]
    rejection_counts: dict[str, int] = {}
    for row in policy_diagnostics:
        for reason in row.get("rejected_reasons", []):
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    regime_winners = ", ".join(
        f"{row['grid']}:{row['winner']}={row['count']}" for row in regime_summary_rows
    )
    dominant_blockers = ", ".join(
        f"{row['grid']}:{row['workload']}={row['dominant_rejection_reason']}"
        for row in sorted(rejection_surface_rows, key=lambda item: (item["grid"], item["workload"]))
    )
    top_rejection_rows = sorted(
        regime_rejection_rows,
        key=lambda item: (-item["count"], item["grid"], item["workload"], item["rejection_reason"]),
    )[:6]

    shared_8x8 = next(
        row for row in static_rows if row["grid"] == "8x8" and row["architecture"] == "shared"
    )
    baseline_8x8 = next(
        row for row in static_rows if row["grid"] == "8x8" and row["architecture"] == "baseline"
    )
    replicated_8x8 = next(
        row for row in static_rows if row["grid"] == "8x8" and row["architecture"] == "replicated"
    )

    lines = [
        "# MAC-Array Framework V2 Results Pack",
        "",
        "## Framing",
        "",
        "This results pack preserves the repo's checked-in measured FPGA study and extends it with a workload-aware, constraint-aware MAC-array decision layer.",
        "",
        "Measured evidence preserved from the current repo:",
        f"- CNN baseline characterization remains sourced from `{measured_summary['source_datasets'][0]}`.",
        f"- Dense-parallel scaling remains sourced from `{measured_summary['source_datasets'][1]}`.",
        f"- Canonical MAC-array static evidence now loads from `{config['evidence_path']}`.",
        "",
        "Static architecture evidence integrated into this framework:",
        f"- 8x8 shared uses `{shared_8x8['dsp']}` DSP versus `{baseline_8x8['dsp']}` DSP for baseline.",
        f"- That DSP anchor is carried as `{shared_8x8['dsp_provenance_kind']}` via source id `{shared_8x8['dsp_source_id']}`.",
        f"- 8x8 replicated is marked `{replicated_8x8['implementation_status']}`, preserving implementation failure as meaningful evidence.",
        "",
        "## Measured vs Modelled",
        "",
        "- Measured: current checked-in CNN baseline and dense-parallel sweep aggregates.",
        "- Direct measured MAC slice: baseline calibration points plus directly measured 4x4 and 8x4 shared implementations when the direct slice artifacts are present.",
        "- Framework shared rows refer to the modelled shared family `shared_modelled_dsp_reducing`; the direct shared slices are separate implementation-specific measured observations.",
        "- Anchored: prior MAC-array static facts explicitly carried in the evidence registry.",
        "- Modelled: workload classes, utilization-aware latency/throughput estimates, adaptive switching costs, break-even thresholds, and policy recommendations.",
        "",
        "## Provenance Summary",
        "",
    ]

    for row in provenance_rows:
        lines.append(f"- `{row['value_kind']}`: {row['count']} records from `{row['source_path']}`")

    direct_tradeoff = measured_summary.get("direct_tradeoff", {})
    direct_tradeoff_pair = direct_tradeoff.get("first_pair")
    direct_tradeoff_note = None
    direct_shared_summary_wrapper = measured_summary.get("direct_shared_summary", {})
    direct_shared_summary = direct_shared_summary_wrapper.get("summary")
    trust_summary_wrapper = measured_summary.get("measured_trust_summary", {})
    trust_summary = trust_summary_wrapper.get("summary")
    support_map_wrapper = measured_summary.get("measured_support_map", {})
    support_map_rows = support_map_wrapper.get("rows") or []
    calibration_summary_wrapper = measured_summary.get("shared_family_calibration_summary", {})
    calibration_summary = calibration_summary_wrapper.get("summary")
    calibration_overlay_wrapper = measured_summary.get("framework_calibration_overlay", {})
    calibration_overlay_rows = calibration_overlay_wrapper.get("rows") or []
    utility_summary_wrapper = measured_summary.get("measured_utility_summary", {})
    utility_summary = utility_summary_wrapper.get("summary")
    bottleneck_map_wrapper = measured_summary.get("measured_bottleneck_choice_map", {})
    bottleneck_map_rows = bottleneck_map_wrapper.get("rows") or []
    design_rule_summary_wrapper = measured_summary.get("measured_design_rule_extraction_summary", {})
    design_rule_summary = design_rule_summary_wrapper.get("summary")
    direct_shared_notes: list[str] = []
    trust_notes: list[str] = []
    calibration_notes: list[str] = []
    utility_notes: list[str] = []
    design_rule_notes: list[str] = []
    if direct_tradeoff_pair:
        direct_tradeoff_note = (
            f"- Direct measured bridge now covers the isolated slice at 4x4 and 8x4, with each shared point anchored back to the measured baseline schedule and resource point for its grid."
        )
    if direct_shared_summary:
        direct_shared_notes.append(f"- {direct_shared_summary['headline']}")
        for line in direct_shared_summary.get('summary_lines', []):
            direct_shared_notes.append(f"- {line}")
    if trust_summary:
        trust_notes.append(f"- {trust_summary['headline']}")
        for line in trust_summary.get('summary_lines', []):
            trust_notes.append(f"- {line}")
    if calibration_summary:
        calibration_notes.append(f"- {calibration_summary['headline']}")
        for line in calibration_summary.get("summary_lines", []):
            calibration_notes.append(f"- {line}")
    if utility_summary:
        utility_notes.append(f"- {utility_summary['headline']}")
        for line in utility_summary.get("summary_lines", []):
            utility_notes.append(f"- {line}")
    if design_rule_summary:
        design_rule_notes.append(f"- {design_rule_summary['headline']}")
        for line in design_rule_summary.get("summary_lines", []):
            design_rule_notes.append(f"- {line}")
    for row in bottleneck_map_rows[:5]:
        utility_notes.append(f"- `{row['grid']}` / `{row['bottleneck_kind']}` -> `{row['preferred_variant']}`: {row['decision_basis']}")
    for row in calibration_overlay_rows[:4]:
        calibration_notes.append(f"- `{row['overlay_topic']}` -> `{row['calibration_status']}`: {row['calibration_reading']}")
    for row in support_map_rows[:4]:
        trust_notes.append(f"- `{row['claim_id']}` -> `{row['support_level']}`: {row['trust_note']}")

    lines.extend(
        [
            "",
            "## Strongest Insights",
            "",
            f"- The framework shared family currently means the modelled variant `{shared_8x8['architecture_variant_id']}`, not the directly measured 4x4 and 8x4 shared implementations.",
            f"- Shared 8x8 keeps the anchored prior-study DSP reduction from `{baseline_8x8['dsp']}` to `{shared_8x8['dsp']}`, but that anchor now sits alongside measured 4x4 and 8x4 shared implementations that separate LUT-oriented sharing from DSP-oriented sharing on Artix-7.",
            *([direct_tradeoff_note] if direct_tradeoff_note else []),
            *direct_shared_notes,
            *trust_notes,
            *calibration_notes,
            *utility_notes,
            *design_rule_notes,
            f"- Replicated 8x8 remains excluded on Artix-7 because the framework preserves the `{replicated_8x8['implementation_status']}` evidence rather than treating the missing implementation as neutral.",
            f"- Adaptive mode switching wins in `{len(adaptive_wins)}` policy scenarios; under the tighter mode-pair-aware switching model it is currently a conservative option rather than a default recommendation.",
            f"- The bounded regime map covers `{regime_meta['regime_points']}` points; adaptive win region present: `{regime_meta['adaptive_has_win_region']}`.",
            f"- Regime winner counts by grid are: {regime_winners}.",
            f"- The most common policy rejection reasons are: {', '.join(f'{k}={v}' for k, v in sorted(rejection_counts.items()))}.",
            f"- Adaptive blocker surface by grid/workload is: {dominant_blockers}.",
            "",
            "## Regime Result",
            "",
            f"- Bounded regime dimensions are `grid x workload x budget_class x throughput_class` = `{regime_meta['regime_points']}` points.",
            "- `budget_class` comes from the evaluated architecture resource rows: `tight_shared_model_only`, `baseline_fit`, and `expanded_headroom`.",
            "- `throughput_class` comes from each workload/grid throughput envelope: `efficiency_oriented`, `balanced`, and `stretch`.",
            f"- Adaptive win region present: `{regime_meta['adaptive_has_win_region']}`.",
            "",
            "Top derived regime insights:",
            "",
        ]
    )

    for row in regime_insight_rows[:8]:
        lines.append(f"- {row['note']}")

    lines.extend(
        [
            "",
            "Dominant adaptive rejection slices:",
            "",
        ]
    )

    for row in top_rejection_rows:
        lines.append(
            f"- `{row['grid']}` / `{row['workload']}` -> `{row['rejection_reason']}` in {row['count']} points "
            f"({row['share_of_regime_points']:.2%} of that slice)."
        )

    lines.extend(
        [
            "",
            "## Policy Snapshot",
            "",
        ]
    )

    for row in strongest_policy:
        recommendation_label = (
            f"{row['recommendation']} (modelled family; read through calibration aid)"
            if row["recommendation"] == "shared"
            else row["recommendation"]
        )
        runner_up = row.get("runner_up", "")
        runner_up_label = (
            f"{runner_up} (modelled family; read through calibration aid)"
            if runner_up == "shared"
            else runner_up
        )
        lines.append(
            f"- `{row['constraint']}` / `{row['workload']}` / `{row['grid']}` -> `{recommendation_label}` over `{runner_up_label}`: {row['reason']}"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `evidence_registry.csv/json`: first-class evidence records for architecture and switching assumptions.",
            "- `provenance_summary.csv/json`: count summary of evidence/value kinds.",
            "- `workload_manifest.csv/json`: explicit workload-class descriptors and phase traits.",
            "- `static_architectures.csv/json`: canonical 3x3 architecture-grid evidence table.",
            "- `workload_evaluations.csv/json`: fixed-mode workload metrics with throughput/utilization/efficiency.",
            "- `adaptive_evaluations.csv/json`: switching-adjusted throughput and per-phase mode selections.",
            "- `adaptive_constraint_evaluations.csv/json`: constraint-filtered adaptive candidates used by the policy layer.",
            "- `adaptive_phase_decisions.csv/json`: phase-level mode choices, pair transitions, and alternative latencies.",
            "- `break_even.csv/json`: minimum phase duration before switching pays off.",
            "- `policy_recommendations.csv/json`: explicit selection-policy outputs under preset constraints.",
            "- `policy_diagnostics.csv/json`: candidate-by-candidate feasibility and rejection reasons.",
            "- `regime_map.csv/json`: bounded winner map across grid/workload/budget/throughput regimes.",
            "- `regime_summary.csv/json`: compact regime winner counts and adaptive-region summary.",
            "- `regime_rejection_summary.csv/json`: adaptive rejection counts by grid/workload/reason.",
            "- `adaptive_rejection_surface.csv/json`: dominant adaptive blocker by grid/workload slice.",
            "- `regime_insights.csv/json` and `regime_insights.md`: derived winner/blocker summaries from the generated regime map.",
            "- `measured_refresh/`: optional selective measured-refresh manifest, single-run queue configs, and proxy comparison outputs.",
            "- `measured_support_map.csv/json`: machine-readable support levels for measured implementation roles and family-level shared claims.",
            "- `framework_trust_overlay.csv/json`: compact trust overlay for reading framework shared conclusions against measured support limits.",
            "- `measured_vs_modelled_trust_summary.md/json`: concise trust-boundary summary for what is directly supported, directionally supported, or extrapolated beyond measured support.",
            "- `framework_calibration_aid.csv/json`: machine-readable calibration aid for reading family-level framework resource expectations through direct-slice evidence.",
            "- `framework_calibration_overlay.csv/json`: compact calibration overlay marking where family-level numeric expectations are aligned, optimistic, or implementation-dependent.",
            "- `shared_family_calibration_summary.md/json`: concise calibration summary for how measured slice evidence should bound shared-family numeric interpretation.",
            "- `measured_utility_table.csv/json`: compact measured utility rows for baseline, shared_lut_saving, and shared_dsp_reducing at each measured grid.",
            "- `measured_bottleneck_choice_map.csv/json`: bounded measured choice map for LUT, DSP, performance, timing-margin, and no-bottleneck cases.",
            "- `measured_utility_summary.md/json`: concise measured design-utility summary for when shared relief is actually worth the performance cost.",
            "- `measured_flexibility_overhead_table.csv/json`: measured overhead rows quantifying what flexibility costs each shared implementation against baseline.",
            "- `measured_flexibility_justification_table.csv/json`: bounded measured table for when flexibility is justified and when baseline should be preferred.",
            "- `measured_design_rule_extraction_summary.md/json`: thesis-style measured design-rule conclusion for when flexibility is justified versus when it is just overhead.",
            "- `direct_slice/`: direct measured-vs-modelled baseline calibration outputs, direct shared scaling outputs, measured support map rows, measured trust summaries, and direct calibration aids.",
            "",
            "## Limitations",
            "",
            "- Adaptive results are analytical pre-built-mode estimates, not partial reconfiguration measurements.",
            "- LUT and most WNS fields remain analytical trend estimates unless explicitly anchored or directly measured on the isolated direct slice.",
            "- This framework is intentionally lightweight and does not run Vivado synthesis or place-and-route.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def render_progress_log(output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(
        "\n".join(
            [
                "# Framework V2 Progress Log",
                "",
                "- Preserved the existing measured FPGA CNN study under `results/fpga/aggregates/` as the baseline evidence layer.",
                "- Added a first-class architecture evidence registry and provenance summary outputs.",
                "- Added explicit workload classes, richer normalized metrics, adaptive switching analysis, break-even calculations, and a rules-based selection policy.",
                "- Kept measured and modelled quantities separate in generated outputs so the repo does not over-claim adaptive hardware data.",
                "- Generated a reproducible v2 results pack under `results/fpga/framework_v2/` with policy diagnostics and adaptive phase decisions.",
                "",
            ]
        )
        + "\n"
    )


def render_regime_insights_markdown(
    output_path: Path,
    regime_meta: dict[str, Any],
    regime_summary_rows: list[dict[str, Any]],
    rejection_surface_rows: list[dict[str, Any]],
    regime_insight_rows: list[dict[str, Any]],
) -> None:
    ensure_dir(output_path.parent)
    lines = [
        "# Regime Insights",
        "",
        f"- Regime points evaluated: `{regime_meta['regime_points']}`.",
        f"- Adaptive win region present: `{regime_meta['adaptive_has_win_region']}`.",
        "",
        "## Winner Structure",
        "",
    ]
    for row in sorted(regime_summary_rows, key=lambda item: (item["grid"], -item["count"], item["winner"])):
        lines.append(f"- `{row['grid']}` -> `{row['winner']}` wins `{row['count']}` points.")

    lines.extend(["", "## Adaptive Blockers", ""])
    for row in sorted(rejection_surface_rows, key=lambda item: (item["grid"], item["workload"])):
        lines.append(
            f"- `{row['grid']}` / `{row['workload']}`: dominant blocker `{row['dominant_rejection_reason']}` "
            f"across `{row['adaptive_candidate_points']}` adaptive-candidate points."
        )

    lines.extend(["", "## Derived Insights", ""])
    for row in regime_insight_rows:
        lines.append(f"- {row['note']}")
    output_path.write_text("\n".join(lines) + "\n")


def render_plots(
    output_dir: Path,
    workload_rows: list[dict[str, Any]],
    break_even_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]] | None = None,
    rejection_surface_rows: list[dict[str, Any]] | None = None,
) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cnn_fpga_mplconfig")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    ensure_dir(output_dir)
    generated: list[Path] = []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in workload_rows:
        grouped.setdefault(row["workload"], []).append(row)

    path = output_dir / "workload_throughput.png"
    plt.figure(figsize=(8.2, 4.8))
    for workload, rows in grouped.items():
        rows = sorted(rows, key=lambda item: (item["grid"], item["architecture"]))
        labels = [f"{row['grid']}:{row['architecture'][0].upper()}" for row in rows]
        values = [row["effective_throughput_ops_per_cycle"] for row in rows]
        plt.plot(labels, values, marker="o", label=workload)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Throughput (ops/cycle)")
    plt.title("Workload-Aware Throughput by Architecture")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    generated.append(path)

    path = output_dir / "break_even_switch_cycles.png"
    selected = [row for row in break_even_rows if row["min_phase_cycles_for_break_even"] is not None]
    if selected:
        plt.figure(figsize=(8.2, 4.8))
        labels = [f"{row['workload']}:{row['phase']}:{row['from_mode']}->{row['to_mode']}" for row in selected]
        values = [row["min_phase_cycles_for_break_even"] for row in selected]
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Min phase cycles")
        plt.title("Break-Even Duration for Switching")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(path)

    if regime_rows:
        counts: dict[str, int] = {}
        for row in regime_rows:
            winner = row["winner"]
            counts[winner] = counts.get(winner, 0) + 1
        path = output_dir / "regime_winner_counts.png"
        plt.figure(figsize=(7.0, 4.6))
        labels = list(counts.keys())
        values = [counts[label] for label in labels]
        plt.bar(labels, values)
        plt.ylabel("Regime points")
        plt.title("Winner Map Counts by Strategy")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(path)

        strategy_codes = {
            "baseline": 0,
            "shared": 1,
            "replicated": 2,
            "adaptive_mode_switching": 3,
        }
        strategy_labels = ["B", "S", "R", "A"]
        workload_order = sorted({row["workload"] for row in regime_rows})
        grid_order = sorted({row["grid"] for row in regime_rows})
        budget_order = ["tight_shared_model_only", "baseline_fit", "expanded_headroom"]
        throughput_order = ["efficiency_oriented", "balanced", "stretch"]
        cmap = plt.get_cmap("tab10", len(strategy_codes))
        fig, axes = plt.subplots(len(grid_order), len(workload_order), figsize=(12.5, 7.8), squeeze=False)
        for row_idx, grid in enumerate(grid_order):
            for col_idx, workload in enumerate(workload_order):
                ax = axes[row_idx][col_idx]
                matrix = []
                for budget in budget_order:
                    current_row = []
                    for throughput in throughput_order:
                        match = next(
                            item
                            for item in regime_rows
                            if item["grid"] == grid
                            and item["workload"] == workload
                            and item["budget_class"] == budget
                            and item["throughput_class"] == throughput
                        )
                        current_row.append(strategy_codes[match["winner"]])
                    matrix.append(current_row)
                ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(strategy_codes) - 1)
                for y in range(len(budget_order)):
                    for x in range(len(throughput_order)):
                        value = matrix[y][x]
                        ax.text(
                            x,
                            y,
                            strategy_labels[value],
                            ha="center",
                            va="center",
                            color="white" if value in {0, 2, 3} else "black",
                            fontsize=8,
                            fontweight="bold",
                        )
                ax.set_title(f"{grid} / {workload}", fontsize=9)
                ax.set_xticks(range(len(throughput_order)))
                ax.set_xticklabels(["eff", "bal", "str"], fontsize=8)
                ax.set_yticks(range(len(budget_order)))
                ax.set_yticklabels(["tight", "fit", "rel"], fontsize=8)
        fig.suptitle("Winner Surface by Grid, Workload, Budget, and Throughput", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = output_dir / "regime_winner_heatmaps.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    if rejection_surface_rows:
        reason_codes = {
            "adaptive_gain_too_small": 0,
            "workload_not_phase_changing": 1,
            "no_switching_observed": 2,
            "below_throughput_target": 3,
            "over_dsp_budget": 4,
            "over_lut_budget": 5,
            "timing_infeasible": 6,
            "no_rejection_recorded": 7,
        }
        reason_labels = {
            "adaptive_gain_too_small": "gain",
            "workload_not_phase_changing": "phase",
            "no_switching_observed": "nosw",
            "below_throughput_target": "thr",
            "over_dsp_budget": "dsp",
            "over_lut_budget": "lut",
            "timing_infeasible": "timing",
            "no_rejection_recorded": "n/a",
        }
        workload_order = sorted({row["workload"] for row in rejection_surface_rows})
        grid_order = sorted({row["grid"] for row in rejection_surface_rows})
        cmap = plt.get_cmap("tab20", len(reason_codes))
        fig, axes = plt.subplots(1, len(grid_order), figsize=(12.0, 4.0), squeeze=False)
        for idx, grid in enumerate(grid_order):
            ax = axes[0][idx]
            rows = sorted(
                [row for row in rejection_surface_rows if row["grid"] == grid],
                key=lambda item: item["workload"],
            )
            matrix = [[reason_codes[row["dominant_rejection_reason"]]] for row in rows]
            ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(reason_codes) - 1, aspect="auto")
            for y, row in enumerate(rows):
                ax.text(
                    0,
                    y,
                    reason_labels[row["dominant_rejection_reason"]],
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
            ax.set_title(f"{grid} adaptive blocker", fontsize=10)
            ax.set_xticks([0])
            ax.set_xticklabels(["dominant"], fontsize=8)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([row["workload"] for row in rows], fontsize=8)
        fig.suptitle("Adaptive Rejection Surface by Grid and Workload", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = output_dir / "adaptive_rejection_surface.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    return generated
