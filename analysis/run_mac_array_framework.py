#!/usr/bin/env python3
"""Run the MAC-array evaluation framework and emit a v2 results pack."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from .mac_array_evidence import load_evidence, provenance_summary_rows
    from .fpga_results import REPO_ROOT
    from .mac_array_adaptive import build_break_even_rows, evaluate_adaptive_workload
    from .mac_array_direct_slice import (
        build_direct_slice_comparison_rows,
        build_direct_tradeoff_rows,
        build_framework_calibration_aid_rows,
        build_framework_calibration_overlay_rows,
        build_framework_trust_overlay_rows,
        build_measured_bottleneck_choice_map,
        build_measured_design_rule_extraction_summary,
        build_measured_flexibility_justification_table,
        build_measured_flexibility_overhead_rows,
        build_measured_support_rows,
        build_measured_trust_summary,
        build_measured_utility_rows,
        build_measured_utility_summary,
        build_shared_family_calibration_summary,
        render_measured_vs_modelled_trust_summary,
        render_measured_utility_summary,
        render_shared_family_calibration_summary,
        render_measured_design_rule_extraction_summary,
    )
    from .mac_array_metrics import derive_static_row, summarize_workload
    from .mac_array_policy import evaluate_policy
    from .mac_array_regime import (
        build_adaptive_rejection_surface,
        build_rejection_summary,
        build_regime_insights,
        build_regime_map,
    )
    from .mac_array_report import (
        current_measured_summary,
        render_markdown_report,
        render_regime_insights_markdown,
        render_plots,
        render_progress_log,
        write_csv,
        write_json,
    )
    from .mac_array_types import ConstraintSpec, GridSpec
    from .mac_array_workloads import workloads_from_config
except ImportError:
    from mac_array_evidence import load_evidence, provenance_summary_rows
    from fpga_results import REPO_ROOT
    from mac_array_adaptive import build_break_even_rows, evaluate_adaptive_workload
    from mac_array_direct_slice import (
        build_direct_slice_comparison_rows,
        build_direct_tradeoff_rows,
        build_framework_calibration_aid_rows,
        build_framework_calibration_overlay_rows,
        build_framework_trust_overlay_rows,
        build_measured_bottleneck_choice_map,
        build_measured_design_rule_extraction_summary,
        build_measured_flexibility_justification_table,
        build_measured_flexibility_overhead_rows,
        build_measured_support_rows,
        build_measured_trust_summary,
        build_measured_utility_rows,
        build_measured_utility_summary,
        build_shared_family_calibration_summary,
        render_measured_vs_modelled_trust_summary,
        render_measured_utility_summary,
        render_shared_family_calibration_summary,
        render_measured_design_rule_extraction_summary,
    )
    from mac_array_metrics import derive_static_row, summarize_workload
    from mac_array_policy import evaluate_policy
    from mac_array_regime import (
        build_adaptive_rejection_surface,
        build_rejection_summary,
        build_regime_insights,
        build_regime_map,
    )
    from mac_array_report import (
        current_measured_summary,
        render_markdown_report,
        render_regime_insights_markdown,
        render_plots,
        render_progress_log,
        write_csv,
        write_json,
    )
    from mac_array_types import ConstraintSpec, GridSpec
    from mac_array_workloads import workloads_from_config


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _grid_specs(items: list[dict[str, Any]]) -> list[GridSpec]:
    return [GridSpec(label=str(item["label"]), rows=int(item["rows"]), cols=int(item["cols"])) for item in items]


def _constraint_specs(items: list[dict[str, Any]]) -> list[ConstraintSpec]:
    return [
        ConstraintSpec(
            name=str(item["name"]),
            dsp_budget=int(item["dsp_budget"]),
            lut_budget=int(item["lut_budget"]),
            min_throughput_ops_per_cycle=(
                float(item["min_throughput_ops_per_cycle"])
                if item.get("min_throughput_ops_per_cycle") is not None
                else None
            ),
            target_grid=str(item["target_grid"]) if item.get("target_grid") else None,
        )
        for item in items
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the MAC-array framework v2 analysis flow")
    ap.add_argument(
        "--config",
        default="experiments/configs/mac_array_framework_v2.json",
        help="Framework config JSON",
    )
    ap.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "fpga" / "framework_v2"),
        help="Output directory",
    )
    args = ap.parse_args()

    config = _load_config(Path(args.config))
    output_dir = Path(args.output_dir).resolve()
    plots_dir = output_dir / "plots"

    grids = _grid_specs(config["grids"])
    evidence = load_evidence((REPO_ROOT / config["evidence_path"]).resolve())
    architectures = evidence.architectures
    workloads = workloads_from_config(config["workloads"])
    constraints = _constraint_specs(config["constraints"])
    switch_pairs = [tuple(pair) for pair in config["candidate_pairs"]]
    adaptive_modes = tuple(config["adaptive_modes"])

    static_rows = []
    workload_manifest_rows = []
    workload_rows = []
    phase_rows = []
    adaptive_rows = []
    adaptive_phase_rows = []
    adaptive_constraint_rows = []
    break_even_rows = []
    policy_rows = []
    policy_diagnostics = []
    workloads_by_name = {workload.name: workload for workload in workloads}
    grid_map = {grid.label: grid for grid in grids}

    for grid in grids:
        for arch in architectures.values():
            static_rows.append(derive_static_row(grid, arch, evidence))

        for workload in workloads:
            workload_manifest_rows.append(
                {
                    "workload": workload.name,
                    "workload_class": workload.workload_class,
                    "phase_count": workload.phase_count,
                    "total_ops": workload.total_ops,
                    "dominant_utilization": workload.dominant_utilization,
                    "utilization_variance": workload.utilization_variance,
                    "burstiness": workload.burstiness,
                    "notes": workload.notes,
                    "manifest_provenance_kind": "config_manifest",
                    "manifest_source_path": str(Path(args.config)),
                }
            )
            fixed_for_policy = []
            for arch in architectures.values():
                summary, phases = summarize_workload(grid, arch, workload, evidence)
                static_row = derive_static_row(grid, arch, evidence)
                row = asdict(summary)
                row.update(
                    {
                        "phase_count": workload.phase_count,
                        "workload_class": workload.workload_class,
                        "dominant_utilization": workload.dominant_utilization,
                        "utilization_variance": workload.utilization_variance,
                        "burstiness": workload.burstiness,
                        "workload_notes": workload.notes,
                        "latency_cycles_provenance_kind": "analytical_workload_model",
                        "latency_cycles_derivation": "sum(phase latency cycles across workload phases)",
                        "effective_throughput_ops_per_cycle_provenance_kind": "analytical_workload_model",
                        "effective_throughput_ops_per_cycle_derivation": "total_ops / total_latency_cycles",
                        "dsp_provenance_kind": static_row["dsp_provenance_kind"],
                        "dsp_source_id": static_row["dsp_source_id"],
                        "lut_provenance_kind": static_row["lut_provenance_kind"],
                        "lut_source_id": static_row["lut_source_id"],
                        "wns_estimate_ns_provenance_kind": static_row["wns_estimate_ns_provenance_kind"],
                        "wns_estimate_ns_source_id": static_row["wns_estimate_ns_source_id"],
                        "architecture_variant_id": static_row["architecture_variant_id"],
                        "architecture_variant_kind": static_row["architecture_variant_kind"],
                        "architecture_scope_note": static_row["architecture_scope_note"],
                        "direct_measurement_alignment": static_row["direct_measurement_alignment"],
                        "note": static_row["note"],
                        "note_provenance_kind": static_row["note_provenance_kind"],
                        "note_source_id": static_row["note_source_id"],
                    }
                )
                workload_rows.append(row)
                fixed_for_policy.append(row)
                for phase in phases:
                    phase_rows.append(
                        {
                            **asdict(phase),
                            "phase_latency_provenance_kind": "analytical_phase_model",
                            "phase_latency_derivation": "ops / served_parallelism + workload/architecture overheads",
                            "workload_phase_count": workload.phase_count,
                        }
                    )

            adaptive = evaluate_adaptive_workload(
                grid=grid,
                workload=workload,
                architectures=architectures,
                allowed_modes=adaptive_modes,
                evidence=evidence,
            )
            adaptive_rows.append(adaptive)
            for item in adaptive["phase_rows"]:
                adaptive_phase_rows.append(
                    {
                        "grid": grid.label,
                        "workload": workload.name,
                        **item,
                    }
                )
            break_even_rows.extend(
                build_break_even_rows(
                    grid=grid,
                    workload=workload,
                    architectures=architectures,
                    switch_pairs=switch_pairs,
                    evidence=evidence,
                )
            )

            for constraint in constraints:
                if constraint.target_grid and constraint.target_grid != grid.label:
                    continue
                feasible_modes = [
                    row["architecture"]
                    for row in fixed_for_policy
                    if row["dsp"] <= constraint.dsp_budget
                    and row["lut"] <= constraint.lut_budget
                    and row["timing_feasible"]
                ]
                adaptive_row = None
                if len(feasible_modes) >= 2:
                    adaptive_row = evaluate_adaptive_workload(
                        grid=grid,
                        workload=workload,
                        architectures=architectures,
                        allowed_modes=tuple(feasible_modes),
                        evidence=evidence,
                    )
                    adaptive_row.update(
                        {
                            "constraint": constraint.name,
                            "constraint_aware": True,
                            "allowed_modes_reason": "filtered to modes that satisfy DSP/LUT/timing constraints",
                        }
                    )
                    adaptive_constraint_rows.append(adaptive_row)
                recommendation, diagnostics = evaluate_policy(
                    fixed_rows=fixed_for_policy,
                    adaptive_row=adaptive_row,
                    constraint=constraint,
                    workload_class=workload.workload_class,
                )
                policy_rows.append(recommendation)
                policy_diagnostics.extend(diagnostics)

    measured_summary = current_measured_summary()
    direct_rows = build_direct_slice_comparison_rows()
    direct_tradeoff_rows = build_direct_tradeoff_rows(direct_rows)
    support_rows = build_measured_support_rows(direct_tradeoff_rows)
    trust_overlay_rows = build_framework_trust_overlay_rows(support_rows)
    trust_summary = build_measured_trust_summary(support_rows)
    calibration_rows = build_framework_calibration_aid_rows(direct_rows, direct_tradeoff_rows)
    calibration_overlay_rows = build_framework_calibration_overlay_rows(calibration_rows)
    calibration_summary = build_shared_family_calibration_summary(calibration_rows, calibration_overlay_rows)
    utility_rows = build_measured_utility_rows(direct_tradeoff_rows)
    bottleneck_rows = build_measured_bottleneck_choice_map(utility_rows, direct_tradeoff_rows)
    utility_summary = build_measured_utility_summary(utility_rows, bottleneck_rows)
    flexibility_overhead_rows = build_measured_flexibility_overhead_rows(direct_tradeoff_rows, utility_rows)
    flexibility_justification_rows = build_measured_flexibility_justification_table(flexibility_overhead_rows, bottleneck_rows)
    design_rule_extraction_summary = build_measured_design_rule_extraction_summary(
        flexibility_overhead_rows,
        flexibility_justification_rows,
    )
    provenance_rows = provenance_summary_rows(evidence)

    def regime_adaptive_factory(grid_label: str, workload_name: str, constraint: ConstraintSpec) -> dict[str, Any] | None:
        fixed_rows = [
            row for row in workload_rows if row["grid"] == grid_label and row["workload"] == workload_name
        ]
        feasible_modes = [
            row["architecture"]
            for row in fixed_rows
            if row["dsp"] <= constraint.dsp_budget
            and row["lut"] <= constraint.lut_budget
            and row["timing_feasible"]
        ]
        if len(feasible_modes) < 2:
            return None
        adaptive_row = evaluate_adaptive_workload(
            grid=grid_map[grid_label],
            workload=workloads_by_name[workload_name],
            architectures=architectures,
            allowed_modes=tuple(feasible_modes),
            evidence=evidence,
        )
        adaptive_row.update(
            {
                "constraint": constraint.name,
                "constraint_aware": True,
                "allowed_modes_reason": "filtered to modes that satisfy DSP/LUT/timing constraints",
            }
        )
        return adaptive_row

    regime_rows, regime_summary_rows, regime_meta = build_regime_map(workload_rows, regime_adaptive_factory)
    regime_rejection_rows = build_rejection_summary(regime_rows)
    rejection_surface_rows = build_adaptive_rejection_surface(regime_rows, regime_rejection_rows)
    regime_insight_rows = build_regime_insights(regime_rows, regime_summary_rows, rejection_surface_rows)

    write_json(output_dir / "config_snapshot.json", config)
    write_json(output_dir / "evidence_registry.json", evidence.registry_rows)
    write_csv(output_dir / "evidence_registry.csv", evidence.registry_rows)
    write_json(output_dir / "provenance_summary.json", provenance_rows)
    write_csv(output_dir / "provenance_summary.csv", provenance_rows)
    write_json(output_dir / "legacy_measured_summary.json", measured_summary)
    write_csv(output_dir / "measured_support_map.csv", support_rows)
    write_json(output_dir / "measured_support_map.json", support_rows)
    write_csv(output_dir / "framework_trust_overlay.csv", trust_overlay_rows)
    write_json(output_dir / "framework_trust_overlay.json", trust_overlay_rows)
    write_json(output_dir / "measured_trust_summary.json", trust_summary)
    write_csv(output_dir / "framework_calibration_aid.csv", calibration_rows)
    write_json(output_dir / "framework_calibration_aid.json", calibration_rows)
    write_csv(output_dir / "framework_calibration_overlay.csv", calibration_overlay_rows)
    write_json(output_dir / "framework_calibration_overlay.json", calibration_overlay_rows)
    write_json(output_dir / "shared_family_calibration_summary.json", calibration_summary)
    write_csv(output_dir / "measured_utility_table.csv", utility_rows)
    write_json(output_dir / "measured_utility_table.json", utility_rows)
    write_csv(output_dir / "measured_bottleneck_choice_map.csv", bottleneck_rows)
    write_json(output_dir / "measured_bottleneck_choice_map.json", bottleneck_rows)
    write_json(output_dir / "measured_utility_summary.json", utility_summary)
    write_csv(output_dir / "measured_flexibility_overhead_table.csv", flexibility_overhead_rows)
    write_json(output_dir / "measured_flexibility_overhead_table.json", flexibility_overhead_rows)
    write_csv(output_dir / "measured_flexibility_justification_table.csv", flexibility_justification_rows)
    write_json(output_dir / "measured_flexibility_justification_table.json", flexibility_justification_rows)
    write_json(output_dir / "measured_design_rule_extraction_summary.json", design_rule_extraction_summary)
    write_csv(output_dir / "workload_manifest.csv", workload_manifest_rows)
    write_json(output_dir / "workload_manifest.json", workload_manifest_rows)
    write_csv(output_dir / "static_architectures.csv", static_rows)
    write_json(output_dir / "static_architectures.json", static_rows)
    write_csv(output_dir / "workload_evaluations.csv", workload_rows)
    write_json(output_dir / "workload_evaluations.json", workload_rows)
    write_csv(output_dir / "phase_evaluations.csv", phase_rows)
    write_json(output_dir / "phase_evaluations.json", phase_rows)
    write_csv(output_dir / "adaptive_evaluations.csv", adaptive_rows)
    write_json(output_dir / "adaptive_evaluations.json", adaptive_rows)
    write_csv(output_dir / "adaptive_phase_decisions.csv", adaptive_phase_rows)
    write_json(output_dir / "adaptive_phase_decisions.json", adaptive_phase_rows)
    write_csv(output_dir / "adaptive_constraint_evaluations.csv", adaptive_constraint_rows)
    write_json(output_dir / "adaptive_constraint_evaluations.json", adaptive_constraint_rows)
    write_csv(output_dir / "break_even.csv", break_even_rows)
    write_json(output_dir / "break_even.json", break_even_rows)
    write_csv(output_dir / "policy_recommendations.csv", policy_rows)
    write_json(output_dir / "policy_recommendations.json", policy_rows)
    write_csv(output_dir / "policy_diagnostics.csv", policy_diagnostics)
    write_json(output_dir / "policy_diagnostics.json", policy_diagnostics)
    write_csv(output_dir / "regime_map.csv", regime_rows)
    write_json(output_dir / "regime_map.json", regime_rows)
    write_csv(output_dir / "regime_summary.csv", regime_summary_rows)
    write_json(output_dir / "regime_summary.json", {"meta": regime_meta, "rows": regime_summary_rows})
    write_csv(output_dir / "regime_rejection_summary.csv", regime_rejection_rows)
    write_json(output_dir / "regime_rejection_summary.json", regime_rejection_rows)
    write_csv(output_dir / "adaptive_rejection_surface.csv", rejection_surface_rows)
    write_json(output_dir / "adaptive_rejection_surface.json", rejection_surface_rows)
    write_csv(output_dir / "regime_insights.csv", regime_insight_rows)
    write_json(output_dir / "regime_insights.json", regime_insight_rows)

    render_markdown_report(
        output_path=output_dir / "framework_report.md",
        config=config,
        measured_summary=measured_summary,
        provenance_rows=provenance_rows,
        static_rows=static_rows,
        workload_rows=workload_rows,
        adaptive_rows=adaptive_rows,
        break_even_rows=break_even_rows,
        policy_rows=policy_rows,
        policy_diagnostics=policy_diagnostics,
        regime_rows=regime_rows,
        regime_summary_rows=regime_summary_rows,
        regime_meta=regime_meta,
        regime_rejection_rows=regime_rejection_rows,
        rejection_surface_rows=rejection_surface_rows,
        regime_insight_rows=regime_insight_rows,
    )
    render_shared_family_calibration_summary(
        output_path=output_dir / "shared_family_calibration_summary.md",
        summary=calibration_summary,
        overlay_rows=calibration_overlay_rows,
    )
    render_measured_utility_summary(
        output_path=output_dir / "measured_utility_summary.md",
        summary=utility_summary,
        bottleneck_rows=bottleneck_rows,
    )
    render_measured_design_rule_extraction_summary(
        output_path=output_dir / "measured_design_rule_extraction_summary.md",
        summary=design_rule_extraction_summary,
        justification_rows=flexibility_justification_rows,
    )
    render_regime_insights_markdown(
        output_path=output_dir / "regime_insights.md",
        regime_meta=regime_meta,
        regime_summary_rows=regime_summary_rows,
        rejection_surface_rows=rejection_surface_rows,
        regime_insight_rows=regime_insight_rows,
    )
    render_progress_log(output_dir / "progress_log.md")
    generated_plots = render_plots(
        plots_dir,
        workload_rows,
        break_even_rows,
        regime_rows,
        rejection_surface_rows,
    )
    write_json(output_dir / "generated_plots.json", [str(path) for path in generated_plots])

    print(f"Wrote MAC-array framework v2 outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
