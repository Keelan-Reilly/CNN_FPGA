#!/usr/bin/env python3
"""Generate direct measured-vs-modelled MAC-array slice artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .mac_array_direct_slice import (
        build_direct_calibration_summary,
        build_direct_tradeoff_rows,
        build_direct_slice_comparison_rows,
        build_direct_shared_implementation_comparison_rows,
        build_direct_shared_implementation_summary,
        build_direct_shared_scaling_summary,
        build_framework_calibration_aid_rows,
        build_framework_calibration_overlay_rows,
        build_framework_trust_overlay_rows,
        build_measured_bottleneck_choice_map,
        build_measured_design_rule_extraction_summary,
        build_measured_decision_surface,
        build_measured_budget_boundary_rows,
        build_measured_regime_transfer_summary,
        build_measured_supported_region_map,
        build_measured_design_rule_summary,
        build_measured_extrapolation_boundary_summary,
        build_measured_fit_residual_rows,
        build_measured_flexibility_justification_table,
        build_measured_flexibility_overhead_rows,
        build_measured_predictor_rows,
        build_measured_predictor_summary,
        build_measured_support_rows,
        build_measured_trust_summary,
        build_measured_utility_rows,
        build_measured_utility_summary,
        build_shared_family_calibration_summary,
        build_measured_tradeoff_decision_rows,
        render_direct_calibration_plot,
        render_direct_calibration_summary,
        render_direct_slice_summary,
        render_direct_tradeoff_summary,
        render_direct_shared_implementation_summary,
        render_direct_shared_scaling_summary,
        render_measured_design_rules,
        render_measured_design_rule_extraction_summary,
        render_measured_regime_transfer_summary,
        render_measured_supported_region_map,
        render_measured_utility_summary,
        render_measured_extrapolation_boundary,
        render_measured_predictor_summary,
        render_measured_vs_modelled_trust_summary,
        render_shared_family_calibration_summary,
        render_measured_tradeoff_regime_summary,
    )
    from .mac_array_report import write_csv, write_json
except ImportError:
    from mac_array_direct_slice import (
        build_direct_calibration_summary,
        build_direct_tradeoff_rows,
        build_direct_slice_comparison_rows,
        build_direct_shared_implementation_comparison_rows,
        build_direct_shared_implementation_summary,
        build_direct_shared_scaling_summary,
        build_framework_calibration_aid_rows,
        build_framework_calibration_overlay_rows,
        build_framework_trust_overlay_rows,
        build_measured_bottleneck_choice_map,
        build_measured_design_rule_extraction_summary,
        build_measured_decision_surface,
        build_measured_budget_boundary_rows,
        build_measured_regime_transfer_summary,
        build_measured_supported_region_map,
        build_measured_design_rule_summary,
        build_measured_extrapolation_boundary_summary,
        build_measured_fit_residual_rows,
        build_measured_flexibility_justification_table,
        build_measured_flexibility_overhead_rows,
        build_measured_predictor_rows,
        build_measured_predictor_summary,
        build_measured_support_rows,
        build_measured_trust_summary,
        build_measured_utility_rows,
        build_measured_utility_summary,
        build_shared_family_calibration_summary,
        build_measured_tradeoff_decision_rows,
        render_direct_calibration_plot,
        render_direct_calibration_summary,
        render_direct_slice_summary,
        render_direct_tradeoff_summary,
        render_direct_shared_implementation_summary,
        render_direct_shared_scaling_summary,
        render_measured_design_rules,
        render_measured_design_rule_extraction_summary,
        render_measured_regime_transfer_summary,
        render_measured_supported_region_map,
        render_measured_utility_summary,
        render_measured_extrapolation_boundary,
        render_measured_predictor_summary,
        render_measured_vs_modelled_trust_summary,
        render_shared_family_calibration_summary,
        render_measured_tradeoff_regime_summary,
    )
    from mac_array_report import write_csv, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate direct MAC-array slice comparison artifacts")
    ap.add_argument(
        "--output-dir",
        default="results/fpga/framework_v2/direct_slice",
        help="Output directory for direct-slice artifacts",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    rows = build_direct_slice_comparison_rows()
    summary = build_direct_calibration_summary(rows)
    predictor_rows = build_measured_predictor_rows(rows)
    predictor_residual_rows = build_measured_fit_residual_rows(rows, predictor_rows)
    predictor_summary = build_measured_predictor_summary(predictor_rows)
    extrapolation_boundary = build_measured_extrapolation_boundary_summary(predictor_rows)
    tradeoff_rows = build_direct_tradeoff_rows(rows)
    comparison_rows = build_direct_shared_implementation_comparison_rows(tradeoff_rows)
    comparison_summary = build_direct_shared_implementation_summary(tradeoff_rows)
    scaling_summary = build_direct_shared_scaling_summary(tradeoff_rows)
    support_rows = build_measured_support_rows(tradeoff_rows)
    trust_overlay_rows = build_framework_trust_overlay_rows(support_rows)
    trust_summary = build_measured_trust_summary(support_rows)
    calibration_rows = build_framework_calibration_aid_rows(rows, tradeoff_rows)
    calibration_overlay_rows = build_framework_calibration_overlay_rows(calibration_rows)
    calibration_summary = build_shared_family_calibration_summary(calibration_rows, calibration_overlay_rows)
    utility_rows = build_measured_utility_rows(tradeoff_rows)
    bottleneck_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
    utility_summary = build_measured_utility_summary(utility_rows, bottleneck_rows)
    flexibility_overhead_rows = build_measured_flexibility_overhead_rows(tradeoff_rows, utility_rows)
    flexibility_justification_rows = build_measured_flexibility_justification_table(flexibility_overhead_rows, bottleneck_rows)
    design_rule_extraction_summary = build_measured_design_rule_extraction_summary(
        flexibility_overhead_rows,
        flexibility_justification_rows,
    )
    decision_surface_rows = build_measured_decision_surface(predictor_rows, utility_rows, bottleneck_rows)
    budget_boundary_rows = build_measured_budget_boundary_rows(predictor_rows)
    regime_transfer_summary = build_measured_regime_transfer_summary(
        decision_surface_rows,
        budget_boundary_rows,
        predictor_rows,
    )
    supported_region_map = build_measured_supported_region_map(decision_surface_rows, budget_boundary_rows)
    decision_rows = build_measured_tradeoff_decision_rows(tradeoff_rows)
    design_rule_summary = build_measured_design_rule_summary(tradeoff_rows, decision_rows)
    write_csv(output_dir / "direct_measured_vs_modelled.csv", rows)
    write_json(output_dir / "direct_measured_vs_modelled.json", rows)
    write_json(output_dir / "direct_calibration_summary.json", summary)
    write_csv(output_dir / "measured_predictor_table.csv", predictor_rows)
    write_json(output_dir / "measured_predictor_table.json", predictor_rows)
    write_csv(output_dir / "measured_fit_residuals.csv", predictor_residual_rows)
    write_json(output_dir / "measured_fit_residuals.json", predictor_residual_rows)
    write_json(output_dir / "measured_predictor_summary.json", predictor_summary)
    write_json(output_dir / "measured_extrapolation_boundary.json", extrapolation_boundary)
    write_csv(output_dir / "direct_tradeoff_measured_vs_modelled.csv", tradeoff_rows)
    write_json(output_dir / "direct_tradeoff_measured_vs_modelled.json", tradeoff_rows)
    write_csv(output_dir / "direct_shared_implementation_comparison.csv", comparison_rows)
    write_json(output_dir / "direct_shared_implementation_comparison.json", comparison_rows)
    write_json(output_dir / "direct_shared_implementation_summary.json", comparison_summary)
    write_json(output_dir / "direct_shared_scaling_summary.json", scaling_summary)
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
    write_csv(output_dir / "measured_decision_surface.csv", decision_surface_rows)
    write_json(output_dir / "measured_decision_surface.json", decision_surface_rows)
    write_csv(output_dir / "measured_budget_boundary_table.csv", budget_boundary_rows)
    write_json(output_dir / "measured_budget_boundary_table.json", budget_boundary_rows)
    write_json(output_dir / "measured_regime_transfer_summary.json", regime_transfer_summary)
    write_json(output_dir / "measured_supported_region_map.json", supported_region_map)
    write_csv(output_dir / "measured_tradeoff_decision_table.csv", decision_rows)
    write_json(output_dir / "measured_tradeoff_decision_table.json", decision_rows)
    write_json(output_dir / "measured_design_rule_summary.json", design_rule_summary)
    render_direct_slice_summary(output_dir / "direct_slice_summary.md", rows, summary, design_rule_summary)
    render_direct_calibration_summary(output_dir / "direct_calibration_summary.md", summary)
    render_measured_predictor_summary(output_dir / "measured_predictor_summary.md", predictor_summary, predictor_rows)
    render_measured_extrapolation_boundary(output_dir / "measured_extrapolation_boundary.md", extrapolation_boundary)
    render_direct_tradeoff_summary(output_dir / "direct_tradeoff_summary.md", tradeoff_rows)
    render_direct_shared_implementation_summary(output_dir / "direct_shared_implementation_summary.md", comparison_summary)
    render_direct_shared_scaling_summary(output_dir / "direct_shared_scaling_summary.md", scaling_summary)
    render_measured_vs_modelled_trust_summary(output_dir / "measured_vs_modelled_trust_summary.md", trust_summary, support_rows)
    render_shared_family_calibration_summary(output_dir / "shared_family_calibration_summary.md", calibration_summary, calibration_overlay_rows)
    render_measured_utility_summary(output_dir / "measured_utility_summary.md", utility_summary, bottleneck_rows)
    render_measured_design_rule_extraction_summary(
        output_dir / "measured_design_rule_extraction_summary.md",
        design_rule_extraction_summary,
        flexibility_justification_rows,
    )
    render_measured_regime_transfer_summary(
        output_dir / "measured_regime_transfer_summary.md",
        regime_transfer_summary,
        decision_surface_rows,
    )
    render_measured_supported_region_map(
        output_dir / "measured_supported_region_map.md",
        supported_region_map,
    )
    render_measured_tradeoff_regime_summary(output_dir / "measured_tradeoff_regime_summary.md", decision_rows)
    render_measured_design_rules(output_dir / "measured_design_rules.md", design_rule_summary)
    plot_path = render_direct_calibration_plot(output_dir / "direct_calibration_plot.png", rows)
    write_json(output_dir / "direct_generated_plot.json", plot_path)
    print(f"Wrote direct-slice outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
