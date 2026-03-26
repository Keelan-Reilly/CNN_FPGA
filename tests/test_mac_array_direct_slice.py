#!/usr/bin/env python3
"""Deterministic tests for the direct MAC-array slice plumbing."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from analysis.mac_array_direct_slice import (
    build_direct_calibration_summary,
    build_direct_shared_implementation_comparison_rows,
    build_direct_shared_implementation_summary,
    build_direct_shared_scaling_summary,
    build_direct_slice_comparison_rows,
    build_final_architecture_choice_boundary_table,
    build_final_artifact_index,
    build_final_design_rule_table,
    build_final_reproducibility_guide,
    build_final_results_summary,
    build_final_trust_calibration_table,
    build_framework_calibration_aid_rows,
    build_framework_calibration_overlay_rows,
    build_framework_trust_overlay_rows,
    build_measured_bottleneck_choice_map,
    build_measured_design_rule_extraction_summary,
    build_measured_support_rows,
    build_measured_trust_summary,
    build_direct_tradeoff_rows,
    build_measured_design_rule_summary,
    build_measured_budget_boundary_rows,
    build_measured_decision_surface,
    build_measured_extrapolation_boundary_summary,
    build_measured_fit_residual_rows,
    build_measured_flexibility_justification_table,
    build_measured_flexibility_overhead_rows,
    build_measured_predictor_rows,
    build_measured_predictor_summary,
    build_measured_regime_transfer_summary,
    build_measured_supported_region_map,
    build_measured_tradeoff_decision_rows,
    build_measured_utility_rows,
    build_measured_utility_summary,
    build_shared_family_calibration_summary,
    direct_architecture_name,
    direct_latency_model,
    direct_throughput_model,
)
from run_fpga_experiments import SUPPORTED_SWEEP_PARAMS


class DirectMacSliceTest(unittest.TestCase):
    def _multi_shared_input_rows(self) -> list[dict[str, object]]:
        return [
            {
                "grid": "4x4",
                "k_depth": 32,
                "architecture": "baseline",
                "architecture_family": "baseline",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "baseline",
                "direct_evidence_kind": "direct_measured_mac_array_baseline_slice",
                "measured_dsp": 16,
                "model_dsp": 16,
                "measured_lut": 1061,
                "model_lut": 660,
                "measured_ff": 524,
                "measured_wns_ns": 1.942,
                "model_wns_estimate_ns": 0.808,
                "measured_latency_cycles": 33,
                "direct_slice_latency_model_cycles": 33,
                "measured_effective_throughput_ops_per_cycle": 15.515151515151516,
                "direct_slice_throughput_model_ops_per_cycle": 15.515152,
            },
            {
                "grid": "4x4",
                "k_depth": 32,
                "architecture": "shared_lut_saving",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_lut",
                "direct_evidence_kind": "direct_measured_mac_array_shared_lut_saving_slice",
                "measured_dsp": 16,
                "model_dsp": 8,
                "measured_lut": 679,
                "model_lut": 508,
                "measured_ff": 525,
                "measured_wns_ns": 1.199,
                "model_wns_estimate_ns": 0.708,
                "measured_latency_cycles": 65,
                "direct_slice_latency_model_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 7.876923076923077,
                "direct_slice_throughput_model_ops_per_cycle": 7.876923,
            },
            {
                "grid": "4x4",
                "k_depth": 32,
                "architecture": "shared_dsp_reducing",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_dsp",
                "direct_evidence_kind": "direct_measured_mac_array_shared_dsp_reducing_slice",
                "measured_dsp": 8,
                "model_dsp": 8,
                "measured_lut": 820,
                "model_lut": 508,
                "measured_ff": 526,
                "measured_wns_ns": 0.950,
                "model_wns_estimate_ns": 0.708,
                "measured_latency_cycles": 65,
                "direct_slice_latency_model_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 7.876923076923077,
                "direct_slice_throughput_model_ops_per_cycle": 7.876923,
            },
        ]

    def _two_scale_input_rows(self) -> list[dict[str, object]]:
        return self._multi_shared_input_rows() + [
            {
                "grid": "8x4",
                "k_depth": 32,
                "architecture": "baseline",
                "architecture_family": "baseline",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "baseline_8x4",
                "direct_evidence_kind": "direct_measured_mac_array_baseline_slice",
                "measured_dsp": 32,
                "model_dsp": 32,
                "measured_lut": 2134,
                "model_lut": 1200,
                "measured_ff": 1036,
                "measured_wns_ns": 2.019,
                "model_wns_estimate_ns": 0.808,
                "measured_latency_cycles": 33,
                "direct_slice_latency_model_cycles": 33,
                "measured_effective_throughput_ops_per_cycle": 31.03030303030303,
                "direct_slice_throughput_model_ops_per_cycle": 31.030303,
            },
            {
                "grid": "8x4",
                "k_depth": 32,
                "architecture": "shared_lut_saving",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_lut_8x4",
                "direct_evidence_kind": "direct_measured_mac_array_shared_lut_saving_slice",
                "measured_dsp": 32,
                "model_dsp": 16,
                "measured_lut": 1400,
                "model_lut": 980,
                "measured_ff": 1040,
                "measured_wns_ns": 1.700,
                "model_wns_estimate_ns": 0.708,
                "measured_latency_cycles": 65,
                "direct_slice_latency_model_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 15.753846153846155,
                "direct_slice_throughput_model_ops_per_cycle": 15.753846,
            },
            {
                "grid": "8x4",
                "k_depth": 32,
                "architecture": "shared_dsp_reducing",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_dsp_8x4",
                "direct_evidence_kind": "direct_measured_mac_array_shared_dsp_reducing_slice",
                "measured_dsp": 0,
                "model_dsp": 16,
                "measured_lut": 1820,
                "model_lut": 980,
                "measured_ff": 1150,
                "measured_wns_ns": 2.300,
                "model_wns_estimate_ns": 0.708,
                "measured_latency_cycles": 65,
                "direct_slice_latency_model_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 15.753846153846155,
                "direct_slice_throughput_model_ops_per_cycle": 15.753846,
            },
        ]

    def _three_scale_actual_input_rows(self) -> list[dict[str, object]]:
        return [
            {
                "grid": "4x4",
                "grid_rows": 4,
                "grid_cols": 4,
                "mac_units": 16,
                "k_depth": 32,
                "architecture": "baseline",
                "architecture_family": "baseline",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "baseline_4x4",
                "direct_evidence_kind": "direct_measured_mac_array_baseline_slice",
                "measured_dsp": 16,
                "measured_lut": 1061,
                "measured_ff": 524,
                "measured_wns_ns": 1.942,
                "measured_latency_cycles": 33,
                "measured_effective_throughput_ops_per_cycle": 15.515151515151516,
            },
            {
                "grid": "8x4",
                "grid_rows": 8,
                "grid_cols": 4,
                "mac_units": 32,
                "k_depth": 32,
                "architecture": "baseline",
                "architecture_family": "baseline",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "baseline_8x4",
                "direct_evidence_kind": "direct_measured_mac_array_baseline_slice",
                "measured_dsp": 32,
                "measured_lut": 2134,
                "measured_ff": 1036,
                "measured_wns_ns": 2.019,
                "measured_latency_cycles": 33,
                "measured_effective_throughput_ops_per_cycle": 31.03030303030303,
            },
            {
                "grid": "8x8",
                "grid_rows": 8,
                "grid_cols": 8,
                "mac_units": 64,
                "k_depth": 32,
                "architecture": "baseline",
                "architecture_family": "baseline",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "baseline_8x8",
                "direct_evidence_kind": "direct_measured_mac_array_baseline_slice",
                "measured_dsp": 64,
                "measured_lut": 4287,
                "measured_ff": 2060,
                "measured_wns_ns": 0.634,
                "measured_latency_cycles": 33,
                "measured_effective_throughput_ops_per_cycle": 62.06060606060606,
            },
            {
                "grid": "4x4",
                "grid_rows": 4,
                "grid_cols": 4,
                "mac_units": 16,
                "k_depth": 32,
                "architecture": "shared_lut_saving",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_lut_4x4",
                "direct_evidence_kind": "direct_measured_mac_array_shared_lut_saving_slice",
                "measured_dsp": 16,
                "measured_lut": 679,
                "measured_ff": 525,
                "measured_wns_ns": 1.199,
                "measured_latency_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 7.876923076923077,
            },
            {
                "grid": "8x4",
                "grid_rows": 8,
                "grid_cols": 4,
                "mac_units": 32,
                "k_depth": 32,
                "architecture": "shared_lut_saving",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_lut_8x4",
                "direct_evidence_kind": "direct_measured_mac_array_shared_lut_saving_slice",
                "measured_dsp": 32,
                "measured_lut": 1351,
                "measured_ff": 1037,
                "measured_wns_ns": 1.038,
                "measured_latency_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 15.753846153846155,
            },
            {
                "grid": "8x8",
                "grid_rows": 8,
                "grid_cols": 8,
                "mac_units": 64,
                "k_depth": 32,
                "architecture": "shared_lut_saving",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_lut_8x8",
                "direct_evidence_kind": "direct_measured_mac_array_shared_lut_saving_slice",
                "measured_dsp": 64,
                "measured_lut": 2694,
                "measured_ff": 2068,
                "measured_wns_ns": 0.948,
                "measured_latency_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 31.50769230769231,
            },
            {
                "grid": "4x4",
                "grid_rows": 4,
                "grid_cols": 4,
                "mac_units": 16,
                "k_depth": 32,
                "architecture": "shared_dsp_reducing",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_dsp_4x4",
                "direct_evidence_kind": "direct_measured_mac_array_shared_dsp_reducing_slice",
                "measured_dsp": 0,
                "measured_lut": 910,
                "measured_ff": 589,
                "measured_wns_ns": 2.261,
                "measured_latency_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 7.876923076923077,
            },
            {
                "grid": "8x4",
                "grid_rows": 8,
                "grid_cols": 4,
                "mac_units": 32,
                "k_depth": 32,
                "architecture": "shared_dsp_reducing",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_dsp_8x4",
                "direct_evidence_kind": "direct_measured_mac_array_shared_dsp_reducing_slice",
                "measured_dsp": 0,
                "measured_lut": 1817,
                "measured_ff": 1165,
                "measured_wns_ns": 1.378,
                "measured_latency_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 15.753846153846155,
            },
            {
                "grid": "8x8",
                "grid_rows": 8,
                "grid_cols": 8,
                "mac_units": 64,
                "k_depth": 32,
                "architecture": "shared_dsp_reducing",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "run_id": "shared_dsp_8x8",
                "direct_evidence_kind": "direct_measured_mac_array_shared_dsp_reducing_slice",
                "measured_dsp": 0,
                "measured_lut": 3620,
                "measured_ff": 2324,
                "measured_wns_ns": 0.773,
                "measured_latency_cycles": 65,
                "measured_effective_throughput_ops_per_cycle": 31.50769230769231,
            },
        ]

    def test_direct_slice_models_are_simple_and_stable(self) -> None:
        self.assertEqual(direct_latency_model(32), 33)
        self.assertEqual(direct_latency_model(32, "shared_lut_saving"), 65)
        self.assertEqual(direct_latency_model(32, "shared_dsp_reducing"), 65)
        self.assertAlmostEqual(direct_throughput_model(4, 4, 32), 512.0 / 33.0)
        self.assertAlmostEqual(direct_throughput_model(4, 4, 32, "shared_lut_saving"), 512.0 / 65.0)
        self.assertAlmostEqual(direct_throughput_model(4, 4, 32, "shared_dsp_reducing"), 512.0 / 65.0)

    def test_supported_params_include_direct_slice_knobs(self) -> None:
        self.assertIn("ARRAY_ROWS", SUPPORTED_SWEEP_PARAMS)
        self.assertIn("ARRAY_COLS", SUPPORTED_SWEEP_PARAMS)
        self.assertIn("K_DEPTH", SUPPORTED_SWEEP_PARAMS)
        self.assertIn("ARCH_MODE", SUPPORTED_SWEEP_PARAMS)

    def test_architecture_name_distinguishes_shared_variants(self) -> None:
        self.assertEqual(direct_architecture_name({}), "baseline")
        self.assertEqual(direct_architecture_name({"ARCH_MODE": 1}), "shared_lut_saving")
        self.assertEqual(direct_architecture_name({"ARCH_MODE": 2}), "shared_dsp_reducing")

    def test_calibration_summary_reports_exact_matches_and_lut_fit(self) -> None:
        rows = [
            {
                "grid": "4x4",
                "mac_units": 16,
                "architecture": "baseline",
                "architecture_family": "baseline",
                "comparison_status": "direct_measured_vs_modelled",
                "direct_evidence_source": "results/fpga/aggregates/study_mac_array_direct_baseline_4x4.json",
                "dsp_delta": 0,
                "latency_delta_cycles": 0,
                "throughput_delta_ops_per_cycle": 0.0,
                "lut_delta": 401,
                "measured_lut": 1061,
                "measured_wns_ns": 1.942,
            },
            {
                "grid": "8x4",
                "mac_units": 32,
                "architecture": "baseline",
                "architecture_family": "baseline",
                "comparison_status": "direct_measured_vs_modelled",
                "direct_evidence_source": "results/fpga/aggregates/study_mac_array_direct_baseline_8x4.json",
                "dsp_delta": 0,
                "latency_delta_cycles": 0,
                "throughput_delta_ops_per_cycle": 0.0,
                "lut_delta": 994,
                "measured_lut": 2134,
                "measured_wns_ns": 2.019,
            },
            {
                "grid": "8x8",
                "mac_units": 64,
                "architecture": "baseline",
                "architecture_family": "baseline",
                "comparison_status": "direct_measured_vs_modelled",
                "direct_evidence_source": "results/fpga/aggregates/study_mac_array_direct_baseline_8x8.json",
                "dsp_delta": 0,
                "latency_delta_cycles": 0,
                "throughput_delta_ops_per_cycle": 0.0,
                "lut_delta": 2187,
                "measured_lut": 4287,
                "measured_wns_ns": 0.634,
            },
            {
                "grid": "4x4",
                "mac_units": 16,
                "architecture": "shared_lut_saving",
                "architecture_family": "shared",
                "comparison_status": "direct_measured_vs_modelled",
                "direct_evidence_source": "results/fpga/aggregates/study_mac_array_direct_tradeoff_4x4.json",
                "dsp_delta": 0,
                "latency_delta_cycles": 0,
                "throughput_delta_ops_per_cycle": 0.0,
                "lut_delta": 12,
                "measured_lut": 520,
                "measured_wns_ns": 3.0,
            },
        ]
        summary = build_direct_calibration_summary(rows)
        self.assertEqual(summary["measured_points"], 3)
        self.assertEqual(summary["dsp_exact_match_count"], 3)
        self.assertEqual(summary["latency_exact_match_count"], 3)
        self.assertEqual(summary["throughput_exact_match_count"], 3)
        self.assertEqual(summary["lut_error_min"], 401)
        self.assertEqual(summary["lut_error_max"], 2187)
        self.assertAlmostEqual(summary["lut_error_mean"], 1194.0)
        self.assertIn("calibrated_lut_model", summary)
        self.assertEqual(summary["calibrated_lut_model"]["calibration_kind"], "direct_slice_linear_fit")
        self.assertEqual(summary["calibrated_lut_model"]["provenance_kind"], "derived_from_direct_measured_mac_slice")

    def test_comparison_rows_keep_direct_measured_distinction(self) -> None:
        payload = {
            "runs": [
                {
                    "run_id": "mac_array_baseline_4x4_k32",
                    "status": "succeeded",
                    "dsp": 16,
                    "ff": 524,
                    "lut": 700,
                    "wns_ns": 0.25,
                    "latency_cycles": 33,
                    "effective_throughput_ops_per_cycle": 15.5,
                    "params": '{"ARRAY_ROWS": 4, "ARRAY_COLS": 4, "K_DEPTH": 32}',
                },
                {
                    "run_id": "mac_array_shared_lut_4x4_k32",
                    "status": "succeeded",
                    "dsp": 8,
                    "ff": 300,
                    "lut": 520,
                    "wns_ns": 0.5,
                    "latency_cycles": 65,
                    "effective_throughput_ops_per_cycle": 7.876923,
                    "params": '{"ARRAY_ROWS": 4, "ARRAY_COLS": 4, "K_DEPTH": 32, "ARCH_MODE": 1}',
                },
                {
                    "run_id": "mac_array_shared_dsp_4x4_k32",
                    "status": "succeeded",
                    "dsp": 4,
                    "ff": 310,
                    "lut": 580,
                    "wns_ns": 0.45,
                    "latency_cycles": 65,
                    "effective_throughput_ops_per_cycle": 7.876923,
                    "params": '{"ARRAY_ROWS": 4, "ARRAY_COLS": 4, "K_DEPTH": 32, "ARCH_MODE": 2}',
                }
            ]
        }
        rows = build_direct_slice_comparison_rows(payload)
        self.assertEqual(len(rows), 3)
        baseline = next(row for row in rows if row["architecture"] == "baseline")
        shared_lut = next(row for row in rows if row["architecture"] == "shared_lut_saving")
        shared_dsp = next(row for row in rows if row["architecture"] == "shared_dsp_reducing")
        self.assertEqual(baseline["direct_evidence_kind"], "direct_measured_mac_array_baseline_slice")
        self.assertEqual(shared_lut["direct_evidence_kind"], "direct_measured_mac_array_shared_lut_saving_slice")
        self.assertEqual(shared_dsp["direct_evidence_kind"], "direct_measured_mac_array_shared_dsp_reducing_slice")
        self.assertEqual(shared_lut["direct_slice_latency_model_cycles"], 65)
        self.assertEqual(shared_dsp["architecture_family"], "shared")

    def test_tradeoff_rows_capture_multiple_shared_variants(self) -> None:
        tradeoff = build_direct_tradeoff_rows(self._multi_shared_input_rows())
        self.assertEqual(len(tradeoff), 2)
        by_variant = {row["shared_architecture_variant"]: row for row in tradeoff}

        lut_row = by_variant["shared_lut_saving"]
        self.assertEqual(lut_row["measured_dsp_delta_shared_minus_baseline"], 0)
        self.assertEqual(lut_row["shared_measured_relief_kind"], "lut_relief")
        self.assertAlmostEqual(lut_row["measured_lut_reduction_pct"], 36.00377, places=5)
        self.assertAlmostEqual(lut_row["measured_latency_increase_factor"], 1.969697, places=6)
        self.assertAlmostEqual(lut_row["measured_throughput_retention_pct"], 50.769231, places=5)
        self.assertIn("LUT-saving shared implementation", lut_row["tradeoff_note"])

        dsp_row = by_variant["shared_dsp_reducing"]
        self.assertEqual(dsp_row["measured_dsp_delta_shared_minus_baseline"], -8)
        self.assertEqual(dsp_row["shared_measured_relief_kind"], "dsp_and_lut_relief")
        self.assertIn("DSP-reducing shared implementation", dsp_row["tradeoff_note"])

    def test_shared_implementation_comparison_rows_capture_three_way_view(self) -> None:
        tradeoff = build_direct_tradeoff_rows(self._multi_shared_input_rows())
        comparison_rows = build_direct_shared_implementation_comparison_rows(tradeoff)
        self.assertEqual(len(comparison_rows), 3)
        pairwise = next(row for row in comparison_rows if row["comparison_kind"] == "shared_variant_vs_shared_variant")
        self.assertEqual(pairwise["measured_dsp_delta_lhs_minus_rhs"], -8)
        self.assertEqual(pairwise["measured_lut_delta_lhs_minus_rhs"], 141)

    def test_design_rule_summary_distinguishes_two_shared_implementations(self) -> None:
        tradeoff = build_direct_tradeoff_rows(self._two_scale_input_rows())
        decision_rows = build_measured_tradeoff_decision_rows(tradeoff)
        summary = build_measured_design_rule_summary(tradeoff, decision_rows)
        comparison_summary = build_direct_shared_implementation_summary(tradeoff)
        joined = "\n".join(summary["summary_lines"])
        comparison_joined = "\n".join(comparison_summary["summary_lines"])
        comparison_grid_joined = "\n".join(
            line
            for grid_summary in comparison_summary["grid_summaries"]
            for line in grid_summary["summary_lines"]
        )
        self.assertIn("LUT-saving shared implementation is not a DSP-saving strategy", joined)
        self.assertIn("lowers DSP by `8`", joined)
        self.assertIn("Sharing is not one thing", joined)
        self.assertIn("survives at 8x4", comparison_joined)
        self.assertIn("shared_lut_saving saves LUT", comparison_grid_joined)
        self.assertIn("shared_dsp_reducing reduces DSP", comparison_grid_joined)

    def test_measured_decision_rows_stay_bound_to_lut_saving_variant(self) -> None:
        tradeoff = build_direct_tradeoff_rows(self._multi_shared_input_rows())
        decision_rows = build_measured_tradeoff_decision_rows(tradeoff)
        self.assertEqual(len(decision_rows), 7)
        lut_shared_only = next(row for row in decision_rows if row["regime_id"] == "lut_shared_only_relaxed_perf")
        self.assertEqual(lut_shared_only["shared_architecture_variant"], "shared_lut_saving")
        self.assertEqual(lut_shared_only["preferred_implementation"], "shared_preferred")
        slack_or_dsp = next(row for row in decision_rows if row["regime_id"] == "slack_sensitive_or_dsp_pressure")
        self.assertEqual(slack_or_dsp["secondary_preferred_implementation"], "no_feasible_measured_option")


    def test_scaling_summary_detects_survival_across_two_grids(self) -> None:
        tradeoff = build_direct_tradeoff_rows(self._two_scale_input_rows())
        scaling = build_direct_shared_scaling_summary(tradeoff)
        comparison = build_direct_shared_implementation_summary(tradeoff)
        self.assertEqual(scaling["scale_points"], 2)
        self.assertEqual(scaling["scaling_rule_status"], "survives")
        self.assertIn("survives at 8x4", scaling["scaling_rule_line"])
        self.assertEqual(len(comparison["grid_summaries"]), 2)
        grid_map = {row["grid"]: row for row in scaling["grid_summaries"]}
        self.assertTrue(grid_map["4x4"]["baseline_performance_first"])
        self.assertTrue(grid_map["8x4"]["shared_lut_saving_is_best_lut_relief"])
        self.assertTrue(grid_map["8x4"]["shared_dsp_reducing_is_best_dsp_relief"])


    def test_support_rows_classify_measured_roles_and_family_claims(self) -> None:
        tradeoff = build_direct_tradeoff_rows(self._two_scale_input_rows())
        support_rows = build_measured_support_rows(tradeoff)
        by_claim = {row["claim_id"]: row for row in support_rows}
        self.assertEqual(by_claim["baseline_performance_first_role"]["support_level"], "directly_measured_supported")
        self.assertEqual(by_claim["shared_lut_saving_lut_relief_role"]["support_level"], "directly_measured_supported")
        self.assertEqual(by_claim["shared_dsp_reducing_dsp_relief_role"]["support_level"], "directly_measured_supported")
        self.assertEqual(by_claim["shared_family_dsp_relief"]["support_level"], "measured_partial_support")
        self.assertEqual(by_claim["shared_family_single_measured_truth"]["support_level"], "contradicted_by_measured_implementation")
        self.assertEqual(by_claim["shared_8x8_dsp_reduction_anchor"]["support_level"], "extrapolated_beyond_measured_support")

    def test_trust_summary_and_overlay_expose_boundary_language(self) -> None:
        tradeoff = build_direct_tradeoff_rows(self._two_scale_input_rows())
        support_rows = build_measured_support_rows(tradeoff)
        overlay_rows = build_framework_trust_overlay_rows(support_rows)
        summary = build_measured_trust_summary(support_rows)
        joined = "\n".join(summary["summary_lines"])
        overlay_joined = "\n".join(row["trust_boundary"] for row in overlay_rows)
        self.assertIn("Directly measured support", joined)
        self.assertIn("implementation-specific", joined)
        self.assertIn("extrapolated beyond measured support", joined)
        self.assertIn("modelled-family", joined)
        self.assertIn("one measured truth", overlay_joined)
        self.assertIn("not supported", overlay_joined)

    def test_framework_calibration_aid_marks_baseline_optimism_and_shared_direction(self) -> None:
        direct_rows = self._two_scale_input_rows()
        tradeoff_rows = build_direct_tradeoff_rows(direct_rows)
        calibration_rows = build_framework_calibration_aid_rows(direct_rows, tradeoff_rows)

        baseline_lut_rows = [
            row
            for row in calibration_rows
            if row["architecture_variant_or_family"] == "baseline" and row["metric"] == "lut"
        ]
        self.assertEqual(len(baseline_lut_rows), 2)
        self.assertTrue(
            all(row["calibration_status"] == "directionally_aligned_but_numerically_optimistic" for row in baseline_lut_rows)
        )

        direction_rows = [
            row
            for row in calibration_rows
            if row["architecture_variant_or_family"] == "shared_modelled_family_direction"
        ]
        self.assertTrue(direction_rows)
        self.assertTrue(all(row["calibration_status"] == "well_aligned" for row in direction_rows))

    def test_framework_calibration_summary_keeps_shared_resources_bounded(self) -> None:
        direct_rows = self._two_scale_input_rows()
        tradeoff_rows = build_direct_tradeoff_rows(direct_rows)
        calibration_rows = build_framework_calibration_aid_rows(direct_rows, tradeoff_rows)
        overlay_rows = build_framework_calibration_overlay_rows(calibration_rows)
        summary = build_shared_family_calibration_summary(calibration_rows, overlay_rows)

        by_topic = {row["overlay_topic"]: row for row in overlay_rows}
        self.assertEqual(by_topic["shared_family_dsp_expectation"]["calibration_status"], "implementation_dependent")
        self.assertEqual(by_topic["shared_family_numeric_projection_boundary"]["calibration_status"], "too_uncertain_for_numeric_trust")

        joined = " ".join(summary["summary_lines"])
        self.assertIn("calibration aid", summary["headline"])
        self.assertIn("caution band", joined)
        self.assertIn("implementation-dependent", joined)
        self.assertIn("modelled/anchored", joined)
        self.assertIn("not as a direct replacement", joined)

    def test_measured_utility_rows_capture_bottleneck_specific_reading(self) -> None:
        tradeoff_rows = build_direct_tradeoff_rows(self._two_scale_input_rows())
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        by_key = {(row["grid"], row["architecture_variant"]): row for row in utility_rows}

        self.assertEqual(by_key[("4x4", "baseline")]["utility_status"], "performance_first_default")
        self.assertEqual(by_key[("4x4", "shared_lut_saving")]["resource_relief_kind"], "lut_relief")
        self.assertEqual(by_key[("4x4", "shared_dsp_reducing")]["resource_relief_kind"], "dsp_relief")
        self.assertAlmostEqual(by_key[("8x4", "shared_lut_saving")]["throughput_retention_pct"], 50.769231, places=5)
        self.assertAlmostEqual(by_key[("8x4", "shared_dsp_reducing")]["latency_increase_factor"], 1.969697, places=5)

    def test_measured_bottleneck_choice_map_prefers_expected_variants(self) -> None:
        tradeoff_rows = build_direct_tradeoff_rows(self._two_scale_input_rows())
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        choice_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
        by_key = {(row["grid"], row["bottleneck_kind"]): row for row in choice_rows}

        self.assertEqual(by_key[("4x4", "lut")]["preferred_variant"], "shared_lut_saving")
        self.assertEqual(by_key[("8x4", "dsp")]["preferred_variant"], "shared_dsp_reducing")
        self.assertEqual(by_key[("4x4", "performance")]["preferred_variant"], "baseline")
        self.assertEqual(by_key[("8x4", "no_hard_resource_bottleneck")]["preferred_variant"], "baseline")

    def test_measured_utility_summary_uses_design_utility_language(self) -> None:
        tradeoff_rows = build_direct_tradeoff_rows(self._two_scale_input_rows())
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        choice_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
        summary = build_measured_utility_summary(utility_rows, choice_rows)

        joined = " ".join(summary["summary_lines"])
        self.assertIn("performance dominates", joined)
        self.assertIn("bottleneck-specific relief mechanisms", joined)
        self.assertIn("Flexibility pays only when the relieved bottleneck matters more than the lost performance", joined)

    def test_measured_flexibility_overhead_rows_capture_shared_overhead(self) -> None:
        tradeoff_rows = build_direct_tradeoff_rows(self._two_scale_input_rows())
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        overhead_rows = build_measured_flexibility_overhead_rows(tradeoff_rows, utility_rows)
        by_key = {(row["grid"], row["architecture_variant"]): row for row in overhead_rows}

        self.assertEqual(by_key[("4x4", "shared_lut_saving")]["overhead_kind"], "latency_and_throughput_penalty")
        self.assertEqual(by_key[("4x4", "shared_lut_saving")]["justification_status"], "justified_only_for_lut_dominant_relief")
        self.assertEqual(by_key[("8x4", "shared_dsp_reducing")]["justification_status"], "justified_only_for_dsp_dominant_relief")
        self.assertEqual(by_key[("8x4", "shared_dsp_reducing")]["overhead_severity"], "high")

    def test_measured_flexibility_justification_table_prefers_expected_variants(self) -> None:
        tradeoff_rows = build_direct_tradeoff_rows(self._two_scale_input_rows())
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        bottleneck_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
        overhead_rows = build_measured_flexibility_overhead_rows(tradeoff_rows, utility_rows)
        justification_rows = build_measured_flexibility_justification_table(overhead_rows, bottleneck_rows)
        by_key = {(row["grid"], row["dominant_condition"]): row for row in justification_rows}

        self.assertEqual(by_key[("4x4", "lut_dominant_need")]["preferred_variant"], "shared_lut_saving")
        self.assertEqual(by_key[("8x4", "dsp_dominant_need")]["preferred_variant"], "shared_dsp_reducing")
        self.assertEqual(by_key[("4x4", "performance_or_latency_dominant")]["preferred_variant"], "baseline")
        self.assertEqual(by_key[("8x4", "no_hard_resource_bottleneck")]["preferred_variant"], "baseline")

    def test_measured_design_rule_extraction_summary_uses_thesis_language(self) -> None:
        tradeoff_rows = build_direct_tradeoff_rows(self._two_scale_input_rows())
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        bottleneck_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
        overhead_rows = build_measured_flexibility_overhead_rows(tradeoff_rows, utility_rows)
        justification_rows = build_measured_flexibility_justification_table(overhead_rows, bottleneck_rows)
        summary = build_measured_design_rule_extraction_summary(overhead_rows, justification_rows)

        joined = " ".join(summary["summary_lines"])
        self.assertIn("Flexibility introduces real overhead", joined)
        self.assertIn("justified only when the relieved bottleneck matters more than the overhead it introduces", joined)
        self.assertIn("not a free win", joined)
        self.assertIn("Implementation style determines both the relieved bottleneck and the overhead paid", joined)

    def test_measured_predictor_rows_mark_exact_and_caution_metrics(self) -> None:
        rows = self._three_scale_actual_input_rows()
        predictor_rows = build_measured_predictor_rows(rows)
        by_key = {(row["architecture_variant"], row["metric"]): row for row in predictor_rows}

        self.assertEqual(by_key[("baseline", "dsp")]["fit_status"], "exact_validated_formula")
        self.assertEqual(by_key[("shared_dsp_reducing", "dsp")]["predictor_formula"], "dsp = 0")
        self.assertEqual(by_key[("baseline", "latency_cycles")]["max_abs_residual"], 0.0)
        self.assertEqual(by_key[("shared_lut_saving", "effective_throughput_ops_per_cycle")]["max_abs_residual"], 0.0)
        self.assertEqual(by_key[("baseline", "wns_ns")]["fit_status"], "too_unstable_for_trusted_prediction")

    def test_measured_fit_residual_rows_cover_measured_lattice(self) -> None:
        rows = self._three_scale_actual_input_rows()
        predictor_rows = build_measured_predictor_rows(rows)
        residual_rows = build_measured_fit_residual_rows(rows, predictor_rows)

        baseline_latency = [
            row for row in residual_rows
            if row["architecture_variant"] == "baseline" and row["metric"] == "latency_cycles"
        ]
        self.assertEqual(len(baseline_latency), 3)
        self.assertTrue(all(row["residual"] == 0.0 for row in baseline_latency))

    def test_measured_predictor_summary_and_boundary_expose_trust_limits(self) -> None:
        rows = self._three_scale_actual_input_rows()
        predictor_rows = build_measured_predictor_rows(rows)
        summary = build_measured_predictor_summary(predictor_rows)
        boundary = build_measured_extrapolation_boundary_summary(predictor_rows)

        joined = " ".join(summary["summary_lines"])
        boundary_joined = " ".join(boundary["summary_lines"])
        self.assertIn("DSP, latency, and throughput are well fit", joined)
        self.assertIn("WNS remains the least stable metric", joined)
        self.assertEqual(boundary["interpolation_domain_mac_units_min"], 16)
        self.assertEqual(boundary["interpolation_domain_mac_units_max"], 64)
        self.assertIn("interpolation is bounded", boundary_joined.lower())
        self.assertIn("unvalidated extrapolation", boundary_joined)

    def test_measured_decision_surface_prefers_expected_variants_and_tags_trust(self) -> None:
        rows = self._three_scale_actual_input_rows()
        tradeoff_rows = build_direct_tradeoff_rows(self._two_scale_input_rows())
        predictor_rows = build_measured_predictor_rows(rows)
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        bottleneck_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
        decision_rows = build_measured_decision_surface(predictor_rows, utility_rows, bottleneck_rows)

        by_key = {(row["mac_units"], row["regime_id"]): row for row in decision_rows}
        self.assertEqual(by_key[(16, "lut_budget_tight_relaxed_performance")]["preferred_variant"], "shared_lut_saving")
        self.assertEqual(by_key[(32, "dsp_budget_tight_relaxed_performance")]["preferred_variant"], "shared_dsp_reducing")
        self.assertEqual(by_key[(64, "no_hard_resource_bottleneck")]["preferred_variant"], "baseline")
        self.assertEqual(by_key[(48, "lut_budget_tight_relaxed_performance")]["trust_status"], "interpolated_within_measured_domain")
        self.assertEqual(by_key[(32, "timing_margin_sensitive")]["decision_status"], "unsupported_due_to_wns_instability")

    def test_measured_budget_boundaries_and_supported_region_map_keep_domain_bounded(self) -> None:
        rows = self._three_scale_actual_input_rows()
        predictor_rows = build_measured_predictor_rows(rows)
        boundary_rows = build_measured_budget_boundary_rows(predictor_rows)
        tradeoff_rows = build_direct_tradeoff_rows(self._two_scale_input_rows())
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        bottleneck_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
        decision_rows = build_measured_decision_surface(
            predictor_rows,
            utility_rows,
            bottleneck_rows,
        )
        supported_region_map = build_measured_supported_region_map(decision_rows, boundary_rows)

        self.assertTrue(any(row["mac_units"] == 15 and row["trust_status"] == "unsupported_extrapolation" for row in boundary_rows))
        self.assertTrue(any(row["mac_units"] == 65 and row["trust_status"] == "unsupported_extrapolation" for row in boundary_rows))
        joined = " ".join(supported_region_map["summary_lines"])
        self.assertIn("shared_lut_saving owns the LUT-only window", joined)
        self.assertIn("refuse to claim", joined)

    def test_measured_regime_transfer_summary_uses_bounded_surface_language(self) -> None:
        rows = self._three_scale_actual_input_rows()
        tradeoff_rows = build_direct_tradeoff_rows(self._two_scale_input_rows())
        predictor_rows = build_measured_predictor_rows(rows)
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        bottleneck_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
        decision_rows = build_measured_decision_surface(predictor_rows, utility_rows, bottleneck_rows)
        boundary_rows = build_measured_budget_boundary_rows(predictor_rows)
        summary = build_measured_regime_transfer_summary(decision_rows, boundary_rows, predictor_rows)

        joined = " ".join(summary["summary_lines"])
        self.assertIn("bounded to the validated direct-slice interpolation domain", joined)
        self.assertIn("Baseline remains dominant whenever the design is performance-first", joined)
        self.assertIn("Timing-sensitive transfer is intentionally excluded", joined)
        self.assertIn("refuse to claim a predictor-backed architecture choice", joined)

    def test_final_results_tables_and_summary_capture_thesis_grade_boundary(self) -> None:
        direct_rows = self._two_scale_input_rows()
        actual_rows = self._three_scale_actual_input_rows()
        tradeoff_rows = build_direct_tradeoff_rows(direct_rows)
        scaling_summary = build_direct_shared_scaling_summary(tradeoff_rows)
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        bottleneck_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
        flexibility_rows = build_measured_flexibility_overhead_rows(tradeoff_rows, utility_rows)
        flexibility_justification_rows = build_measured_flexibility_justification_table(flexibility_rows, bottleneck_rows)
        predictor_rows = build_measured_predictor_rows(actual_rows)
        boundary_summary = build_measured_extrapolation_boundary_summary(predictor_rows)
        decision_rows = build_measured_decision_surface(predictor_rows, utility_rows, bottleneck_rows)
        boundary_rows = build_measured_budget_boundary_rows(predictor_rows)
        regime_transfer_summary = build_measured_regime_transfer_summary(decision_rows, boundary_rows, predictor_rows)
        support_rows = build_measured_support_rows(tradeoff_rows)
        calibration_rows = build_framework_calibration_aid_rows(direct_rows, tradeoff_rows)
        calibration_overlay_rows = build_framework_calibration_overlay_rows(calibration_rows)

        design_rule_rows = build_final_design_rule_table(tradeoff_rows, flexibility_justification_rows, scaling_summary)
        trust_rows = build_final_trust_calibration_table(
            support_rows,
            calibration_overlay_rows,
            predictor_rows,
            boundary_summary,
        )
        choice_boundary_rows = build_final_architecture_choice_boundary_table(boundary_rows, regime_transfer_summary)
        summary = build_final_results_summary(design_rule_rows, trust_rows, choice_boundary_rows, regime_transfer_summary)

        self.assertTrue(any(row["rule_id"] == "shared_lut_saving_lut_only_rule" for row in design_rule_rows))
        self.assertTrue(any(row["topic"] == "wns_numeric_use" and row["predictor_status"] == "caution_only_local_fit" for row in trust_rows))
        self.assertTrue(any(row["boundary_id"] == "outside_domain_refusal" and row["trust_status"] == "unsupported_extrapolation" for row in choice_boundary_rows))
        joined = " ".join(summary["summary_lines"])
        self.assertIn("bounded", joined)
        self.assertIn("local interpolation is acceptable", joined)
        self.assertIn("explicitly refuses unsupported extrapolation", joined)

    def test_final_artifact_index_and_reproducibility_guide_use_consistent_boundary_language(self) -> None:
        direct_rows = self._two_scale_input_rows()
        actual_rows = self._three_scale_actual_input_rows()
        tradeoff_rows = build_direct_tradeoff_rows(direct_rows)
        scaling_summary = build_direct_shared_scaling_summary(tradeoff_rows)
        utility_rows = build_measured_utility_rows(tradeoff_rows)
        bottleneck_rows = build_measured_bottleneck_choice_map(utility_rows, tradeoff_rows)
        flexibility_rows = build_measured_flexibility_overhead_rows(tradeoff_rows, utility_rows)
        flexibility_justification_rows = build_measured_flexibility_justification_table(flexibility_rows, bottleneck_rows)
        predictor_rows = build_measured_predictor_rows(actual_rows)
        boundary_summary = build_measured_extrapolation_boundary_summary(predictor_rows)
        decision_rows = build_measured_decision_surface(predictor_rows, utility_rows, bottleneck_rows)
        boundary_rows = build_measured_budget_boundary_rows(predictor_rows)
        regime_transfer_summary = build_measured_regime_transfer_summary(decision_rows, boundary_rows, predictor_rows)
        support_rows = build_measured_support_rows(tradeoff_rows)
        calibration_rows = build_framework_calibration_aid_rows(direct_rows, tradeoff_rows)
        calibration_overlay_rows = build_framework_calibration_overlay_rows(calibration_rows)
        design_rule_rows = build_final_design_rule_table(tradeoff_rows, flexibility_justification_rows, scaling_summary)
        trust_rows = build_final_trust_calibration_table(support_rows, calibration_overlay_rows, predictor_rows, boundary_summary)
        choice_boundary_rows = build_final_architecture_choice_boundary_table(boundary_rows, regime_transfer_summary)
        summary = build_final_results_summary(design_rule_rows, trust_rows, choice_boundary_rows, regime_transfer_summary)
        artifact_index_rows = build_final_artifact_index(REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice", summary)
        reproducibility_guide = build_final_reproducibility_guide(summary)

        self.assertTrue(any(row["filename"] == "final_decision_surface_figures/timing_sensitive_unsupported.png" and row["measured_interpolation_status"] == "unsupported_extrapolation" for row in artifact_index_rows))
        self.assertTrue(any("validated domain" in row["thesis_use_note"] or "Use in" in row["thesis_use_note"] for row in artifact_index_rows))
        joined = " ".join(reproducibility_guide["summary_lines"])
        self.assertIn("Interpolated within measured domain", joined)
        self.assertIn("Unsupported extrapolation", joined)
        self.assertIn("WNS remains caution-only", joined)

    def test_final_manifest_references_existing_generated_artifacts(self) -> None:
        output_dir = REPO_ROOT / "results" / "fpga" / "framework_v2" / "direct_slice"
        manifest = json.loads((output_dir / "final_results_pack_manifest.json").read_text())
        index_rows = json.loads((output_dir / "final_artifact_index.json").read_text())

        for group_key in (
            "final_tradeoff_figures",
            "final_predictor_validation_figures",
            "final_decision_surface_figures",
        ):
            for path_str in manifest[group_key]:
                self.assertTrue(Path(path_str).exists(), path_str)
        self.assertEqual(manifest["regeneration_command"], "make fpga_mac_direct_final_pack")
        self.assertIn("validated_domain", manifest)
        for row in index_rows:
            self.assertTrue(Path(row["absolute_path"]).exists(), row["absolute_path"])
        timing_row = next(row for row in index_rows if row["filename"] == "final_decision_surface_figures/timing_sensitive_unsupported.png")
        self.assertEqual(timing_row["measured_interpolation_status"], "unsupported_extrapolation")


if __name__ == "__main__":
    unittest.main()
