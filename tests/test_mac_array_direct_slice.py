#!/usr/bin/env python3
"""Deterministic tests for the direct MAC-array slice plumbing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from analysis.mac_array_direct_slice import (
    build_direct_calibration_summary,
    build_direct_slice_comparison_rows,
    direct_latency_model,
    direct_throughput_model,
)
from run_fpga_experiments import SUPPORTED_SWEEP_PARAMS


class DirectMacSliceTest(unittest.TestCase):
    def test_direct_slice_models_are_simple_and_stable(self) -> None:
        self.assertEqual(direct_latency_model(32), 33)
        self.assertAlmostEqual(direct_throughput_model(4, 4, 32), 512.0 / 33.0)

    def test_supported_params_include_direct_slice_knobs(self) -> None:
        self.assertIn("ARRAY_ROWS", SUPPORTED_SWEEP_PARAMS)
        self.assertIn("ARRAY_COLS", SUPPORTED_SWEEP_PARAMS)
        self.assertIn("K_DEPTH", SUPPORTED_SWEEP_PARAMS)


    def test_calibration_summary_reports_exact_matches_and_lut_fit(self) -> None:
        rows = [
            {
                "grid": "4x4",
                "mac_units": 16,
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
                "comparison_status": "direct_measured_vs_modelled",
                "direct_evidence_source": "results/fpga/aggregates/study_mac_array_direct_baseline_8x8.json",
                "dsp_delta": 0,
                "latency_delta_cycles": 0,
                "throughput_delta_ops_per_cycle": 0.0,
                "lut_delta": 2187,
                "measured_lut": 4287,
                "measured_wns_ns": 0.634,
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
                    "lut": 700,
                    "wns_ns": 0.25,
                    "latency_cycles": 33,
                    "effective_throughput_ops_per_cycle": 15.5,
                    "params": '{"ARRAY_ROWS": 4, "ARRAY_COLS": 4, "K_DEPTH": 32}',
                }
            ]
        }
        rows = build_direct_slice_comparison_rows(payload)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["direct_evidence_kind"], "direct_measured_mac_array_slice")
        self.assertEqual(row["comparison_status"], "direct_measured_vs_modelled")
        self.assertEqual(row["grid"], "4x4")


if __name__ == "__main__":
    unittest.main()
