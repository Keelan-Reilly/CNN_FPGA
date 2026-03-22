#!/usr/bin/env python3
"""Deterministic tests for measured-refresh manifest and comparison helpers."""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from analysis.mac_array_refresh import (
    build_measured_model_comparison,
    build_measured_refresh_manifest,
    build_refresh_queue,
    materialize_refresh_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class MeasuredRefreshTest(unittest.TestCase):
    def setUp(self) -> None:
        self.regime_rows = [
            {
                "grid": "4x4",
                "workload": "dense_steady",
                "budget_class": "tight_shared_only",
                "throughput_class": "balanced",
                "winner": "shared",
                "winner_reason": "shared win",
                "runner_up": "baseline",
                "adaptive_candidate_present": False,
                "adaptive_rejected_reasons": "",
            },
            {
                "grid": "4x4",
                "workload": "dense_steady",
                "budget_class": "baseline_fit",
                "throughput_class": "balanced",
                "winner": "baseline",
                "winner_reason": "baseline win",
                "runner_up": "shared",
                "adaptive_candidate_present": True,
                "adaptive_rejected_reasons": "adaptive_gain_too_small,no_switching_observed",
            },
            {
                "grid": "4x4",
                "workload": "phase_changing",
                "budget_class": "expanded_headroom",
                "throughput_class": "stretch",
                "winner": "replicated",
                "winner_reason": "replicated win",
                "runner_up": "baseline",
                "adaptive_candidate_present": True,
                "adaptive_rejected_reasons": "adaptive_gain_too_small",
            },
        ]
        self.rejection_surface_rows = [
            {
                "grid": "4x4",
                "workload": "dense_steady",
                "adaptive_candidate_points": 3,
                "adaptive_feasible_points": 0,
                "adaptive_win_points": 0,
                "dominant_rejection_reason": "adaptive_gain_too_small",
                "dominant_rejection_share_of_regime_points": 0.66,
                "adaptive_win_region_present": False,
            },
            {
                "grid": "4x4",
                "workload": "phase_changing",
                "adaptive_candidate_points": 4,
                "adaptive_feasible_points": 0,
                "adaptive_win_points": 0,
                "dominant_rejection_reason": "adaptive_gain_too_small",
                "dominant_rejection_share_of_regime_points": 0.5,
                "adaptive_win_region_present": False,
            },
        ]
        self.temp_dir = REPO_ROOT / "results" / "fpga" / "framework_v2" / "_tmp_refresh_test"
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_manifest_is_grounded_in_regime_rows(self) -> None:
        manifest = build_measured_refresh_manifest(self.regime_rows, self.rejection_surface_rows)
        candidate_ids = {row["candidate_id"] for row in manifest}
        self.assertIn("shared_dominant_proxy", candidate_ids)
        self.assertIn("baseline_dominant_proxy", candidate_ids)
        self.assertIn("replicated_edge_proxy", candidate_ids)
        self.assertIn("adaptive_near_miss", candidate_ids)
        adaptive = next(row for row in manifest if row["candidate_id"] == "adaptive_near_miss")
        self.assertFalse(adaptive["vivado_runnable"])
        self.assertEqual(adaptive["mapping_status"], "not_directly_measurable_with_current_rtl")

    def test_materialized_queue_uses_single_run_configs(self) -> None:
        manifest = build_measured_refresh_manifest(self.regime_rows, self.rejection_surface_rows)
        queue = build_refresh_queue(manifest)
        materialized = materialize_refresh_configs(queue, self.temp_dir)
        self.assertTrue(all(row["config_materialized"] for row in materialized))
        for row in materialized:
            cfg_path = REPO_ROOT / row["generated_config_path"]
            payload = json.loads(cfg_path.read_text())
            self.assertEqual(len(payload["runs"]), 1)
            self.assertEqual(payload["runs"][0]["run_id"], row["run_id_hint"])

    def test_comparison_honestly_marks_unmeasurable_candidate(self) -> None:
        manifest = build_measured_refresh_manifest(self.regime_rows, self.rejection_surface_rows)
        comparisons = build_measured_model_comparison(manifest)
        adaptive = next(row for row in comparisons if row["candidate_id"] == "adaptive_near_miss")
        baseline = next(row for row in comparisons if row["candidate_id"] == "baseline_dominant_proxy")
        self.assertEqual(adaptive["comparison_status"], "not_directly_comparable")
        self.assertEqual(baseline["agreement"], "consistent_with_proxy")


if __name__ == "__main__":
    unittest.main()
