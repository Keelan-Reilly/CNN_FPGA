#!/usr/bin/env python3
"""Deterministic tests for bounded regime-map generation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from analysis.mac_array_evidence import load_evidence
from analysis.mac_array_regime import build_regime_map
from analysis.mac_array_adaptive import evaluate_adaptive_workload
from analysis.mac_array_metrics import summarize_workload
from analysis.mac_array_types import GridSpec
from analysis.mac_array_workloads import workloads_from_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "mac_array_framework_v2.json"


class RegimeMapTest(unittest.TestCase):
    def setUp(self) -> None:
        cfg = json.loads(CONFIG_PATH.read_text())
        self.evidence = load_evidence(REPO_ROOT / cfg["evidence_path"])
        self.architectures = self.evidence.architectures
        self.grids = [GridSpec(label=item["label"], rows=item["rows"], cols=item["cols"]) for item in cfg["grids"]]
        self.workloads = workloads_from_config(cfg["workloads"])

        self.workload_rows = []
        self.grid_map = {grid.label: grid for grid in self.grids}
        self.workload_map = {workload.name: workload for workload in self.workloads}
        for grid in self.grids:
            for workload in self.workloads:
                for arch in self.architectures.values():
                    summary, _ = summarize_workload(grid, arch, workload, self.evidence)
                    row = summary.__dict__.copy()
                    row["workload_class"] = workload.workload_class
                    row["phase_count"] = workload.phase_count
                    row["dominant_utilization"] = workload.dominant_utilization
                    row["utilization_variance"] = workload.utilization_variance
                    row["burstiness"] = workload.burstiness
                    self.workload_rows.append(row)

    def test_regime_map_is_bounded_and_nonempty(self) -> None:
        def adaptive_factory(grid_label, workload_name, constraint):
            fixed_rows = [
                row for row in self.workload_rows if row["grid"] == grid_label and row["workload"] == workload_name
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
            return evaluate_adaptive_workload(
                grid=self.grid_map[grid_label],
                workload=self.workload_map[workload_name],
                architectures=self.architectures,
                allowed_modes=tuple(feasible_modes),
                evidence=self.evidence,
            )

        regime_rows, regime_summary_rows, regime_meta = build_regime_map(self.workload_rows, adaptive_factory)
        self.assertGreater(len(regime_rows), 10)
        self.assertLess(len(regime_rows), 200)
        self.assertTrue(regime_summary_rows)
        self.assertIn("adaptive_has_win_region", regime_meta)
        self.assertTrue(all("winner" in row for row in regime_rows))


if __name__ == "__main__":
    unittest.main()
