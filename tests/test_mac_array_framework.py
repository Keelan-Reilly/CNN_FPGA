#!/usr/bin/env python3
"""Deterministic tests for the MAC-array framework v2 logic."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from analysis.mac_array_adaptive import break_even_phase_cycles, evaluate_adaptive_workload
from analysis.mac_array_evidence import load_evidence, provenance_summary_rows, static_field_provenance, switching_pair_record
from analysis.mac_array_metrics import derive_static_row, summarize_workload
from analysis.mac_array_policy import evaluate_policy
from analysis.mac_array_types import ConstraintSpec, GridSpec
from analysis.mac_array_workloads import workloads_from_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "mac_array_framework_v2.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


class FrameworkLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        cfg = _load_config()
        self.grid = GridSpec(label="8x8", rows=8, cols=8)
        self.evidence = load_evidence(REPO_ROOT / cfg["evidence_path"])
        self.architectures = self.evidence.architectures
        self.workloads = {w.name: w for w in workloads_from_config(cfg["workloads"])}

    def test_shared_8x8_uses_measured_dsp_anchor(self) -> None:
        summary, _ = summarize_workload(
            self.grid,
            self.architectures["shared"],
            self.workloads["dense_steady"],
            self.evidence,
        )
        self.assertEqual(summary.dsp, 32)

    def test_replicated_is_marked_timing_infeasible_on_8x8(self) -> None:
        summary, _ = summarize_workload(
            self.grid,
            self.architectures["replicated"],
            self.workloads["dense_steady"],
            self.evidence,
        )
        self.assertFalse(summary.timing_feasible)
        self.assertIn("failed", summary.implementation_status)

    def test_provenance_exposes_anchor_source(self) -> None:
        row = derive_static_row(self.grid, self.architectures["shared"], self.evidence)
        self.assertEqual(row["dsp_provenance_kind"], "anchored_prior_study_fact")
        self.assertEqual(row["dsp_source_id"], "anchor_8x8_shared_dsp_prior_study")

    def test_evidence_summary_counts_are_present(self) -> None:
        counts = {row["value_kind"]: row["count"] for row in provenance_summary_rows(self.evidence)}
        self.assertIn("analytical_linear_model", counts)
        self.assertIn("anchored_prior_study_fact", counts)

    def test_break_even_requires_positive_improvement(self) -> None:
        self.assertIsNone(break_even_phase_cycles(2.0, 2.5, 100))
        self.assertEqual(break_even_phase_cycles(3.0, 2.0, 100), 200)

    def test_adaptive_sequence_tracks_phase_changes(self) -> None:
        result = evaluate_adaptive_workload(
            grid=self.grid,
            workload=self.workloads["phase_changing"],
            architectures=self.architectures,
            allowed_modes=("shared", "baseline", "replicated"),
            evidence=self.evidence,
        )
        self.assertEqual(len(result["mode_sequence"]), 3)
        self.assertIn("adaptive_provenance_kind", result)
        self.assertGreaterEqual(result["transitions"], 0)

    def test_policy_prefers_shared_under_tight_budget(self) -> None:
        dense = self.workloads["dense_steady"]
        fixed_rows = []
        for name in ("baseline", "shared", "replicated"):
            summary, _ = summarize_workload(self.grid, self.architectures[name], dense, self.evidence)
            fixed_rows.append(summary.__dict__)
        adaptive = evaluate_adaptive_workload(
            grid=self.grid,
            workload=dense,
            architectures=self.architectures,
            allowed_modes=("shared", "baseline", "replicated"),
            evidence=self.evidence,
        )
        decision, diagnostics = evaluate_policy(
            fixed_rows=fixed_rows,
            adaptive_row=adaptive,
            constraint=ConstraintSpec(
                name="tight",
                dsp_budget=40,
                lut_budget=1800,
                min_throughput_ops_per_cycle=14.0,
                target_grid="8x8",
            ),
            workload_class=dense.workload_class,
        )
        self.assertEqual(decision["recommendation"], "shared")
        self.assertTrue(any(item["candidate"] == "baseline" for item in diagnostics))

    def test_switching_pairs_are_mode_aware(self) -> None:
        shared_to_base = switching_pair_record(self.evidence, "shared", "baseline")
        shared_to_rep = switching_pair_record(self.evidence, "shared", "replicated")
        self.assertLess(shared_to_base["switch_cycles"], shared_to_rep["switch_cycles"])

    def test_static_field_provenance_comes_from_registry(self) -> None:
        prov = static_field_provenance(self.evidence, "replicated", "8x8", "implementation_status")
        self.assertEqual(prov.source_id, "anchor_8x8_replicated_impl_status_prior_study")


if __name__ == "__main__":
    unittest.main()
