#!/usr/bin/env python3
"""Deterministic tests for the resource-aware Vivado scheduler helpers."""

from __future__ import annotations

import unittest

from experiments.vivado_scheduler import (
    ResourceSnapshot,
    SchedulerConfig,
    can_launch_more,
    scheduler_summary_rows,
)


class SchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SchedulerConfig(
            enabled=True,
            dry_run=True,
            max_concurrent_jobs=2,
            cpu_utilization_threshold_pct=85.0,
            min_free_mem_gb=4.0,
            per_job_mem_gb=8.0,
            poll_interval_sec=1.0,
        )

    def test_can_launch_accepts_safe_snapshot(self) -> None:
        ok, reason = can_launch_more(
            ResourceSnapshot(cpu_utilization_pct=20.0, available_mem_gb=20.0),
            running_jobs=1,
            cfg=self.cfg,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "launch_allowed")

    def test_can_launch_blocks_on_cpu(self) -> None:
        ok, reason = can_launch_more(
            ResourceSnapshot(cpu_utilization_pct=95.0, available_mem_gb=20.0),
            running_jobs=0,
            cfg=self.cfg,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "cpu_threshold_exceeded")

    def test_can_launch_blocks_on_memory(self) -> None:
        ok, reason = can_launch_more(
            ResourceSnapshot(cpu_utilization_pct=10.0, available_mem_gb=10.0),
            running_jobs=0,
            cfg=self.cfg,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "memory_reserve_violation")

    def test_queue_preview_is_deterministic(self) -> None:
        rows = scheduler_summary_rows(
            [
                {"run_id": "a", "run_dir": "results/fpga/runs/x/a", "clock_period_ns": 10},
                {"run_id": "b", "run_dir": "results/fpga/runs/x/b", "clock_period_ns": 8},
            ],
            self.cfg,
        )
        self.assertEqual([row["queue_index"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["run_id"], "a")
        self.assertEqual(rows[1]["run_id"], "b")


if __name__ == "__main__":
    unittest.main()
