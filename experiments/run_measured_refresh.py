#!/usr/bin/env python3
"""Generate and optionally preview/run selective measured-refresh jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.mac_array_refresh import (  # noqa: E402
    build_measured_model_comparison,
    build_measured_refresh_manifest,
    build_refresh_queue,
    materialize_refresh_configs,
    render_comparison_summary,
)
from analysis.mac_array_report import write_csv, write_json  # noqa: E402


RUNNER = REPO_ROOT / "experiments" / "run_fpga_experiments.py"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _runner_command(queue_row: dict[str, Any], dry_run: bool, passthrough_args: list[str]) -> list[str]:
    config_path = Path(queue_row["generated_config_path"])
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(config_path),
    ]
    if dry_run:
        command.append("--dry-run")
    command.extend(passthrough_args)
    return command


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate and optionally refresh selective measured proxy points")
    ap.add_argument(
        "--framework-dir",
        default=str(REPO_ROOT / "results" / "fpga" / "framework_v2"),
        help="Framework-v2 output directory",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Refresh output directory; defaults to <framework-dir>/measured_refresh",
    )
    ap.add_argument(
        "--preview-scheduler",
        action="store_true",
        help="Preview selected refresh jobs through the existing scheduler-backed Vivado runner",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute selected refresh jobs through the existing Vivado runner",
    )
    args, passthrough_args = ap.parse_known_args()

    framework_dir = Path(args.framework_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else framework_dir / "measured_refresh"
    output_dir.mkdir(parents=True, exist_ok=True)

    regime_rows = _load_json(framework_dir / "regime_map.json")
    rejection_surface_rows = _load_json(framework_dir / "adaptive_rejection_surface.json")

    manifest_rows = build_measured_refresh_manifest(regime_rows, rejection_surface_rows)
    queue_rows = build_refresh_queue(manifest_rows)
    materialized_queue_rows = materialize_refresh_configs(queue_rows, output_dir / "generated_configs")
    comparison_rows = build_measured_model_comparison(manifest_rows)

    write_csv(output_dir / "measured_refresh_manifest.csv", manifest_rows)
    write_json(output_dir / "measured_refresh_manifest.json", manifest_rows)
    write_csv(output_dir / "measured_refresh_queue.csv", materialized_queue_rows)
    write_json(output_dir / "measured_refresh_queue.json", materialized_queue_rows)
    write_csv(output_dir / "measured_model_comparison.csv", comparison_rows)
    write_json(output_dir / "measured_model_comparison.json", comparison_rows)
    render_comparison_summary(output_dir / "comparison_summary.md", manifest_rows, comparison_rows)

    if args.preview_scheduler or args.execute:
        run_records = []
        for queue_row in materialized_queue_rows:
            if not queue_row.get("config_materialized"):
                continue
            command = _runner_command(queue_row, dry_run=not args.execute, passthrough_args=passthrough_args)
            completed = subprocess.run(command, cwd=REPO_ROOT)
            run_records.append(
                {
                    "experiment_id": queue_row["experiment_id"],
                    "config_path": queue_row["generated_config_path"],
                    "dry_run": not args.execute,
                    "returncode": completed.returncode,
                    "command": " ".join(command),
                }
            )
        write_json(output_dir / "refresh_runner_invocations.json", run_records)

    print(f"Wrote measured-refresh outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
