#!/usr/bin/env python3
"""Generate direct measured-vs-modelled MAC-array slice artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .mac_array_direct_slice import (
        build_direct_calibration_summary,
        build_direct_slice_comparison_rows,
        render_direct_calibration_plot,
        render_direct_calibration_summary,
        render_direct_slice_summary,
    )
    from .mac_array_report import write_csv, write_json
except ImportError:
    from mac_array_direct_slice import (
        build_direct_calibration_summary,
        build_direct_slice_comparison_rows,
        render_direct_calibration_plot,
        render_direct_calibration_summary,
        render_direct_slice_summary,
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
    write_csv(output_dir / "direct_measured_vs_modelled.csv", rows)
    write_json(output_dir / "direct_measured_vs_modelled.json", rows)
    write_json(output_dir / "direct_calibration_summary.json", summary)
    render_direct_slice_summary(output_dir / "direct_slice_summary.md", rows, summary)
    render_direct_calibration_summary(output_dir / "direct_calibration_summary.md", summary)
    plot_path = render_direct_calibration_plot(output_dir / "direct_calibration_plot.png", rows)
    write_json(output_dir / "direct_generated_plot.json", plot_path)
    print(f"Wrote direct-slice outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
