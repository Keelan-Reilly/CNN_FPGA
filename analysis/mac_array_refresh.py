#!/usr/bin/env python3
"""Selective measured-refresh helpers for framework-v2 regime points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .fpga_results import REPO_ROOT
except ImportError:
    from fpga_results import REPO_ROOT


EXPERIMENT_CONFIG_BY_ID = {
    "study_baseline_characterization": "experiments/configs/study_baseline_characterization.json",
    "study_dense_parallel_scaling": "experiments/configs/study_dense_parallel_scaling.json",
    "study_timing_target": "experiments/configs/study_timing_target.json",
    "study_precision_resource": "experiments/configs/study_precision_resource.json",
}


def _load_aggregate_rows(experiment_id: str) -> list[dict[str, Any]]:
    path = REPO_ROOT / "results" / "fpga" / "aggregates" / f"{experiment_id}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("runs", [])


def _normalize_params(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _first_matching(rows: list[dict[str, Any]], predicate: Any) -> dict[str, Any] | None:
    for row in rows:
        if predicate(row):
            return row
    return None


def _select_row(regime_rows: list[dict[str, Any]], winner: str, workload: str | None = None) -> dict[str, Any] | None:
    matches = [row for row in regime_rows if row["winner"] == winner]
    if workload is not None:
        preferred = [row for row in matches if row["workload"] == workload]
        if preferred:
            matches = preferred
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda item: (
            item["grid"],
            item["workload"],
            item["budget_class"],
            item["throughput_class"],
        ),
    )[0]


def build_measured_refresh_manifest(
    regime_rows: list[dict[str, Any]],
    rejection_surface_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    candidate_specs = [
        (
            "shared_dominant_proxy",
            _select_row(regime_rows, "shared"),
            "Select the smallest tight-budget shared win as the resource-first regime representative.",
            "proxy_only_current_cnn_rtl",
            "study_baseline_characterization",
            "baseline_dw16_fb7_par1_100mhz",
            True,
            "Current RTL can only proxy the low-resource operating point, not direct shared time-multiplexing.",
        ),
        (
            "baseline_dominant_proxy",
            _select_row(regime_rows, "baseline"),
            "Select the first baseline regime win where timing/throughput filters favor staying fixed.",
            "proxy_only_current_cnn_rtl",
            "study_timing_target",
            "target_100mhz",
            True,
            "Current RTL can refresh the timing-feasible baseline proxy but not a direct MAC-array baseline implementation.",
        ),
        (
            "replicated_edge_proxy",
            _select_row(regime_rows, "replicated", workload="phase_changing"),
            "Select the narrow replicated-survival regime point as the high-parallelism edge case.",
            "proxy_only_current_cnn_rtl",
            "study_dense_parallel_scaling",
            "sweep_DENSE_OUT_PAR10",
            True,
            "Current RTL can refresh a high-parallel proxy, not the replicated MAC-array itself.",
        ),
    ]

    adaptive_surface = sorted(
        [row for row in rejection_surface_rows if not row["adaptive_win_region_present"]],
        key=lambda item: (-item["adaptive_candidate_points"], item["grid"], item["workload"]),
    )
    adaptive_row = None
    if adaptive_surface:
        target = adaptive_surface[0]
        candidates = [
            row
            for row in regime_rows
            if row["grid"] == target["grid"]
            and row["workload"] == target["workload"]
            and row["adaptive_candidate_present"]
        ]
        adaptive_row = sorted(
            candidates,
            key=lambda item: (
                item["budget_class"],
                item["throughput_class"],
            ),
        )[0] if candidates else None
    candidate_specs.append(
        (
            "adaptive_near_miss",
            adaptive_row,
            "Select the phase-varying slice with the strongest adaptive candidacy but no observed adaptive win region.",
            "not_directly_measurable_with_current_rtl",
            None,
            None,
            False,
            "Current Vivado flow has no direct adaptive or pre-built multi-mode MAC-array RTL to refresh.",
        )
    )

    for candidate_id, regime_row, selection_reason, mapping_status, experiment_id, run_id, runnable, meas_note in candidate_specs:
        if regime_row is None:
            continue
        manifest.append(
            {
                "candidate_id": candidate_id,
                "grid": regime_row["grid"],
                "workload": regime_row["workload"],
                "budget_class": regime_row["budget_class"],
                "throughput_class": regime_row["throughput_class"],
                "expected_winner": regime_row["winner"],
                "expected_winner_reason": regime_row["winner_reason"],
                "runner_up": regime_row["runner_up"],
                "adaptive_rejected_reasons": regime_row.get("adaptive_rejected_reasons", ""),
                "selection_reason": selection_reason,
                "mapping_status": mapping_status,
                "measurement_status": "checked_in_runnable_proxy" if runnable else "selection_only",
                "measured_vs_modelled_basis": "proxy_trend_check" if runnable else "not_directly_comparable",
                "proposed_experiment_id": experiment_id or "",
                "proposed_config_path": EXPERIMENT_CONFIG_BY_ID.get(experiment_id, "") if experiment_id else "",
                "proposed_run_id": run_id or "",
                "vivado_runnable": runnable,
                "measurability_note": meas_note,
                "selection_provenance_kind": "derived_from_regime_map",
                "selection_source_path": "results/fpga/framework_v2/regime_map.csv",
            }
        )
    return manifest


def build_refresh_queue(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in manifest_rows:
        if not row["vivado_runnable"]:
            continue
        key = (row["proposed_experiment_id"], row["proposed_run_id"])
        if key in seen:
            continue
        seen.add(key)
        queue.append(
            {
                "candidate_id": row["candidate_id"],
                "experiment_id": row["proposed_experiment_id"],
                "config_path": row["proposed_config_path"],
                "run_id_hint": row["proposed_run_id"],
                "mapping_status": row["mapping_status"],
                "queue_reason": row["selection_reason"],
            }
        )
    return queue


def materialize_refresh_configs(queue_rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_rows: list[dict[str, Any]] = []
    for row in queue_rows:
        experiment_id = row["experiment_id"]
        run_id = row["run_id_hint"]
        aggregate_rows = _load_aggregate_rows(experiment_id)
        measured_row = _first_matching(aggregate_rows, lambda item: str(item.get("run_id")) == run_id)
        if measured_row is None:
            materialized_rows.append({**row, "generated_config_path": "", "config_materialized": False})
            continue
        source_path = REPO_ROOT / EXPERIMENT_CONFIG_BY_ID[experiment_id]
        source_cfg = json.loads(source_path.read_text())
        generated_cfg = {
            "experiment_id": f"measured_refresh_{row['candidate_id']}",
            "vivado": dict(source_cfg["vivado"]),
            "clock_period_ns": measured_row.get("clock_period_ns", source_cfg.get("clock_period_ns")),
            "base_params": {},
            "runs": [
                {
                    "run_id": run_id,
                    "clock_period_ns": measured_row.get("clock_period_ns", source_cfg.get("clock_period_ns")),
                    "params": _normalize_params(measured_row.get("params")),
                }
            ],
        }
        generated_path = output_dir / f"{row['candidate_id']}.json"
        generated_path.write_text(json.dumps(generated_cfg, indent=2) + "\n")
        try:
            generated_path_label = str(generated_path.relative_to(REPO_ROOT))
        except ValueError:
            generated_path_label = str(generated_path)
        materialized_rows.append(
            {
                **row,
                "generated_config_path": generated_path_label,
                "config_materialized": True,
            }
        )
    return materialized_rows


def build_measured_model_comparison(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_rows = _load_aggregate_rows("study_baseline_characterization")
    dense_rows = _load_aggregate_rows("study_dense_parallel_scaling")
    timing_rows = _load_aggregate_rows("study_timing_target")

    baseline = baseline_rows[0] if baseline_rows else None
    dense_par10 = _first_matching(dense_rows, lambda row: str(row.get("run_id")) == "sweep_DENSE_OUT_PAR10")
    target_100 = _first_matching(timing_rows, lambda row: str(row.get("run_id")) == "target_100mhz")

    comparisons: list[dict[str, Any]] = []
    for row in manifest_rows:
        comparison = {
            "candidate_id": row["candidate_id"],
            "expected_winner": row["expected_winner"],
            "mapping_status": row["mapping_status"],
            "comparison_status": "not_directly_comparable",
            "agreement": "not_assessed",
            "measured_source_experiment": row["proposed_experiment_id"],
            "measured_source_run_id": row["proposed_run_id"],
            "model_scope": "bounded MAC-array regime model",
            "measured_scope": "existing CNN FPGA study proxy" if row["vivado_runnable"] else "no direct measured path",
            "comparison_note": row["measurability_note"],
        }

        if row["candidate_id"] == "shared_dominant_proxy" and baseline is not None and dense_par10 is not None:
            comparison.update(
                {
                    "comparison_status": "proxy_trend_compared",
                    "agreement": "partial_proxy_agreement",
                    "measured_lut_delta_vs_high_parallel": baseline["lut"] - dense_par10["lut"],
                    "measured_dsp_delta_vs_high_parallel": baseline["dsp"] - dense_par10["dsp"],
                    "measured_wns_ns": baseline["wns_ns"],
                    "comparison_note": (
                        "Tight-budget shared wins are only proxy-checkable today; the existing CNN low-parallel run "
                        "does confirm that the resource-minimizing operating point uses materially fewer LUT/DSP "
                        "than the high-parallel proxy, but it does not directly measure shared time-multiplexing."
                    ),
                }
            )
        elif row["candidate_id"] == "baseline_dominant_proxy" and target_100 is not None:
            comparison.update(
                {
                    "comparison_status": "proxy_trend_compared",
                    "agreement": "consistent_with_proxy",
                    "measured_lut": target_100["lut"],
                    "measured_dsp": target_100["dsp"],
                    "measured_wns_ns": target_100["wns_ns"],
                    "measured_throughput_inferences_per_sec": target_100["throughput_inferences_per_sec"],
                    "comparison_note": (
                        "The timing-feasible 100 MHz baseline proxy aligns with the model's claim that a fixed baseline-like "
                        "mode remains attractive once timing and throughput constraints are applied."
                    ),
                }
            )
        elif row["candidate_id"] == "replicated_edge_proxy" and dense_par10 is not None and baseline is not None:
            comparison.update(
                {
                    "comparison_status": "proxy_trend_compared",
                    "agreement": "partial_proxy_agreement",
                    "measured_lut_increase_vs_baseline": dense_par10["lut"] - baseline["lut"],
                    "measured_dsp_increase_vs_baseline": dense_par10["dsp"] - baseline["dsp"],
                    "measured_wns_ns": dense_par10["wns_ns"],
                    "measured_throughput_gain_vs_baseline": (
                        dense_par10["throughput_inferences_per_sec"] - baseline["throughput_inferences_per_sec"]
                    ),
                    "comparison_note": (
                        "The high-parallel proxy shows the expected resource growth and modest throughput gain, which supports "
                        "the model's narrow replicated-viability story, but timing is worse and the current RTL is still not a direct replicated MAC-array check."
                    ),
                }
            )
        comparisons.append(comparison)
    return comparisons


def render_comparison_summary(
    output_path: Path,
    manifest_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    status_counts: dict[str, int] = {}
    agreement_counts: dict[str, int] = {}
    for row in comparison_rows:
        status_counts[row["comparison_status"]] = status_counts.get(row["comparison_status"], 0) + 1
        agreement_counts[row["agreement"]] = agreement_counts.get(row["agreement"], 0) + 1

    lines = [
        "# Measured vs Modelled Comparison",
        "",
        "- This refresh loop is intentionally selective and uses the existing CNN Vivado studies as proxy evidence where direct MAC-array RTL is not yet available.",
        f"- Manifest candidates: `{len(manifest_rows)}`.",
        f"- Comparison status counts: {', '.join(f'{k}={v}' for k, v in sorted(status_counts.items()))}.",
        f"- Agreement counts: {', '.join(f'{k}={v}' for k, v in sorted(agreement_counts.items()))}.",
        "",
        "## Candidate Notes",
        "",
    ]
    comparison_by_id = {row["candidate_id"]: row for row in comparison_rows}
    for row in manifest_rows:
        comparison = comparison_by_id[row["candidate_id"]]
        lines.append(
            f"- `{row['candidate_id']}` -> expected `{row['expected_winner']}`; "
            f"mapping `{row['mapping_status']}`; comparison `{comparison['agreement']}`: {comparison['comparison_note']}"
        )
    output_path.write_text("\n".join(lines) + "\n")
