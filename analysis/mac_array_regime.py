#!/usr/bin/env python3
"""Bounded regime-map generation for the MAC-array framework."""

from __future__ import annotations

from typing import Any, Callable

try:
    from .mac_array_policy import evaluate_policy
    from .mac_array_types import ConstraintSpec
except ImportError:
    from mac_array_policy import evaluate_policy
    from mac_array_types import ConstraintSpec


def _group_workload_rows(workload_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in workload_rows:
        grouped.setdefault((row["grid"], row["workload"]), []).append(row)
    return grouped


def _rows_by_architecture(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["architecture"]: row for row in rows}


def _budget_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arch = _rows_by_architecture(rows)
    shared = by_arch["shared"]
    baseline = by_arch["baseline"]
    replicated = by_arch["replicated"]
    return [
        {
            "budget_class": "tight_shared_only",
            "dsp_budget": shared["dsp"],
            "lut_budget": shared["lut"],
            "budget_tightness": "tight",
        },
        {
            "budget_class": "baseline_fit",
            "dsp_budget": baseline["dsp"],
            "lut_budget": baseline["lut"],
            "budget_tightness": "moderate",
        },
        {
            "budget_class": "expanded_headroom",
            "dsp_budget": max(shared["dsp"], baseline["dsp"], replicated["dsp"]),
            "lut_budget": max(shared["lut"], baseline["lut"], replicated["lut"]),
            "budget_tightness": "relaxed",
        },
    ]


def _throughput_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["effective_throughput_ops_per_cycle"])
    low = ordered[0]["effective_throughput_ops_per_cycle"] * 0.98
    mid = ordered[1]["effective_throughput_ops_per_cycle"] * 0.99
    high = ordered[-1]["effective_throughput_ops_per_cycle"] * 0.995
    return [
        {"throughput_class": "efficiency_oriented", "min_throughput_ops_per_cycle": low},
        {"throughput_class": "balanced", "min_throughput_ops_per_cycle": mid},
        {"throughput_class": "stretch", "min_throughput_ops_per_cycle": high},
    ]


def build_regime_map(
    workload_rows: list[dict[str, Any]],
    adaptive_factory: Callable[[str, str, ConstraintSpec], dict[str, Any] | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped = _group_workload_rows(workload_rows)
    regime_rows: list[dict[str, Any]] = []

    for (grid, workload), rows in sorted(grouped.items()):
        sample = rows[0]
        budget_variants = _budget_variants(rows)
        throughput_variants = _throughput_variants(rows)
        for budget in budget_variants:
            for target in throughput_variants:
                constraint = ConstraintSpec(
                    name=f"{grid}_{workload}_{budget['budget_class']}_{target['throughput_class']}",
                    dsp_budget=int(budget["dsp_budget"]),
                    lut_budget=int(budget["lut_budget"]),
                    min_throughput_ops_per_cycle=float(target["min_throughput_ops_per_cycle"]),
                    target_grid=grid,
                )
                adaptive_row = adaptive_factory(grid, workload, constraint)
                recommendation, diagnostics = evaluate_policy(
                    fixed_rows=rows,
                    adaptive_row=adaptive_row,
                    constraint=constraint,
                    workload_class=sample["workload_class"],
                )
                runner_up = recommendation.get("runner_up", "")
                adaptive_diag = next((row for row in diagnostics if row["candidate"] == "adaptive_mode_switching"), None)
                regime_rows.append(
                    {
                        "grid": grid,
                        "workload": workload,
                        "workload_class": sample["workload_class"],
                        "phase_count": sample["phase_count"],
                        "dominant_utilization": sample["dominant_utilization"],
                        "utilization_variance": sample["utilization_variance"],
                        "burstiness": sample["burstiness"],
                        "budget_class": budget["budget_class"],
                        "budget_tightness": budget["budget_tightness"],
                        "dsp_budget": budget["dsp_budget"],
                        "lut_budget": budget["lut_budget"],
                        "throughput_class": target["throughput_class"],
                        "min_throughput_ops_per_cycle": target["min_throughput_ops_per_cycle"],
                        "winner": recommendation["recommendation"],
                        "winner_reason": recommendation["reason"],
                        "runner_up": runner_up,
                        "runner_up_reason": recommendation.get("runner_up_reason", ""),
                        "result_basis": "analytical_policy_over_evidence_backed_inputs",
                        "adaptive_candidate_present": adaptive_diag is not None,
                        "adaptive_candidate_feasible": adaptive_diag["feasible"] if adaptive_diag else False,
                        "adaptive_rejected_reasons": ",".join(adaptive_diag["rejected_reasons"]) if adaptive_diag else "",
                    }
                )

    summary_counts: dict[tuple[str, str], int] = {}
    adaptive_wins = 0
    for row in regime_rows:
        key = (row["grid"], row["winner"])
        summary_counts[key] = summary_counts.get(key, 0) + 1
        if row["winner"] == "adaptive_mode_switching":
            adaptive_wins += 1

    summary_rows = [
        {"grid": grid, "winner": winner, "count": count}
        for (grid, winner), count in sorted(summary_counts.items())
    ]
    meta = {
        "regime_points": len(regime_rows),
        "adaptive_win_points": adaptive_wins,
        "adaptive_has_win_region": adaptive_wins > 0,
    }
    return regime_rows, summary_rows, meta


def build_rejection_summary(regime_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = {}
    totals: dict[tuple[str, str], int] = {}
    for row in regime_rows:
        bucket = (row["grid"], row["workload"])
        totals[bucket] = totals.get(bucket, 0) + 1
        reasons = [item for item in str(row.get("adaptive_rejected_reasons", "")).split(",") if item]
        if not reasons:
            continue
        for reason in reasons:
            key = (bucket[0], bucket[1], reason)
            counts[key] = counts.get(key, 0) + 1
    summary = [
        {
            "grid": grid,
            "workload": workload,
            "rejection_reason": reason,
            "count": count,
            "share_of_regime_points": round(count / totals[(grid, workload)], 6),
        }
        for (grid, workload, reason), count in sorted(counts.items())
    ]
    return summary


def build_adaptive_rejection_surface(
    regime_rows: list[dict[str, Any]],
    rejection_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_points: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in regime_rows:
        grouped_points.setdefault((row["grid"], row["workload"]), []).append(row)

    reasons_by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rejection_summary:
        reasons_by_bucket.setdefault((row["grid"], row["workload"]), []).append(row)

    surface_rows: list[dict[str, Any]] = []
    for (grid, workload), rows in sorted(grouped_points.items()):
        adaptive_candidates = [row for row in rows if row.get("adaptive_candidate_present")]
        feasible_candidates = [row for row in adaptive_candidates if row.get("adaptive_candidate_feasible")]
        wins = [row for row in rows if row["winner"] == "adaptive_mode_switching"]
        dominant_rows = sorted(
            reasons_by_bucket.get((grid, workload), []),
            key=lambda item: (-item["count"], item["rejection_reason"]),
        )
        dominant_reason = dominant_rows[0]["rejection_reason"] if dominant_rows else "no_rejection_recorded"
        dominant_share = dominant_rows[0]["share_of_regime_points"] if dominant_rows else 0.0
        surface_rows.append(
            {
                "grid": grid,
                "workload": workload,
                "adaptive_candidate_points": len(adaptive_candidates),
                "adaptive_feasible_points": len(feasible_candidates),
                "adaptive_win_points": len(wins),
                "dominant_rejection_reason": dominant_reason,
                "dominant_rejection_share_of_regime_points": dominant_share,
                "adaptive_win_region_present": bool(wins),
            }
        )
    return surface_rows


def build_regime_insights(
    regime_rows: list[dict[str, Any]],
    regime_summary_rows: list[dict[str, Any]],
    rejection_surface_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    total_points = len(regime_rows)
    winner_counts: dict[str, int] = {}
    for row in regime_rows:
        winner_counts[row["winner"]] = winner_counts.get(row["winner"], 0) + 1
    for winner, count in sorted(winner_counts.items(), key=lambda item: (-item[1], item[0])):
        insights.append(
            {
                "insight_type": "winner_share",
                "subject": winner,
                "value": count,
                "share": round(count / total_points, 6) if total_points else 0.0,
                "note": f"{winner} wins {count} of {total_points} bounded regime points.",
            }
        )

    for row in sorted(regime_summary_rows, key=lambda item: (item["grid"], -item["count"], item["winner"])):
        insights.append(
            {
                "insight_type": "grid_winner_count",
                "subject": f"{row['grid']}:{row['winner']}",
                "value": row["count"],
                "share": None,
                "note": f"{row['winner']} wins {row['count']} regime points on {row['grid']}.",
            }
        )

    blocked = sorted(
        rejection_surface_rows,
        key=lambda item: (-item["dominant_rejection_share_of_regime_points"], item["grid"], item["workload"]),
    )
    for row in blocked[:6]:
        insights.append(
            {
                "insight_type": "adaptive_blocker",
                "subject": f"{row['grid']}:{row['workload']}",
                "value": row["dominant_rejection_reason"],
                "share": row["dominant_rejection_share_of_regime_points"],
                "note": (
                    f"Adaptive is mainly blocked on {row['grid']} / {row['workload']} by "
                    f"{row['dominant_rejection_reason']}."
                ),
            }
        )
    return insights
