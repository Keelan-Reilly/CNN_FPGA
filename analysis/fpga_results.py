#!/usr/bin/env python3
"""Utilities for loading FPGA aggregate datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGG_DIR = REPO_ROOT / "results" / "fpga" / "aggregates"


def _parse_number(value: Any) -> Any:
    if isinstance(value, (int, float)) or value is None:
        return value
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            return value
    return value


def _parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["params"] = _parse_params(row.get("params", {}))

    numeric_fields = [
        "returncode",
        "clock_period_ns",
        "duration_sec",
        "lut",
        "ff",
        "dsp",
        "bram",
        "wns_ns",
        "fmax_mhz_est",
        "latency_cycles",
        "latency_time_ms",
        "throughput_inferences_per_sec",
        "stage_cycles_conv",
        "stage_cycles_relu",
        "stage_cycles_pool",
        "stage_cycles_dense",
        "stage_cycles_argmax",
        "bubble_cycles",
        "busy_cycles",
        "tx_wait_cycles",
        "clock_hz",
        "predicted_latency_cycles",
        "predicted_latency_time_ms",
        "predicted_throughput_inferences_per_sec",
        "latency_error_cycles",
        "latency_error_pct",
        "model_fixed_cycles",
        "model_dense_cycles",
    ]
    for key in numeric_fields:
        normalized[key] = _parse_number(row.get(key))

    return normalized


def load_aggregate(aggregate_path: Path) -> tuple[str, list[dict[str, Any]]]:
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Aggregate file not found: {aggregate_path}")

    if aggregate_path.suffix.lower() == ".json":
        payload = json.loads(aggregate_path.read_text())
        experiment_id = str(payload.get("experiment_id", aggregate_path.stem))
        rows = [normalize_row(r) for r in payload.get("runs", [])]
        return experiment_id, rows

    if aggregate_path.suffix.lower() == ".csv":
        with aggregate_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = [normalize_row(r) for r in reader]
        experiment_id = aggregate_path.stem
        return experiment_id, rows

    raise ValueError("Aggregate path must be .json or .csv")


def resolve_aggregate_path(experiment_id: str | None, aggregate: str | None) -> Path:
    if aggregate:
        return Path(aggregate).resolve()
    if not experiment_id:
        raise ValueError("Provide --experiment-id or --aggregate")

    json_path = DEFAULT_AGG_DIR / f"{experiment_id}.json"
    csv_path = DEFAULT_AGG_DIR / f"{experiment_id}.csv"

    if json_path.exists():
        return json_path
    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        f"No aggregate dataset found for '{experiment_id}' in {DEFAULT_AGG_DIR}"
    )
