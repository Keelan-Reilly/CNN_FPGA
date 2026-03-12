#!/usr/bin/env python3
"""Build a simple model-vs-measurement latency dataset from an aggregate CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


MODEL_NAME = "sequential_calibrated_dense_v1"


def maybe_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key, value in list(normalized.items()):
        if key == "params":
            continue
        normalized[key] = maybe_number(value)
    params = normalized.get("params")
    if isinstance(params, str):
        try:
            normalized["params"] = json.loads(params)
        except json.JSONDecodeError:
            normalized["params"] = None
    return normalized


def load_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            rows = [normalize_row(dict(row)) for row in csv.DictReader(f)]
        experiment_id = rows[0]["experiment_id"] if rows else path.stem
        return str(experiment_id), rows

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        runs = [normalize_row(dict(run)) for run in payload.get("runs", [])]
        return str(payload.get("experiment_id", path.stem)), runs

    raise SystemExit(f"Unsupported input format: {path}")


def select_reference_run(
    rows: list[dict[str, Any]],
    reference_run_id: str | None,
    default_dense_out_par: int,
) -> dict[str, Any]:
    if reference_run_id:
        for row in rows:
            if row.get("run_id") == reference_run_id:
                return row
        raise SystemExit(f"Reference run_id not found: {reference_run_id}")

    for row in rows:
        params = row.get("params") or {}
        if (
            row.get("status") == "succeeded"
            and params.get("DENSE_OUT_PAR") == default_dense_out_par
            and row.get("stage_cycles_conv") is not None
        ):
            return row

    for row in rows:
        if row.get("status") == "succeeded" and row.get("stage_cycles_conv") is not None:
            return row

    raise SystemExit("No suitable reference run found")


def require_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int):
        raise SystemExit(f"Reference run missing integer field '{key}'")
    return value


def dense_cycles_model(num_classes: int, dense_out_par: int, in_dim: int, dense_lat: int) -> int:
    p_eff = max(1, min(dense_out_par, num_classes))
    batches = math.ceil(num_classes / p_eff)
    return batches * (in_dim * (dense_lat + 2) + 1) + 2


def with_model_fields(
    row: dict[str, Any],
    model_fixed_cycles: int,
    in_dim: int,
    num_classes: int,
    dense_lat: int,
    reference_run_id: str,
) -> dict[str, Any]:
    out = dict(row)
    params = row.get("params") or {}
    dense_out_par = params.get("DENSE_OUT_PAR")
    clock_hz = row.get("clock_hz")

    out["model_name"] = MODEL_NAME
    out["model_reference_run_id"] = reference_run_id
    out["model_fixed_cycles"] = model_fixed_cycles

    if not isinstance(dense_out_par, int):
        out["model_dense_cycles"] = None
        out["predicted_latency_cycles"] = None
        out["predicted_latency_time_ms"] = None
        out["predicted_throughput_inferences_per_sec"] = None
        out["latency_error_cycles"] = None
        out["latency_error_pct"] = None
        return out

    model_dense_cycles = dense_cycles_model(
        num_classes=num_classes,
        dense_out_par=dense_out_par,
        in_dim=in_dim,
        dense_lat=dense_lat,
    )
    predicted_latency_cycles = model_fixed_cycles + model_dense_cycles
    predicted_latency_time_ms = None
    predicted_throughput = None
    if isinstance(clock_hz, int) and predicted_latency_cycles > 0:
        predicted_latency_time_ms = 1000.0 * predicted_latency_cycles / clock_hz
        predicted_throughput = clock_hz / predicted_latency_cycles

    measured_latency_cycles = row.get("latency_cycles")
    latency_error_cycles = None
    latency_error_pct = None
    if isinstance(measured_latency_cycles, int) and measured_latency_cycles != 0:
        latency_error_cycles = measured_latency_cycles - predicted_latency_cycles
        latency_error_pct = 100.0 * latency_error_cycles / measured_latency_cycles

    out["model_dense_cycles"] = model_dense_cycles
    out["predicted_latency_cycles"] = predicted_latency_cycles
    out["predicted_latency_time_ms"] = predicted_latency_time_ms
    out["predicted_throughput_inferences_per_sec"] = predicted_throughput
    out["latency_error_cycles"] = latency_error_cycles
    out["latency_error_pct"] = latency_error_pct
    return out


def csv_fields(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "predicted_latency_cycles",
        "predicted_latency_time_ms",
        "predicted_throughput_inferences_per_sec",
        "latency_error_cycles",
        "latency_error_pct",
        "model_fixed_cycles",
        "model_dense_cycles",
        "model_name",
        "model_reference_run_id",
    ]
    existing: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                existing.append(key)
                seen.add(key)
    return existing + [field for field in preferred if field not in seen]


def write_outputs(
    input_path: Path,
    experiment_id: str,
    rows: list[dict[str, Any]],
    model_meta: dict[str, Any],
) -> tuple[Path, Path]:
    out_stem = input_path.with_name(f"{input_path.stem}_model")
    json_path = out_stem.with_suffix(".json")
    csv_path = out_stem.with_suffix(".csv")

    json_payload = {
        "experiment_id": f"{experiment_id}_model",
        "source_dataset": input_path.name,
        "model": model_meta,
        "runs": rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields(rows))
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            if isinstance(csv_row.get("params"), dict):
                csv_row["params"] = json.dumps(csv_row["params"], sort_keys=True)
            writer.writerow(csv_row)

    return json_path, csv_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Add analytical latency model fields to an aggregate dataset")
    ap.add_argument("--input", required=True, help="Aggregate CSV or JSON path")
    ap.add_argument("--reference-run-id", default=None, help="Explicit run_id used for fixed-term calibration")
    ap.add_argument("--reference-dense-out-par", type=int, default=1, help="Default reference DENSE_OUT_PAR value")
    ap.add_argument("--img-size", type=int, default=28, help="Model assumption: image size")
    ap.add_argument("--out-channels", type=int, default=8, help="Model assumption: conv output channels")
    ap.add_argument("--pool", type=int, default=2, help="Model assumption: pooling factor")
    ap.add_argument("--num-classes", type=int, default=10, help="Model assumption: dense output dimension")
    ap.add_argument("--dense-lat", type=int, default=2, help="Model assumption: dense input latency")
    args = ap.parse_args()

    input_path = Path(args.input)
    experiment_id, rows = load_rows(input_path)
    if not rows:
        raise SystemExit("Input dataset is empty")

    reference = select_reference_run(rows, args.reference_run_id, args.reference_dense_out_par)
    model_fixed_cycles = (
        require_int(reference, "stage_cycles_conv")
        + require_int(reference, "stage_cycles_relu")
        + require_int(reference, "stage_cycles_pool")
        + require_int(reference, "stage_cycles_argmax")
        + require_int(reference, "bubble_cycles")
    )
    in_dim = args.out_channels * (args.img_size // args.pool) * (args.img_size // args.pool)

    modeled_rows = [
        with_model_fields(
            row=row,
            model_fixed_cycles=model_fixed_cycles,
            in_dim=in_dim,
            num_classes=args.num_classes,
            dense_lat=args.dense_lat,
            reference_run_id=str(reference.get("run_id")),
        )
        for row in rows
    ]

    model_meta = {
        "model_name": MODEL_NAME,
        "source_dataset": input_path.name,
        "reference_run_id": reference.get("run_id"),
        "reference_dense_out_par": (reference.get("params") or {}).get("DENSE_OUT_PAR"),
        "model_fixed_cycles": model_fixed_cycles,
        "assumptions": {
            "sequential_pipeline": True,
            "no_stage_overlap": True,
            "img_size": args.img_size,
            "out_channels": args.out_channels,
            "pool": args.pool,
            "num_classes": args.num_classes,
            "dense_lat": args.dense_lat,
            "in_dim": in_dim,
            "fixed_term_definition": "stage_cycles_conv + stage_cycles_relu + stage_cycles_pool + stage_cycles_argmax + bubble_cycles",
            "dense_term_definition": "ceil(num_classes / clamp(DENSE_OUT_PAR,1,num_classes)) * (in_dim * (dense_lat + 2) + 1) + 2",
            "bubble_contains_tail_effects": True,
            "tx_wait_cycles_is_diagnostic_only": True,
        },
    }

    json_path, csv_path = write_outputs(input_path, experiment_id, modeled_rows, model_meta)
    print(f"Wrote model datasets: {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
