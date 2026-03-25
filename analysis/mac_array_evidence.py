#!/usr/bin/env python3
"""Evidence loading, validation, and provenance helpers for framework v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .fpga_results import REPO_ROOT
    from .mac_array_types import ArchitectureSpec
except ImportError:
    from fpga_results import REPO_ROOT
    from mac_array_types import ArchitectureSpec


STATIC_FIELDS = (
    "dsp",
    "lut",
    "wns_estimate_ns",
    "latency_penalty_fraction",
    "fixed_phase_overhead_cycles",
    "implementation_status",
    "note",
)


@dataclass(frozen=True)
class FieldProvenance:
    field: str
    value_kind: str
    source_id: str
    source_path: str
    source_desc: str
    derivation: str


@dataclass(frozen=True)
class EvidenceBundle:
    source_path: Path
    architectures: dict[str, ArchitectureSpec]
    architecture_meta: dict[str, dict[str, Any]]
    overrides: list[dict[str, Any]]
    switching: dict[str, Any]
    registry_rows: list[dict[str, Any]]


def _require_keys(payload: dict[str, Any], keys: list[str], ctx: str) -> None:
    for key in keys:
        if key not in payload:
            raise ValueError(f"Missing key '{key}' in {ctx}")


def _normalize_override(item: dict[str, Any], source_path: Path) -> dict[str, Any]:
    _require_keys(
        item,
        ["record_id", "grid", "architecture", "field", "value", "value_kind", "source_desc"],
        "override",
    )
    out = dict(item)
    out.setdefault("source_path", str(source_path.relative_to(REPO_ROOT)))
    out.setdefault("derivation", "explicit override from evidence registry")
    return out


def load_evidence(path: Path) -> EvidenceBundle:
    payload = json.loads(path.read_text())
    _require_keys(payload, ["architectures", "field_overrides", "switching"], "evidence bundle")

    arch_specs: dict[str, ArchitectureSpec] = {}
    arch_meta: dict[str, dict[str, Any]] = {}
    registry_rows: list[dict[str, Any]] = []

    for arch_name, arch_payload in payload["architectures"].items():
        _require_keys(
            arch_payload,
            ["description", "models"],
            f"architecture {arch_name}",
        )
        models = arch_payload["models"]
        required_models = ["dsp", "lut", "wns_estimate_ns", "execution"]
        for key in required_models:
            if key not in models:
                raise ValueError(f"Missing model '{key}' in architecture '{arch_name}'")

        dsp_model = models["dsp"]
        lut_model = models["lut"]
        wns_model = models["wns_estimate_ns"]
        exec_model = models["execution"]

        arch_specs[arch_name] = ArchitectureSpec(
            name=arch_name,
            capacity_scale=float(exec_model["parameters"]["capacity_scale"]),
            dsp_scale=float(dsp_model["parameters"]["dsp_scale"]),
            lut_per_mac=float(lut_model["parameters"]["lut_per_mac"]),
            lut_fixed=float(lut_model["parameters"]["lut_fixed"]),
            latency_penalty_fraction=float(exec_model["parameters"]["latency_penalty_fraction"]),
            fixed_phase_overhead_cycles=int(exec_model["parameters"]["fixed_phase_overhead_cycles"]),
            wns_bias_ns=float(wns_model["parameters"]["wns_bias_ns"]),
            wns_per_effective_mac=float(wns_model["parameters"]["wns_per_effective_mac"]),
            default_implementation_status=str(exec_model["parameters"]["default_implementation_status"]),
        )
        arch_meta[arch_name] = {
            "description": arch_payload["description"],
            "models": models,
            "variant_id": arch_payload.get("variant_id", arch_name),
            "variant_kind": arch_payload.get("variant_kind", "modelled_architecture_variant"),
            "scope_note": arch_payload.get("scope_note", arch_payload["description"]),
        }
        registry_rows.append(
            {
                "record_id": f"variant_{arch_meta[arch_name]['variant_id']}",
                "architecture": arch_name,
                "grid": "*",
                "field": "variant_id",
                "value": arch_meta[arch_name]["variant_id"],
                "value_kind": arch_meta[arch_name]["variant_kind"],
                "source_path": str(path.relative_to(REPO_ROOT)),
                "source_desc": arch_meta[arch_name]["scope_note"],
                "derivation": "architecture variant metadata",
            }
        )

        for field_name, field_model in models.items():
            registry_rows.append(
                {
                    "record_id": field_model["record_id"],
                    "architecture": arch_name,
                    "grid": "*",
                    "field": field_name,
                    "value": json.dumps(field_model["parameters"], sort_keys=True),
                    "value_kind": field_model["value_kind"],
                    "source_path": str(path.relative_to(REPO_ROOT)),
                    "source_desc": field_model["source_desc"],
                    "derivation": field_model["formula"],
                }
            )

    overrides = [_normalize_override(item, path) for item in payload["field_overrides"]]
    registry_rows.extend(overrides)

    switching = dict(payload["switching"])
    _require_keys(switching, ["default_components", "pair_overrides"], "switching")
    for item in switching["default_components"]:
        _require_keys(item, ["name", "cycles", "value_kind", "source_desc"], "default switch component")
    for item in switching["pair_overrides"]:
        _require_keys(item, ["record_id", "from_mode", "to_mode", "components"], "pair override")
        for component in item["components"]:
            _require_keys(component, ["name", "cycles", "value_kind", "source_desc"], "pair component")
        registry_rows.append(
            {
                "record_id": item["record_id"],
                "architecture": f"{item['from_mode']}->{item['to_mode']}",
                "grid": "*",
                "field": "switch_cycles",
                "value": sum(int(comp["cycles"]) for comp in item["components"]),
                "value_kind": "analytical_component_sum",
                "source_path": str(path.relative_to(REPO_ROOT)),
                "source_desc": item.get("source_desc", "Switch-cost pair override"),
                "derivation": "sum(component cycles)",
            }
        )

    return EvidenceBundle(
        source_path=path,
        architectures=arch_specs,
        architecture_meta=arch_meta,
        overrides=overrides,
        switching=switching,
        registry_rows=registry_rows,
    )


def static_override_record(
    bundle: EvidenceBundle,
    architecture: str,
    grid: str,
    field: str,
) -> dict[str, Any] | None:
    for override in bundle.overrides:
        if (
            override["architecture"] == architecture
            and override["grid"] == grid
            and override["field"] == field
        ):
            return override
    return None


def static_field_provenance(
    bundle: EvidenceBundle,
    architecture: str,
    grid: str,
    field: str,
) -> FieldProvenance:
    override = static_override_record(bundle, architecture, grid, field)
    if override is not None:
        return FieldProvenance(
            field=field,
            value_kind=str(override["value_kind"]),
            source_id=str(override["record_id"]),
            source_path=str(override["source_path"]),
            source_desc=str(override["source_desc"]),
            derivation=str(override.get("derivation", "explicit override")),
        )

    models = bundle.architecture_meta[architecture]["models"]
    if field in ("dsp", "lut", "wns_estimate_ns"):
        model = models[field]
    elif field in ("latency_penalty_fraction", "fixed_phase_overhead_cycles", "implementation_status"):
        model = models["execution"]
    elif field == "note":
        scope_note = bundle.architecture_meta[architecture].get("scope_note", "No explicit note")
        return FieldProvenance(
            field=field,
            value_kind="modelled_architecture_variant",
            source_id=f"variant_{bundle.architecture_meta[architecture].get('variant_id', architecture)}",
            source_path=str(bundle.source_path.relative_to(REPO_ROOT)),
            source_desc=str(scope_note),
            derivation="architecture variant metadata",
        )
    else:
        raise KeyError(f"Unsupported static field '{field}'")

    return FieldProvenance(
        field=field,
        value_kind=str(model["value_kind"]),
        source_id=str(model["record_id"]),
        source_path=str(bundle.source_path.relative_to(REPO_ROOT)),
        source_desc=str(model["source_desc"]),
        derivation=str(model["formula"]),
    )


def static_override_value(
    bundle: EvidenceBundle,
    architecture: str,
    grid: str,
    field: str,
) -> Any | None:
    override = static_override_record(bundle, architecture, grid, field)
    if override is None:
        return None
    return override["value"]


def switching_pair_record(bundle: EvidenceBundle, from_mode: str, to_mode: str) -> dict[str, Any]:
    for item in bundle.switching["pair_overrides"]:
        if item["from_mode"] == from_mode and item["to_mode"] == to_mode:
            total = sum(int(component["cycles"]) for component in item["components"])
            return {
                "record_id": item["record_id"],
                "from_mode": from_mode,
                "to_mode": to_mode,
                "switch_cycles": total,
                "components": item["components"],
                "value_kind": "analytical_component_sum",
                "source_path": str(bundle.source_path.relative_to(REPO_ROOT)),
                "source_desc": item.get("source_desc", "mode-pair switching cost"),
                "derivation": "sum(component cycles)",
            }

    total = sum(int(component["cycles"]) for component in bundle.switching["default_components"])
    return {
        "record_id": f"default_switch_{from_mode}_to_{to_mode}",
        "from_mode": from_mode,
        "to_mode": to_mode,
        "switch_cycles": total,
        "components": bundle.switching["default_components"],
        "value_kind": "analytical_component_sum",
        "source_path": str(bundle.source_path.relative_to(REPO_ROOT)),
        "source_desc": "default switching cost",
        "derivation": "sum(default component cycles)",
    }


def provenance_summary_rows(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in bundle.registry_rows:
        kind = str(row["value_kind"])
        counts[kind] = counts.get(kind, 0) + 1
    return [
        {
            "value_kind": kind,
            "count": count,
            "source_path": str(bundle.source_path.relative_to(REPO_ROOT)),
        }
        for kind, count in sorted(counts.items())
    ]
