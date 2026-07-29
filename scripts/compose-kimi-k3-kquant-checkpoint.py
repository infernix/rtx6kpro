#!/usr/bin/env python3
"""Compose a zero-copy HF view from Kimi K3 base + kquant expert overlay."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Any


EXPERT_MARKER = ".block_sparse_moe.experts."


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_len))


def _base_nonexpert_bytes(base: Path, index: dict[str, Any]) -> int:
    weight_map = index["weight_map"]
    total = 0
    seen: set[str] = set()
    for filename in sorted(set(weight_map.values())):
        for name, info in _header(base / filename).items():
            if name == "__metadata__":
                continue
            if weight_map.get(name) != filename:
                raise ValueError(f"base index mismatch for {name}")
            seen.add(name)
            if EXPERT_MARKER not in name:
                start, end = info["data_offsets"]
                total += int(end) - int(start)
    if seen != set(weight_map):
        missing = set(weight_map) - seen
        raise ValueError(f"base index has {len(missing)} missing tensors")
    return total


def _symlink(source: Path, destination: Path) -> None:
    os.symlink(str(source.absolute()), destination)


def compose(base: Path, overlay: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")

    allocation = _read_json(overlay / "allocation.json")
    overlay_index = _read_json(overlay / "model-kquant.index.json")
    base_index = _read_json(base / "model.safetensors.index.json")
    config = _read_json(base / "config.json")

    assignment = allocation["assignment"]
    layers = allocation["layers_solved"]
    formats = allocation["formats"]
    if formats != ["keep_mxfp4", "nf3"]:
        raise ValueError(f"unsupported formats: {formats}")
    if len(assignment) != len(layers):
        raise ValueError("allocation layer/assignment length mismatch")
    hybrid_bit_map = {
        str(layer): [4 if int(choice) == 0 else 3 for choice in choices]
        for layer, choices in zip(layers, assignment, strict=True)
    }

    quantization_config = {
        "quant_method": "modelopt",
        "quant_algo": "W4A16_NVFP4",
        "group_size": 32,
        "ignore": [],
        "hybrid_bit_map": hybrid_bit_map,
        "kept_format": "mxfp4_e8m0k32",
    }
    config["quantization_config"] = quantization_config
    config["text_config"]["quantization_config"] = quantization_config

    combined_map = {
        name: filename
        for name, filename in base_index["weight_map"].items()
        if EXPERT_MARKER not in name
    }
    collisions = set(combined_map) & set(overlay_index["weight_map"])
    if collisions:
        raise ValueError(f"overlay collides with {len(collisions)} nonexpert tensors")
    combined_map.update(overlay_index["weight_map"])

    base_nonexpert_bytes = _base_nonexpert_bytes(base, base_index)
    overlay_bytes = int(overlay_index["metadata"]["total_size"])
    combined_index = {
        "metadata": {
            "total_size": base_nonexpert_bytes + overlay_bytes,
            "base_nonexpert_size": base_nonexpert_bytes,
            "kquant_expert_size": overlay_bytes,
        },
        "weight_map": combined_map,
    }

    destination.mkdir(parents=True)
    for source in sorted(base.iterdir()):
        if source.name in {"config.json", "model.safetensors.index.json"}:
            continue
        _symlink(source, destination / source.name)
    for source in sorted(overlay.glob("model-kquant-*.safetensors")):
        _symlink(source, destination / source.name)

    _write_json(destination / "config.json", config)
    _write_json(destination / "model.safetensors.index.json", combined_index)
    _write_json(
        destination / "kquant-composition.json",
        {
            "schema_version": 1,
            "base": str(base.absolute()),
            "overlay": str(overlay.absolute()),
            "model": allocation["model"],
            "revision": allocation["revision"],
            "achieved_bpw": allocation["achieved"]["bpw"],
            "keep_mxfp4": allocation["achieved"]["counts"]["keep_mxfp4"],
            "nf3": allocation["achieved"]["counts"]["nf3"],
            "base_nonexpert_bytes": base_nonexpert_bytes,
            "overlay_expert_bytes": overlay_bytes,
        },
    )
    print(
        json.dumps(
            {
                "destination": str(destination),
                "tensors": len(combined_map),
                "bytes": base_nonexpert_bytes + overlay_bytes,
                "hybrid_layers": len(hybrid_bit_map),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    compose(args.base, args.overlay, args.destination)


if __name__ == "__main__":
    main()
