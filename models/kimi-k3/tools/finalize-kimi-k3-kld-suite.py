#!/usr/bin/env python3
"""Merge and validate chunked captures into one safetensors file per window."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


CHUNK_RE = re.compile(r"logits\.rows-(\d+)-(\d+)\.safetensors$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_chunks(path: Path, expected_rows: int) -> list[tuple[int, int, Path]]:
    chunks = []
    for item in path.glob("logits.rows-*.safetensors"):
        match = CHUNK_RE.match(item.name)
        if match:
            chunks.append((int(match.group(1)), int(match.group(2)), item))
    chunks.sort()
    row = 0
    for start, end, item in chunks:
        if start != row or end <= start:
            raise RuntimeError(f"Non-contiguous capture at {item}; expected row {row}")
        row = end
    if row != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {path}; got {row}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-vocab", type=int, default=163840)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--stop-window", type=int)
    args = parser.parse_args()

    suite = json.loads(
        (args.suite_dir / "suite-manifest.json").read_text(encoding="utf-8")
    )
    expected_rows = int(suite["context_length"]) - 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    stop = len(suite["windows"]) if args.stop_window is None else args.stop_window
    for window in suite["windows"][args.start_window : stop]:
        index = int(window["index"])
        chunk_dir = (
            args.capture_dir
            / args.run_name
            / f"window-{index:03d}-{window['domain']}-chunks"
        )
        chunks = ordered_chunks(chunk_dir, expected_rows)
        tensors = []
        for start, end, chunk_path in chunks:
            tensor = load_file(str(chunk_path), device="cpu")["logits"]
            if tuple(tensor.shape) != (end - start, args.expected_vocab):
                raise RuntimeError(f"Unexpected shape {tuple(tensor.shape)} in {chunk_path}")
            if tensor.dtype != torch.float32:
                raise RuntimeError(f"Expected float32 in {chunk_path}; got {tensor.dtype}")
            tensors.append(tensor)
        logits = torch.cat(tensors, dim=0).contiguous()
        output = args.output_dir / f"logits_{index:03d}.safetensors"
        save_file(
            {"logits": logits},
            str(output),
            metadata={
                "format": "kimi-k3-kld-reference-v1",
                "window_index": str(index),
                "domain": window["domain"],
                "context_length": str(suite["context_length"]),
                "vocab_size": str(args.expected_vocab),
                "dtype": "float32",
                "token_ids_json_sha256": window["token_ids_json_sha256"],
            },
        )
        del logits, tensors
        with safe_open(output, framework="pt", device="cpu") as handle:
            shape = list(handle.get_slice("logits").get_shape())
            dtype = str(handle.get_slice("logits").get_dtype())
        record = {
            "window_index": index,
            "domain": window["domain"],
            "file": output.name,
            "key": "logits",
            "shape": shape,
            "dtype": dtype,
            "size_bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "token_ids_json_sha256": window["token_ids_json_sha256"],
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    runtime = None
    if args.runtime_manifest:
        runtime = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    result = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "Kimi K3 full original MXFP4 KLD reference logits",
        "tensor_key": "logits",
        "dtype": "float32",
        "vocab_size": args.expected_vocab,
        "context_length": suite["context_length"],
        "window_count": len(records),
        "window_range": [args.start_window, stop],
        "total_scored_positions": len(records) * expected_rows,
        "total_size_bytes": sum(item["size_bytes"] for item in records),
        "suite_token_hash_sha256": suite["suite_token_hash_sha256"],
        "windows": records,
        "runtime": runtime,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
