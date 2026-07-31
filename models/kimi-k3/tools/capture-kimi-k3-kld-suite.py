#!/usr/bin/env python3
"""Capture raw prefill logits for every window in a Kimi K3 KLD suite."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


CHUNK_RE = re.compile(r"logits\.rows-(\d+)-(\d+)\.safetensors$")


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_chunks(path: Path, expected_rows: int) -> list[str]:
    found: list[tuple[int, int, str]] = []
    for item in path.glob("logits.rows-*.safetensors"):
        match = CHUNK_RE.match(item.name)
        if match:
            found.append((int(match.group(1)), int(match.group(2)), item.name))
    found.sort()
    row = 0
    for start, end, _ in found:
        if start != row or end <= start:
            raise RuntimeError(f"Non-contiguous chunks in {path}; expected row {row}")
        row = end
    if row != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {path}; got {row}")
    return [item[2] for item in found]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="Kimi-K3")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="reference")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--stop-window", type=int)
    args = parser.parse_args()

    manifest = json.loads(
        (args.suite_dir / "suite-manifest.json").read_text(encoding="utf-8")
    )
    expected_rows = int(manifest["context_length"]) - 1
    capture_root = args.capture_dir / args.run_name
    capture_root.mkdir(parents=True, exist_ok=True)
    record_path = args.suite_dir / f"capture-{args.run_name}.json"
    records: list[dict[str, Any]] = []
    if record_path.exists():
        records = json.loads(record_path.read_text(encoding="utf-8"))
    completed = {int(record["window_index"]) for record in records}

    windows = manifest["windows"]
    stop = len(windows) if args.stop_window is None else args.stop_window
    for window in windows[args.start_window : stop]:
        index = int(window["index"])
        destination = capture_root / f"window-{index:03d}-{window['domain']}-chunks"
        if index in completed:
            validate_chunks(destination, expected_rows)
            print(f"window {index:03d}: already captured", flush=True)
            continue
        if destination.exists():
            raise RuntimeError(f"Unrecorded destination already exists: {destination}")
        token_ids = json.loads(
            (args.suite_dir / window["token_file"]).read_text(encoding="utf-8")
        )
        before = {path.name for path in args.capture_dir.iterdir() if path.is_dir()}
        body = {
            "model": args.model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 1,
            "prompt_logprobs": 1,
        }
        started = time.monotonic()
        response = requests.post(args.url, json=body, timeout=args.timeout)
        elapsed = time.monotonic() - started
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        after = {path.name for path in args.capture_dir.iterdir() if path.is_dir()}
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError(f"Expected one new capture directory, got {created}")
        source = args.capture_dir / created[0]
        source.rename(destination)
        chunk_names = validate_chunks(destination, expected_rows)
        payload = response.json()
        record = {
            "window_index": index,
            "domain": window["domain"],
            "prompt_tokens": len(token_ids),
            "request_id": payload.get("id"),
            "elapsed_seconds": elapsed,
            "usage": payload.get("usage"),
            "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
            "chunk_dir": str(destination),
            "chunks": chunk_names,
        }
        records.append(record)
        records.sort(key=lambda item: int(item["window_index"]))
        write_records(record_path, records)
        completed.add(index)
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
