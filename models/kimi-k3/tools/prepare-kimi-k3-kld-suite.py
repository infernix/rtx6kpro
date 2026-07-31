#!/usr/bin/env python3
"""Build the canonical 32x2048-token Kimi K3 KLD corpus suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer


@dataclass(frozen=True)
class SourceSpec:
    domain: str
    dataset: str
    config: str | None
    split: str
    windows: int


SOURCES = (
    SourceSpec("prose", "Salesforce/wikitext", "wikitext-2-raw-v1", "test", 16),
    SourceSpec("code", "openai/openai_humaneval", None, "test", 8),
    SourceSpec("instruction", "databricks/databricks-dolly-15k", None, "train", 8),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_source(spec: SourceSpec, rows: Any) -> tuple[str, str]:
    if spec.dataset == "Salesforce/wikitext":
        values = [row["text"] for row in rows if row.get("text", "").strip()]
        return "\n\n".join(values), "non-empty text rows joined by two newlines"
    if spec.dataset == "openai/openai_humaneval":
        values = []
        for row in rows:
            values.append(
                f"# {row['task_id']}\n{row['prompt']}{row['canonical_solution']}"
                f"\n\n# Tests\n{row['test']}"
            )
        return "\n\n\n".join(values), (
            "task_id comment, prompt, canonical_solution, and test joined in dataset order"
        )
    if spec.dataset == "databricks/databricks-dolly-15k":
        values = []
        for row in rows:
            parts = [
                f"### Category\n{row['category']}",
                f"### Instruction\n{row['instruction']}",
            ]
            if row.get("context", "").strip():
                parts.append(f"### Context\n{row['context']}")
            parts.append(f"### Response\n{row['response']}")
            values.append("\n\n".join(parts))
        return "\n\n\n".join(values), (
            "category, instruction, optional context, and response joined in dataset order"
        )
    raise ValueError(f"No source builder for {spec.dataset}")


def spaced_starts(total: int, length: int, count: int) -> list[int]:
    if total < count * length:
        raise RuntimeError(
            f"Need at least {count * length} tokens for {count} disjoint windows; got {total}"
        )
    if count == 1:
        return [(total - length) // 2]
    # Even coverage of the complete source. Since total >= count*length, these
    # starts are guaranteed not to overlap.
    starts = [math.floor(i * (total - length) / (count - 1)) for i in range(count)]
    if any(b - a < length for a, b in zip(starts, starts[1:])):
        raise RuntimeError(f"Generated overlapping starts: {starts}")
    return starts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/k3-serve/model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    token_dir = args.output_dir / "tokens"
    token_dir.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    api = HfApi()

    dataset_records: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    window_index = 0
    for spec in SOURCES:
        revision = api.dataset_info(spec.dataset).sha
        kwargs = {
            "path": spec.dataset,
            "split": spec.split,
            "revision": revision,
        }
        if spec.config is not None:
            kwargs["name"] = spec.config
        dataset = load_dataset(**kwargs)
        source_text, construction = build_source(spec, dataset)
        source_sha256 = sha256_bytes(source_text.encode("utf-8"))
        token_stream = tokenizer.encode(source_text, add_special_tokens=False)
        starts = spaced_starts(len(token_stream), args.context_length, spec.windows)
        dataset_records.append(
            {
                "domain": spec.domain,
                "dataset": spec.dataset,
                "config": spec.config,
                "split": spec.split,
                "revision": revision,
                "dataset_fingerprint": dataset._fingerprint,
                "rows": len(dataset),
                "construction": construction,
                "source_utf8_sha256": source_sha256,
                "source_tokens": len(token_stream),
                "window_starts": starts,
                "windows": spec.windows,
            }
        )
        for domain_index, start in enumerate(starts):
            token_ids = token_stream[start : start + args.context_length]
            compact = json.dumps(token_ids, separators=(",", ":"))
            filename = f"window-{window_index:03d}-{spec.domain}.json"
            (token_dir / filename).write_text(compact + "\n", encoding="utf-8")
            window_records.append(
                {
                    "index": window_index,
                    "domain": spec.domain,
                    "domain_index": domain_index,
                    "dataset": spec.dataset,
                    "dataset_revision": revision,
                    "source_token_start": start,
                    "source_token_end": start + args.context_length,
                    "num_tokens": len(token_ids),
                    "token_file": f"tokens/{filename}",
                    "token_ids_json_sha256": sha256_bytes(compact.encode("utf-8")),
                    "token_ids_first16": token_ids[:16],
                    "token_ids_last16": token_ids[-16:],
                }
            )
            window_index += 1

    suite_hash_input = "\n".join(
        record["token_ids_json_sha256"] for record in window_records
    ).encode("ascii")
    manifest = {
        "format_version": 1,
        "kind": "Kimi K3 KLD corpus suite",
        "context_length": args.context_length,
        "add_special_tokens": False,
        "selection": (
            "evenly spaced, non-overlapping windows over each domain token stream"
        ),
        "window_count": len(window_records),
        "scored_positions_per_window": args.context_length - 1,
        "total_scored_positions": len(window_records) * (args.context_length - 1),
        "domain_counts": {spec.domain: spec.windows for spec in SOURCES},
        "model_tokenizer": args.model,
        "tokenizer_class": tokenizer.__class__.__name__,
        "suite_token_hash_sha256": sha256_bytes(suite_hash_input),
        "datasets": dataset_records,
        "windows": window_records,
    }
    (args.output_dir / "suite-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
