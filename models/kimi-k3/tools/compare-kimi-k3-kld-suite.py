#!/usr/bin/env python3
"""Compare a candidate Kimi K3 logit suite against the canonical reference."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def bootstrap_window_mean(
    window_means: list[float], *, samples: int, seed: int
) -> dict[str, float]:
    values = np.asarray(window_means, dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[choices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "standard_error": float(means.std(ddof=1)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_samples": samples,
        "cluster_unit": "window",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--row-block", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--stop-window", type=int)
    args = parser.parse_args()

    suite = json.loads(args.suite_manifest.read_text(encoding="utf-8"))
    reference_manifest = json.loads(
        (args.reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    candidate_manifest = json.loads(
        (args.candidate_dir / "manifest.json").read_text(encoding="utf-8")
    )
    expected_suite_hash = suite["suite_token_hash_sha256"]
    for label, manifest in (
        ("reference", reference_manifest),
        ("candidate", candidate_manifest),
    ):
        if manifest.get("suite_token_hash_sha256") != expected_suite_hash:
            raise RuntimeError(f"{label} suite token hash does not match corpus suite")
    reference_windows = {
        int(item["window_index"]): item for item in reference_manifest["windows"]
    }
    candidate_windows = {
        int(item["window_index"]): item for item in candidate_manifest["windows"]
    }
    all_kl: list[np.ndarray] = []
    all_js: list[np.ndarray] = []
    window_reports = []
    domain_window_means: dict[str, list[float]] = defaultdict(list)
    total_top1 = 0
    total_positions = 0
    stop = len(suite["windows"]) if args.stop_window is None else args.stop_window
    for window in suite["windows"][args.start_window : stop]:
        index = int(window["index"])
        expected_token_hash = window["token_ids_json_sha256"]
        for label, records in (
            ("reference", reference_windows),
            ("candidate", candidate_windows),
        ):
            if index not in records:
                raise RuntimeError(f"Window {index} missing from {label} manifest")
            if records[index].get("token_ids_json_sha256") != expected_token_hash:
                raise RuntimeError(f"Window {index} token hash differs in {label}")
        ref_path = args.reference_dir / f"logits_{index:03d}.safetensors"
        candidate_path = args.candidate_dir / f"logits_{index:03d}.safetensors"
        kl_parts: list[np.ndarray] = []
        js_parts: list[np.ndarray] = []
        top1 = 0
        positions = 0
        with safe_open(ref_path, framework="pt", device="cpu") as ref_handle, safe_open(
            candidate_path, framework="pt", device="cpu"
        ) as candidate_handle:
            ref_slice = ref_handle.get_slice("logits")
            candidate_slice = candidate_handle.get_slice("logits")
            if ref_slice.get_shape() != candidate_slice.get_shape():
                raise RuntimeError(f"Shape mismatch for window {index}")
            rows, _ = ref_slice.get_shape()
            for start in range(0, rows, args.row_block):
                end = min(start + args.row_block, rows)
                ref = ref_slice[start:end].float()
                candidate = candidate_slice[start:end].float()
                log_ref = torch.log_softmax(ref, dim=-1)
                log_candidate = torch.log_softmax(candidate, dim=-1)
                prob_ref = log_ref.exp()
                prob_candidate = log_candidate.exp()
                log_mid = torch.logaddexp(log_ref, log_candidate) - np.log(2.0)
                kl = (prob_ref * (log_ref - log_candidate)).sum(dim=-1)
                js = 0.5 * (
                    (prob_ref * (log_ref - log_mid)).sum(dim=-1)
                    + (prob_candidate * (log_candidate - log_mid)).sum(dim=-1)
                )
                kl_parts.append(kl.double().numpy())
                js_parts.append(js.double().numpy())
                top1 += int((ref.argmax(dim=-1) == candidate.argmax(dim=-1)).sum())
                positions += end - start
        kl_values = np.concatenate(kl_parts)
        js_values = np.concatenate(js_parts)
        report = {
            "window_index": index,
            "domain": window["domain"],
            "positions": positions,
            "kl_reference_to_candidate": stats(kl_values),
            "js": stats(js_values),
            "top1_agreement": top1 / positions,
        }
        window_reports.append(report)
        all_kl.append(kl_values)
        all_js.append(js_values)
        domain_window_means[window["domain"]].append(float(kl_values.mean()))
        total_top1 += top1
        total_positions += positions
        print(json.dumps(report, sort_keys=True), flush=True)

    combined_kl = np.concatenate(all_kl)
    combined_js = np.concatenate(all_js)
    window_means = [item["kl_reference_to_candidate"]["mean"] for item in window_reports]
    report = {
        "direction": "KL(reference || candidate)",
        "reference_dir": str(args.reference_dir),
        "candidate_dir": str(args.candidate_dir),
        "positions": total_positions,
        "windows": len(window_reports),
        "window_range": [args.start_window, stop],
        "kl_reference_to_candidate": stats(combined_kl),
        "kl_window_cluster_bootstrap": bootstrap_window_mean(
            window_means,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        "js": stats(combined_js),
        "top1_agreement": total_top1 / total_positions,
        "domains": {
            domain: bootstrap_window_mean(
                values,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            for domain, values in sorted(domain_window_means.items())
        },
        "per_window": window_reports,
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
