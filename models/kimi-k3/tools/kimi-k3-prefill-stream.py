#!/usr/bin/env python3
"""Measure end-to-end prefill throughput through vLLM's OpenAI API."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def exact_prompt(seed_tokens: list[int], size: int) -> list[int]:
    if not seed_tokens:
        raise ValueError("Token seed is empty")
    return (seed_tokens * math.ceil(size / len(seed_tokens)))[:size]


def run_once(
    endpoint: str,
    model: str,
    prompt: list[int],
    timeout: float,
) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, int] | None = None
    output_parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error") is not None:
                    raise RuntimeError(f"Server stream error: {event['error']}")
                if event.get("usage") is not None:
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices and first_token_at is None:
                    first_token_at = time.perf_counter()
                output_parts.extend(choice.get("text", "") for choice in choices)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("The stream ended without a completion token")

    ttft = first_token_at - started
    prompt_tokens = int((usage or {}).get("prompt_tokens", len(prompt)))
    return {
        "requested_prompt_tokens": len(prompt),
        "usage_prompt_tokens": prompt_tokens,
        "usage_completion_tokens": (usage or {}).get("completion_tokens"),
        "ttft_seconds": ttft,
        "wall_seconds": finished - started,
        "effective_prefill_tokens_per_second": prompt_tokens / ttft,
        "output_preview": "".join(output_parts)[:80],
    }


def summarize(measurements: list[dict[str, object]]) -> dict[str, float]:
    ttfts = [float(row["ttft_seconds"]) for row in measurements]
    rates = [float(row["effective_prefill_tokens_per_second"]) for row in measurements]
    return {
        "median_ttft_seconds": statistics.median(ttfts),
        "mean_ttft_seconds": statistics.mean(ttfts),
        "median_effective_prefill_tokens_per_second": statistics.median(rates),
        "mean_effective_prefill_tokens_per_second": statistics.mean(rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    seed_tokens = json.loads(args.token_file.read_text(encoding="utf-8"))
    if not isinstance(seed_tokens, list) or not all(
        isinstance(token, int) for token in seed_tokens
    ):
        raise TypeError("Token file must contain a JSON array of integer token IDs")

    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "model": args.model,
        "method": "OpenAI /v1/completions, direct token IDs, max_tokens=1, streamed TTFT",
        "prefix_cache_requirement": "server-side prefix caching disabled",
        "warmups_per_size": args.warmups,
        "measured_runs_per_size": args.runs,
        "sizes": {},
    }

    for size in args.sizes:
        prompt = exact_prompt(seed_tokens, size)
        size_report: dict[str, object] = {"warmups": [], "measurements": []}
        print(json.dumps({"event": "size_start", "prompt_tokens": size}), flush=True)
        for index in range(args.warmups):
            row = run_once(args.url, args.model, prompt, args.timeout)
            row["run"] = index + 1
            size_report["warmups"].append(row)
            print(json.dumps({"phase": "warmup", **row}), flush=True)
        for index in range(args.runs):
            row = run_once(args.url, args.model, prompt, args.timeout)
            row["run"] = index + 1
            size_report["measurements"].append(row)
            print(json.dumps({"phase": "measurement", **row}), flush=True)
        size_report["summary"] = summarize(size_report["measurements"])
        report["sizes"][str(size)] = size_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "report_written", "path": str(args.output)}))


if __name__ == "__main__":
    main()
