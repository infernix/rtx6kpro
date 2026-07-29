#!/usr/bin/env python3
"""Measure Kimi decode from first to last streamed output token.

Unlike completion_tokens/request_wall_time, this excludes tokenization, queueing,
prefill, and TTFT.  For N output tokens the measured interval contains N-1
decode gaps, hence (N-1)/(last_token_time-first_token_time).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


DEFAULT_PROMPT = (
    "Give a rigorous, self-contained proof that there are infinitely many prime "
    "numbers. Explain every logical step, include the contradiction argument, "
    "and then discuss why the same proof does not imply that every number of the "
    "form p1*p2*...*pn+1 is prime."
)


def run(url: str, model: str, prompt: str, max_tokens: int) -> dict[str, float | int]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    token_times: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw_line in response:
            if not raw_line.startswith(b"data: "):
                continue
            data = raw_line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage")
            if usage:
                completion_tokens = int(usage.get("completion_tokens", 0))
                prompt_tokens = int(usage.get("prompt_tokens", 0))
            choices = event.get("choices") or []
            if not choices or choices[0].get("finish_reason") is not None:
                continue
            delta = choices[0].get("delta") or {}
            # Exclude the initial role-only event. vLLM emits one text delta per
            # sampled token for this tokenizer, including reasoning_content.
            if any(
                isinstance(delta.get(name), str) and delta[name] != ""
                for name in ("content", "reasoning_content")
            ):
                token_times.append(time.perf_counter())
    ended = time.perf_counter()
    if completion_tokens < 2 or len(token_times) < 2:
        raise RuntimeError(
            f"insufficient stream data: completion_tokens={completion_tokens}, "
            f"timed_events={len(token_times)}"
        )
    decode_s = token_times[-1] - token_times[0]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "timed_events": len(token_times),
        "ttft_s": token_times[0] - started,
        "decode_s": decode_s,
        "decode_tok_s": (completion_tokens - 1) / decode_s,
        "wall_s": ended - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5670")
    parser.add_argument("--model", default="Kimi-K3-MXFP4-NF3-4p05")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    for index in range(args.warmups):
        result = run(args.url, args.model, args.prompt, args.max_tokens)
        print(json.dumps({"run": f"warmup-{index + 1}", **result}), flush=True)
    measured = []
    for index in range(args.runs):
        result = run(args.url, args.model, args.prompt, args.max_tokens)
        measured.append(float(result["decode_tok_s"]))
        print(json.dumps({"run": f"measure-{index + 1}", **result}), flush=True)
    print(
        json.dumps(
            {
                "decode_tok_s_mean": statistics.mean(measured),
                "decode_tok_s_median": statistics.median(measured),
                "formula": "(completion_tokens - 1) / (last_token_time - first_token_time)",
            }
        )
    )


if __name__ == "__main__":
    main()
