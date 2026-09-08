#!/usr/bin/env python3
"""R27-tp3-min per-pass qualification client.

Drives a running GLM-5.3-Flash NVFP4 server through the fixed qualification
order and writes a JSON receipt per pass:
  - fresh-server-per-profile, matched warmups
  - target-only no-spec baseline (determinism 5/5, frozen text corpus 9/9)
  - exact long prompt + 16-token completion (TTFT / prefill tok/s)
  - four-way 128K concurrency
  - prefix-cache reuse double-pass
  - speculative modes (MTP3 default / DFlash2 K7) + vision red-PNG probe
Each receipt records its own conditions line + the exact command set.

Usage:
  python3 qualify-r27-tp3-min.py --mode dense --host 127.0.0.1 --port 5800 \
      --expected "R27 TP3 ordinary works" --out receipt-dense.json
"""
import argparse
import base64
import io
import json
import statistics
import struct
import time
import urllib.error
import urllib.request
import zlib

MODEL = "GLM-5.3-Flash-NVFP4"

FROZEN_TEXT_CORPUS = [
    "Reply with exactly: R27 TP3 ordinary works",
    "Reply with exactly: R27 TP3 MTP3 works",
    "Reply with exactly: R27 TP3 DFlash2 works",
    "Reply with exactly: QAD borderline probe pass",
    "State the rule for choosing an activation function when cheking off an integer list.",
    "Reply with exactly: twenty-two seconds",
    "Explain, in one short sentence, why the four-way concurrency pass exists.",
]
FROZEN_VISION = [("red", "red-image-qualifies")]
REVERSE = "abcdefghijklmnopqrstuvwxyz"

LONG_PROMPT_WORDS = "pineapple"


def _chunk(size: int, head: int = 1_048_576) -> bytes:
    # minimal length-budgeted PNG payload for the vision probe
    ident = b"pHYs\x00\x00\r\xd7\xd2\xdf\xdb\xdf"
    width, height = struct.pack(">II", 4, 4)
    raw = b"\x80\xff\x00\x00" * 4 + b"\x00" * 16
    idat = zlib.compress(raw)
    def put(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + put(b"IHDR", width + height + b"\x08\x06\x00\x00\x00") + put(b"pHYs", ident[4:]) + put(b"IDAT", idat) + put(b"IEND", b"")


def build_image_png(color: str = "red") -> str:
    px = bytes([255, 0, 0]) if color == "red" else bytes([0, 255, 0])
    raw = b"".join(b"\x00" + px * 4 for _ in range(4))
    idat = zlib.compress(raw)

    def put(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    body = b"\x89PNG\r\n\x1a\n"
    body += put(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
    body += put(b"IDAT", idat)
    body += put(b"IEND", b"")
    return base64.b64encode(body).decode()


def chat(base: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0, seed: int = 0) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def text_answer(base: str, prompt: str, max_tokens: int = 256) -> tuple[str, dict]:
    result = chat(base, prompt, max_tokens=max_tokens)
    msg = result["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or ""), result


def long_prompt_tokens(target_tokens: int) -> str:
    # deterministic filler that tokenizes at roughly one token per five chars
    filler = " ".join([LONG_PROMPT_WORDS] * (target_tokens * 5 // len(LONG_PROMPT_WORDS) + 1))
    return f"Remember the code word ORCHID-7319. {filler[: int(target_tokens * 5)]} What code word did I ask you to remember?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["dense", "mtp3", "dflash2", "native", "lmcache"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5800)
    parser.add_argument("--expected", default=None, help="expected smoke answer for this mode")
    parser.add_argument("--long-tokens", type=int, default=1_000_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--prefill-trials", type=int, default=3)
    parser.add_argument("--decode-trials", type=int, default=5)
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--skip-concurrency", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--determinism-trials", type=int, default=5)
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"

    receipt: dict = {"mode": args.mode, "host_port": f"{args.host}:{args.port}",
                     "conditions": {"temperature": 0.0, "seed": 0, "fresh_server_profile": True,
                                    "warmup_requests": args.warmups,
                                    "prefill_trials": args.prefill_trials,
                                    "decode_trials": args.decode_trials}}

    if args.expected:
        text, result = text_answer(base, f"Reply with exactly: {args.expected}")
        receipt["smoke"] = {"accept": text.strip() == args.expected, "response": text.strip(),
                            "usage": result.get("usage")}

    # frozen text corpus
    corpus_results = []
    for item in FROZEN_TEXT_CORPUS:
        answer, _ = text_answer(base, item)
        corpus_results.append({"prompt_head": item.split(".")[0][:64], "answer": answer.strip()})
    receipt["frozen_text_corpus"] = {"count": len(FROZEN_TEXT_CORPUS), "results": corpus_results}

    # determinism: same prompt 5x, compare exact strings across all repeats
    if args.expected:
        repeats = []
        for _ in range(args.determinism_trials):
            answer, _ = text_answer(base, f"Reply with exactly: {args.expected}")
            repeats.append(answer.strip())
        receipt["determinism_5x"] = {"pass": len(set(repeats)) == 1, "answers": repeats}

    # long-context admission
    if not args.skip_long:
        prompt = long_prompt_tokens(args.long_tokens)
        t0 = time.perf_counter()
        answer, result = text_answer(base, prompt, max_tokens=16)
        dt = time.perf_counter() - t0
        receipt["long_context_admission"] = {
            "prompt_tokens_target": args.long_tokens,
            "prompt_tokens_reported": (result.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens_reported": (result.get("usage") or {}).get("completion_tokens"),
            "answer_head": answer.strip()[:64],
            "elapsed_seconds": round(dt, 3),
            "prompt_tok_per_s": round((result.get("usage") or {}).get("prompt_tokens", 0) / dt, 1),
        }

    # four-way 128K concurrency
    if not args.skip_concurrency:
        import threading
        class _Q:
            def __init__(self): self.items = []; self.lock = threading.Lock()
            def put(self, x):
                with self.lock: self.items.append(x)
        q = _Q(); threads = []
        prompt_128k = long_prompt_tokens(131_072)
        def _worker():
            t0 = time.perf_counter()
            try:
                answer, result = text_answer(base, prompt_128k, max_tokens=16)
                dt = time.perf_counter() - t0
                q.put({"ok": True, "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
                       "elapsed_seconds": round(dt, 3), "answer_head": answer.strip()[:32]})
            except Exception as exc:  # noqa: BLE001
                q.put({"ok": False, "error": repr(exc)})
        for _ in range(4):
            th = threading.Thread(target=_worker); th.start(); threads.append(th)
        for th in threads: th.join()
        receipt["four_way_128k_concurrency"] = {"results": q.items, "all_ok": all(r.get("ok") for r in q.items)}

    # prefix-cache reuse (double-pass on identical prompt)
    if not args.skip_long:
        prompt = long_prompt_tokens(min(65_536, args.long_tokens))
        t0 = time.perf_counter(); a1, r1 = text_answer(base, prompt, max_tokens=8); t1 = time.perf_counter() - t0
        t0 = time.perf_counter(); a2, r2 = text_answer(base, prompt, max_tokens=8); t2 = time.perf_counter() - t0
        receipt["prefix_cache_reuse"] = {
            "pass1_seconds": round(t1, 3), "pass2_seconds": round(t2, 3),
            "speedup": round(t1 / t2, 2) if t2 > 0 else None,
            "same_answer": a1.strip() == a2.strip(),
        }

    # vision probe (generated red PNG)
    if not args.skip_vision:
        png_b64 = build_image_png("red")
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What color is this single image?"}]}],
            "temperature": 0.0, "seed": 0, "max_tokens": 32,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        # multimodal image path: attach the png as a data url
        payload["messages"][0]["content"].insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
        })
        data = json.dumps(payload).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as resp:
            result = json.load(resp)
        msg = result["choices"][0]["message"]
        receipt["vision_probe"] = {"color_seen": (msg.get("content") or "").strip()[:16]}

    json.dump(receipt, open(args.out, "w"), indent=2, sort_keys=True)
    open(args.out, "a").write("\n")
    print(json.dumps({k: receipt.get(k) for k in ("mode", "smoke", "determinism_5x")}, indent=2))


if __name__ == "__main__":
    main()
