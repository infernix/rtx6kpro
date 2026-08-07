# Kimi-K3 MXFP4 TP16/DCP16 1M on Heraldic Harbinger

Status: all three serving profiles measured on 2026-08-07 from the image
below, with no host source mounts. The image is self-contained; only the
HuggingFace cache is mounted.

Kimi-K3 is a 93-layer hybrid: roughly 24 MLA layers and 69 KDA linear-attention
layers, 92 MoE layers, 896 experts with top-16 sigmoid routing. The runtime
serves it at TP16 with decode context parallelism 16 across sixteen RTX PRO
6000 Blackwell cards on PCIe (no NVLink), with a physical one-million-token KV
cache.

## Current Image

```text
voipmonitor/vllm:kimi-k3-hh-runtime-vanilla-pr242-pr124-20260807
```

Image id `sha256:bba7eaf023e28a3bff5f46c107060df650c8e1f1545b3ba83fd0d6428eaf530c`.

## Component Revisions

| Component | Revision |
|---|---|
| Build recipe repo | `https://github.com/local-inference-lab/blackwell-kimi-k3-hh-dcp16-build.git` |
| Build Dockerfile | `Dockerfile.kimi-k3-hh-dcp16` |
| Build branch | `build/kimi-k3-hh-runtime-20260806-perf` |
| Base image | `voipmonitor/vllm:kimi-k3-hh-native-topk-base-20260804` |
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `pr/kimi-k3-dspark-dcp16-perf-20260806` (PR #242, base `dev/heraldic-harbinger`) |
| vLLM commit | `7a5634964efbb83496df07d8c1ff97591fc04a1a` |
| B12X repo | `https://github.com/local-inference-lab/b12x.git` |
| B12X branch | `pr/kimi-k3-dspark-dcp16-perf-20260806` (PR #124, base `master`) |
| B12X commit | `b18919369c934ad7324602f03a60017326081e46` |
| PyTorch | `2.12.0+cu132` |
| vLLM NCCL runtime | `/opt/libnccl-local-inference.so.2.30.4` |

The image composes from vanilla `dev/heraldic-harbinger` plus PR #242 and
vanilla `master` plus PR #124. Nothing else is required: the mandatory runtime
environment (`VLLM_NCCL_SO_PATH`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, the NCCL
knobs) ships as image `ENV`, and every performance-relevant knob is exported by
the launcher scripts rather than by an external env-file.

## Measured Performance

Same window, same machine, from the image above. Decode is the normalized
protocol: fixed 256-token prompt supplied as token ids, 1024 output tokens,
temperature 0, concurrency 1, two warmups and six timed runs, median. Sieve is
the coding watchdog prompt (`llm_cjk_watchdog.py`, 2000 max tokens),
generation-only rate.

| Profile | KV cache | decode tok/s | target cycles/s | accepted/cycle | Sieve tok/s | CJK |
|---|---|---|---|---|---|---|
| No speculation | 1,460,937 | 52.62 | — | — | 52.66 | 0 |
| DSpark K7 | 1,057,049 | **110.95** | 27.324 | 2.888 | 117.84 | 0 |
| DFlash K7 | 1,039,043 | 84.75 | 27.134 | 2.037 | **124.70** | 0 |

Read this as: both speculators drive the target loop at the same rate
(27.1-27.3 cycles/s, the graph-bound ceiling), so the difference between them is
purely acceptance on the given text. DSpark wins the fixed baseline prompt;
DFlash wins the coding prompt. Neither ordering is stable to better than a few
percent, because a free greedy decode's acceptance varies with content.

## Launching

All three profiles take the same container shape. Substitute the launcher
script; nothing else changes.

```bash
IMAGE=voipmonitor/vllm:kimi-k3-hh-runtime-vanilla-pr242-pr124-20260807

docker run -d --name kimi-k3 \
  --hostname aiserver --gpus all --network host --ipc host \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -w /opt/kimi-k3-hh/vllm --entrypoint bash "$IMAGE" \
  -lc 'exec ./<LAUNCHER> --port 8000'
```

### Without speculation

```text
./serve-kimi-k3-full-mxfp4-dcp16-1m-no-dspark.sh
```

Exact weights (routed experts in source MXFP4, dense in BF16), FP8 target KV,
1,460,937 tokens of cache. This is the reference profile for any correctness or
KLD work: it has no draft model and no aux hidden-state taps, so prompt-logprob
captures are trustworthy here and nowhere else.

### With DSpark (recommended default)

```text
./serve-kimi-k3-full-mxfp4-dspark7-dcp16-1m-kda-mxfp8.sh
```

Five-layer BF16 Inferact draft, K7, verify shape 1+7. The concurrency-1
serving controls belong in the container environment:

```text
-e DSPARK_BATCH_SIZE_SPECULATIVE_SCHEDULE='[[1,1,7]]'
-e DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=2
-e VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=2
-e DSPARK_SPS_CURVE=auto
-e VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1
```

The launcher already selects the measured profile: replicated `f_a`
(`KIMI_SHARD_F_A=0`), the fused sigmoid+top-16 routing kernel
(`VLLM_KIMI_CX_TOPK16=1`), full decode CUDA graphs at capture size 8, the
hierarchical PCIe all-reduce settings, and the B12X projection transports.

### With DFlash

```text
./serve-kimi-k3-full-mxfp4-dflash-dcp16-1m.sh
```

Six-layer qwen3-style GQA draft (`modal-labs/Kimi-K3-DFlash`), 4096 sliding
window, K7. Needs no extra environment. Knobs: `DFLASH_TOKENS` (default 7),
`DFLASH_DRAFT`, `DFLASH_DRAFT_MXFP8` (default 1), `KV_CACHE_MEMORY_BYTES`
(default 1200000000).

Three of its defaults are load-bearing and each cost a debugging session:

* **Full decode CUDA graphs**, capture size `1 + K`. `DFlashSpeculator`
  supports only full graphs and *silently* runs the draft eagerly otherwise.
  That single fallback is worth 5.3x: 5.05 versus 26.96 target cycles/s.
* **`draft_load_config: {"load_format": "auto"}`**. InstantTensor demands
  `chunk x concurrency x io_depth x world_size` free device bytes when it opens
  a checkpoint, which a ~90 GiB/rank target does not leave. The plain loader
  needs none of it, and the 4.9 GB draft loads in seconds either way.
* **`VLLM_USE_B12X_FP8_GEMM=1`** together with `KIMI_TARGET_MXFP8_PROFILE=kda_in_proj`.
  The profile frees the ~1.36 GiB/rank that makes room for the draft, but one
  K3 KDA projection is narrower than flashinfer's `mm_mxfp8` `n>=128 / k>=128`
  bound; the B12X kernel has no such bound.

## Runtime Notes

* `KV_CACHE_MEMORY_BYTES=1200000000` for DFlash is a deliberate backoff. At
  DSpark's 1325000000 the pool reports **more** capacity (1,151,050 tokens) but
  leaves only ~1.4 GiB free per GPU, and startup then stalls with all sixteen
  ranks spinning at 100% CPU inside `cuModuleLoadData`. That is a driver
  memory-pressure pathology, not a vLLM bug.
* `GLOO_SOCKET_IFNAME=lo` and `TP_SOCKET_IFNAME=lo` are set by the DFlash
  launcher. Draft graph capture runs long enough that gloo's TCP rendezvous
  otherwise picks a routable-but-unreachable global IPv6 address and aborts the
  barrier. Worth setting for any long-capture profile on this box.
* **Only K=7 works.** K9, K11 and K15 each fail differently (first-request CUDA
  illegal access, post-capture spin in the target KDA layer, capture OOM). The
  B12X DCP16 collective stack — a2a pools, capture plans, workspaces — was built
  and validated at M8, which is DSpark's 1+7 verify shape. Supporting other K is
  kernel work, not configuration.
* KV cost is dominated by the KDA rollback states, not by any draft. Measured
  per-request block split at 128K context (block = 6 x 442,368 B): twelve Mamba
  state groups take 96 blocks (63%), four DCP16-sharded MLA groups take 44
  (29%), the DFlash window group takes 12 (8%). The Mamba cost is constant in
  context length, so it dominates short-context configurations and disappears at
  1M. Enable the per-group diagnostic by reading the `KV group N:` lines that
  the engine now logs at startup.
* First start on a cold JIT cache pays several minutes of Triton compilation.
  Mount a persistent `/cache` (or `TRITON_CACHE_DIR`) for fast restarts.
