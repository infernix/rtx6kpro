# Kimi K3 full MXFP4 TP16/DCP8 with physical 1M KV and 2048-token prefill chunks

Status date: 2026-08-03. This is the validated high-prefill-throughput
successor to the
[HH dense-MLA 1M profile](hh-dense-mla-full-mxfp4-dcp8-2026-08-03.md).
It keeps the original `moonshotai/Kimi-K3` MXFP4 checkpoint, TP16/DCP8,
SparkInfer dense-MLA decode, InstantTensor loading, and all 1,054,602 physical
KV-cache tokens while raising `max-num-batched-tokens` from 256 to 2048.

This profile is not NF3, EXL, EP, PP, or a padded model. The unchanged
`serve-kimi-k3-full-mxfp4-dcp8-1m.sh` launcher is retained as the exact
chunk-256 rollback point.

## Immutable artifacts

| Component | Immutable reference |
|---|---|
| vLLM branch | [`codex/kimi-k3-hh-dense-mla-dcp8-20260803`](https://github.com/local-inference-lab/vllm/tree/codex/kimi-k3-hh-dense-mla-dcp8-20260803) |
| Validated vLLM commit | `b3b8a189e4010343b2d6b83451ed3c41d8b10ff9` |
| FlashKDA allocation predecessor | `721e515069f8dd5b3162c1760cd78ac0f6ec3531` |
| HH base before these two commits | `6f1bcaa05ec603aba1e4b926c71aa8d4dcd8f05d` |
| Earlier vLLM review | [`local-inference-lab/vllm#232`](https://github.com/local-inference-lab/vllm/pull/232), merged at `6f1bcaa05`; the two successor commits are pinned on the branch above |
| SparkInfer branch | [`codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803`](https://github.com/local-inference-lab/sparkinfer/tree/codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803) |
| SparkInfer commit | `f39c6bf26be9d92b65d1f031819289c8c1f084a1` |
| SparkInfer review | [`local-inference-lab/sparkinfer#116`](https://github.com/local-inference-lab/sparkinfer/pull/116), merged |
| Published runtime image | `voipmonitor/vllm:kimi-k3-hh-dense-mla-dcp8-it-20260803` |
| Published runtime digest | `sha256:499c405bb849f9e8fad920ddd90053af60090b592906f34a64bca8a6481a5ce0` |
| Exact current local image tag | `voipmonitor/vllm:kimi-k3-hh-dense-mla-dcp8-it-local-full-20260803` |
| Exact current local image ID | `sha256:f453030864542a91babcbb3565ad185a87ff171a937fe18bf309f72e2270a3dc` |
| InstantTensor | `0.1.9+consumer1` |
| Model snapshot | `moonshotai/Kimi-K3` at `2496450e92e425c886db095102a52a6682ca3970` |

The Docker image is the pinned CUDA/Python/binary runtime. vLLM and SparkInfer
are intentionally bind-mounted source overlays, so commit `b3b8a189e` does not
require another enormous Docker layer. Reconstruct the container from the
published digest and mount the exact source commits above. The current local
image ID is recorded as an additional rollback snapshot, not as the portable
deployment boundary.

## Final launch contract

```text
checkpoint: original moonshotai/Kimi-K3 compressed-tensors MXFP4
non-expert weights: original BF16
loader: InstantTensor
tensor parallel / decode context parallel: 16 / 8
attention backend: B12X_MLA (SparkInfer dense MLA)
KDA prefill backend: FlashKDA with SM120 raw-TMA allocation fix
DCP transport: SparkInfer PCIe A2A for size-1 decode
KV dtype: FP8 E4M3
physical KV allocation: 1,860,000,000 bytes/rank
physical KV capacity: 1,054,602 tokens
maximum model length: 1,048,576
maximum sequences: 1
maximum batched tokens: 2,048
MLA chunked-prefill workspace: 8,192 tokens
prefix caching: disabled
CUDA graph: PIECEWISE, capture size 1
KDA f_a TP sharding: enabled
PyTorch allocator: expandable_segments:True
served model: Kimi-K3-MXFP4-HH-DenseMLA-DCP8-1M-Chunk2048
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by
`serve-kimi-k3.sh`; keep it enabled for all memory-tight profiles.

## Why chunk 2048 originally did not fit

The final fix removes four independent sources of peak or retained memory:

1. Dense-MLA previously retained one capture-stable scratch allocation per
   attention layer. The metadata group now owns and reuses one arena. The full
   profile retains only 4.42 MiB/rank.
2. KDA `f_a` was a replicated BF16 projection. With
   `additional_config={"kda_shard_f_a":true}`, its 128 output rows are split
   to eight rows/rank at TP16 and gathered before `f_b`. Across 69 KDA layers
   this saves exactly 126,615,552 bytes, or 120.75 MiB/rank.
3. The context workspace is bounded at 8,192 instead of 32,768 tokens. This
   returns 30.375 MiB/rank while keeping the 8k validation request in one
   gather segment at every 2048-token scheduler step.
4. A 2048-token MoE prefill needs a 28 MiB routed-up BF16 output. The final
   implementation reuses the now-dead full-width MoE input as the `torch.mm`
   output. The first version still exposed a second 28 MiB `empty_like` inside
   PyNCCL, so the generic communicator now supports an in-place all-reduce
   using the same send and receive pointer.

The routed-up arithmetic is unchanged: the shared-expert partial is still
combined through the original separate BF16 `add_`. The model-free CUDA
oracle measured `max_abs=0` against the old result.

The in-place PyNCCL path is selected only when the all-reduce output is at
least 1 MiB. Size-1 decode produces only about 14 KiB, so it retains the
original B12X custom all-reduce and captured graph. This preserves graph reuse,
mixed prefill/decode scheduling, and transition latency instead of tearing
down and recapturing the decode graph.

Commit `721e5150` separately fixes FlashKDA's SM120 launch-time allocation. It
preserves K1's swizzled workspace representation but reloads it in K2 through
raw-layout tensor TMA. The old non-tensor bulk-copy route committed about
3.74 GiB/rank on first use; the fixed module fits beside the physical 1M
cache and remains bit-exact with the original recurrence.

## Reconstruct sources and container

Clone and pin the source overlays:

```bash
git clone --filter=blob:none --single-branch \
  --branch codex/kimi-k3-hh-dense-mla-dcp8-20260803 \
  https://github.com/local-inference-lab/vllm.git \
  /mnt/luke/vllm-k3-hh-dense-mla-dcp8
git -C /mnt/luke/vllm-k3-hh-dense-mla-dcp8 checkout \
  b3b8a189e4010343b2d6b83451ed3c41d8b10ff9

git clone --single-branch \
  --branch codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803 \
  https://github.com/local-inference-lab/sparkinfer.git \
  /mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest
git -C /mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest checkout \
  f39c6bf26be9d92b65d1f031819289c8c1f084a1
```

Create the container using the command in the
[base HH page](hh-dense-mla-full-mxfp4-dcp8-2026-08-03.md#start-the-docker-environment).
Its portable image reference is:

```bash
docker pull voipmonitor/vllm@sha256:499c405bb849f9e8fad920ddd90053af60090b592906f34a64bca8a6481a5ce0
```

Link the installed vLLM extensions into the source overlay after container
creation, as documented in the
[source-overlay section](hh-dense-mla-full-mxfp4-dcp8-2026-08-03.md#reconstruct-the-source-overlays).
The branch preflight builds the Kimi-specific companion extensions when they
are absent.

## Launch

Optionally apply the validated host clock profile first:

```bash
nvidia-smi -lgc 3000,3090
nvidia-smi -lmc 14001,14001
```

Start the API:

```bash
docker exec -d \
  -e PYTHON_BIN=/opt/venv/bin/python \
  -e PYTHONPATH=/mnt/luke/vllm-k3-hh-dense-mla-dcp8:/mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest \
  -e SERVED_MODEL_NAME=Kimi-K3-MXFP4-HH-DenseMLA-DCP8-1M-Chunk2048 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  kimi-k3-hh bash -lc '
exec /mnt/luke/vllm-k3-hh-dense-mla-dcp8/serve-kimi-k3-full-mxfp4-dcp8-1m-chunk2048.sh \
  >/mnt/luke/k3-hh-full-mxfp4-dcp8-1m-chunk2048.log 2>&1
'
```

Resolve the bridge IP rather than hard-coding it:

```bash
server_ip=$(docker inspect kimi-k3-hh --format \
  '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
curl -f "http://${server_ip}:8000/health"
curl -s "http://${server_ip}:8000/v1/models" | jq \
  '.data[0] | {id, max_model_len}'
```

Expected startup evidence:

```text
Loading weights took 165.69 seconds
Model loading took 90.81 GiB and 188.52-188.85 seconds
GPU KV cache size: 1,054,602 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.01x
Application startup complete
```

CUDA graph capture takes about 2-3 seconds and consumes about 0.17 GiB/rank.
The shared dense-MLA scratch marker must report approximately 4.42 MiB.

## Validation

The full-model capacity run used exact token IDs, streamed one output token,
and disabled prefix caching. All requested prompt-token counts were confirmed
from API usage.

| Prompt | State | TTFT | Effective prefill |
|---:|---|---:|---:|
| 8,192 | first cold request | 2.7166 s | 3,015.58 tok/s |
| 8,192 | warmed median of 3 | 2.2927 s | 3,573.08 tok/s |
| 32,768 | measured run | 9.5144 s | 3,444.06 tok/s |
| 65,536 | measured run | 19.9731 s | 3,281.21 tok/s |

The warmed 8k result is only 0.27% below the 3,582.72 tok/s small-cache
reference. At 64k this is 4.07x the old physical-1M/chunk-256 result of
805.413 tok/s.

Three clock-profiled, forced 1,024-token decode runs with the same 256-token
input produced:

| Run | TTFT | Decode throughput |
|---:|---:|---:|
| 1 | 0.2509 s | 38.8211 tok/s |
| 2 | 0.2465 s | 38.0343 tok/s |
| 3 | 0.2465 s | 39.8079 tok/s |
| median | — | 38.8211 tok/s |

The 5.70% difference from the older 41.1667 tok/s clocked baseline is expected
from the extra KDA `f_a` gather. Its model-free TP16 graph harness projected
about 1.216 ms/token over 69 layers, or a 4.77% decode cost. The same harness
measured less than 1% prefill impact and a roughly 21.97 microsecond TP16
all-gather.

Outputs remained coherent. No repeated `!`, nonfinite values, CUDA errors,
OOM, assertion, or traceback occurred. An 8k prefill immediately followed by
decode also passed, proving the mixed transition and captured size-1 decode
path remained usable.

Useful local raw artifacts on the validated host are:

```text
/mnt/luke/k3-hh-full-mxfp4-dcp8-1m-chunk2048-routed-up-reuse-inplace-ar-v7-20260803.log
/root/vllm/kimi/checkpoints/kimi-k3-hh-dense-mla-dcp8/benchmarks/prefill-8k-full-mxfp4-dcp8-1m-chunk2048-routed-up-reuse-inplace-ar-first-20260803.json
/root/vllm/kimi/checkpoints/kimi-k3-hh-dense-mla-dcp8/benchmarks/prefill-8k-full-mxfp4-dcp8-1m-chunk2048-routed-up-reuse-inplace-ar-warmed-20260803.json
/root/vllm/kimi/checkpoints/kimi-k3-hh-dense-mla-dcp8/benchmarks/prefill-32k-64k-full-mxfp4-dcp8-1m-chunk2048-routed-up-reuse-inplace-ar-20260803.json
/root/vllm/kimi/checkpoints/kimi-k3-hh-dense-mla-dcp8/benchmarks/decode-full-mxfp4-dcp8-1m-chunk2048-inplace-ar-clocked-run{1,2,3}-20260803.json
```

Targeted validation completed before the full-model gate:

- `tests/models/kimi_k3/test_tp_projection.py`: 4 passed;
- `tests/v1/attention/test_b12x_mla.py`: 8 passed;
- four-layer TP16/DCP8 exact-checkpoint 8k prefill: HTTP 200;
- four-layer 128-token decode: 497.151 tok/s;
- standalone output-buffer oracle: `max_abs=0` and CUDA graph replay passed;
- Ruff, formatting checks, and `git diff --check`: clean.

The new two-GPU PyNCCL regression checks both the reduced value and preservation
of the original tensor `data_ptr()`. The production TP16 full-model run also
exercised this in-place path on every 2048-token MoE prefill chunk.

## Rollback

Stop only the active API process, keep the container and source mounts, and
launch the unchanged baseline wrapper:

```bash
docker exec -d \
  -e PYTHON_BIN=/opt/venv/bin/python \
  -e PYTHONPATH=/mnt/luke/vllm-k3-hh-dense-mla-dcp8:/mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest \
  kimi-k3-hh bash -lc '
exec /mnt/luke/vllm-k3-hh-dense-mla-dcp8/serve-kimi-k3-full-mxfp4-dcp8-1m.sh \
  >/mnt/luke/k3-hh-full-mxfp4-dcp8-dense-mla.log 2>&1
'
```

That profile returns to a 256-token scheduler chunk and 32,768-token MLA
workspace. For a complete pre-change reconstruction, check out
`6f1bcaa05ec603aba1e4b926c71aa8d4dcd8f05d` and follow the
[previous HH page](hh-dense-mla-full-mxfp4-dcp8-2026-08-03.md).
