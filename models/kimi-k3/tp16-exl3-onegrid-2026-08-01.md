# Kimi K3 TP16 MXFP4/EXL3 one-grid

Status date: 2026-08-02. This is the reproducible state of the no-EP TP16
Kimi K3 hybrid, including quantization, InstantTensor loading, one-grid MoE,
1M KV capacity, KLD, and decode measurements.

## Result

The production artifact is:

```text
/mnt/luke/models/Kimi-K3-EXL3-3p09-MXFP4-44p0-TP16-1M-20260801-serve
```

It contains all 896 experts per MoE layer. Experts selected by the allocation
stay in their original ModelOpt MXFP4 representation; the remainder are
EXL3-3 MCG quantized after TP16 sharding. K3's 3,072-wide routed intermediate
dimension becomes 192 channels per rank, represented as an H128 transform plus
an H64 tail. There is no expert parallelism, expert padding, pipeline
parallelism, or TP8 fallback.

Decode M=1/top-k=16 executes the MXFP4 and EXL3 tiers in one cooperative
SparkInfer grid. The packaged artifact has 92 MoE layers and 81 unique
`(MXFP4, EXL3)` expert-count splits.

The final B12X E2E run established:

- InstantTensor weight iteration: 146.17 seconds;
- total model load: 164.051 seconds, 75.28 GiB/rank;
- explicit FP8 KV allocation: 14 GiB/rank;
- usable KV cache: 1,084,486 tokens, enough for `max_model_len=1,048,576`;
- concurrency at maximum model length: 1.03x;
- GPU use after startup: about 94,033 MiB/rank, 3,220 MiB free/rank;
- all 1,472 rank-layer one-grid arms completed, with zero fallback/errors;
- true single-request decode: 34.6667 tok/s mean over three 384-token runs.

The otherwise identical 31.2288 tok/s baseline used PyNCCL because the
image's generic `VLLM_PCIE_ALLREDUCE_BACKEND=cpp` cannot serve world size 16.
The TP16 SparkInfer oneshot raises decode by 3.4379 tok/s, or 11.01%. This
does not include prefill.

## Published branches (no PR)

| Repository | Branch / head | Purpose |
|---|---|---|
| SparkInfer | [`codex/kimi-k3-mxfp4-exl3-onegrid-20260801`](https://github.com/local-inference-lab/sparkinfer/tree/codex/kimi-k3-mxfp4-exl3-onegrid-20260801) @ `ae87654` | Mixed MXFP4/EXL3 one-grid, fixed TP16 Trellis planning/tail dispatch, TP16 PCIe oneshot, and model-free benchmarks |
| vLLM | [`codex/kimi-k3-tp16-mxfp4-exl3-onegrid-20260801`](https://github.com/local-inference-lab/vllm/tree/codex/kimi-k3-tp16-mxfp4-exl3-onegrid-20260801) @ `2bd7f0478` | TP-local artifact loader, one-grid dispatch, fixed-capacity MXFP4 prompt tails, production launcher, and exact decode benchmark |
| kquant | [`codex/kimi-k3-tp16-local-exl3-onegrid-20260801`](https://github.com/local-inference-lab/kquant/tree/codex/kimi-k3-tp16-local-exl3-onegrid-20260801) @ `f45264c` | Resumable TP16-local EXL3 conversion, validation, and packaging |
| ExLlamaV3 fork | [`codex/kimi-k3-tp16-local-hadamard-20260801`](https://github.com/voipmonitor/exllamav3/tree/codex/kimi-k3-tp16-local-hadamard-20260801) @ `c1518ee` | H128+H64 EXL3 transforms and bounded conversion scratch |

These are development branches based on Luke's K3 work; no PR was opened.

## Reproduce the server

The vLLM branch contains the complete launcher:

```bash
SPARKINFER_SRC=/path/to/sparkinfer \
  ./serve-kimi-k3-exl3-tp16-1m.sh
```

Its important fixed parameters are:

```text
TP=16, PP=1, DCP=1, EP=false
max_model_len=1048576
max_num_seqs=1
max_num_batched_tokens=256
kv_cache_memory_bytes=15032385536
kv_cache_dtype=fp8
load_format=instanttensor
cudagraph=PIECEWISE, capture size [1]
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
```

The launcher intentionally overrides the GG image's generic `cpp` all-reduce
and graph-memory-estimation defaults. To make a deliberate A/B override, use
the `K3_PCIE_*` or `K3_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` variables defined
in the script, rather than inheriting the image defaults accidentally.

Current container overlays and runtime paths on this machine are:

```text
container: cbc3ead9480b (luke)
vLLM:      /tmp/vllm-onegrid-test.Xj0hv9
SparkInfer:/tmp/sparkinfer-k3-onegrid
ExLlamaV3: /tmp/exllamav3-k3-tp16
kquant:    /tmp/kquant-k3-onegrid
PID file:  /mnt/luke/k3-tp16-exl3-onegrid.pid
log:       /mnt/luke/k3-tp16-exl3-onegrid-tp16ar-20260802T101927Z.log
```

InstantTensor does not create a second resident model copy. It selects tensors
through the checkpoint index and streams each physical tensor directly to its
consumer. `INSTANTTENSOR_BUFFER_SIZE=536870912` was used. The remaining long
startup phase is not weight I/O: compiling/arming the 81 unique one-grid layer
splits took 21 minutes 53 seconds after the 2.4-minute weight iteration. The
complete engine profile/KV/warmup phase took 1,350.72 seconds on this cold run.
SparkInfer's object-cache key includes the fingerprint of the entire
SparkInfer source package and the physical GPU UUID. Consequently, even an
unrelated SparkInfer source edit causes one cold 81-split rebuild on all 16
devices; an unchanged overlay can reuse the objects on the next restart.

## Bugs found and fixed

1. The TP16-local EXL3 artifact declares its H128+H64 contract. The loader
   rejects globally quantized or incompatible artifacts instead of silently
   slicing the wrong transform.
2. Lazy W4A16 execution now preserves the prepared Trellis tile configuration.
   K3 TP16 projection-major I=192 requires N64 FC1 tiles; the generic M=256
   heuristic incorrectly selected N256.
3. Trellis bindings select the smallest already-prewarmed launch capacity
   covering the live token count and the matching prewarmed top-k sum launch.
   Prompt tails no longer trigger a new kernel specialization.
4. Ordinary MXFP4 planning initializes the neutral Hadamard-tail metadata,
   fixing an `UnboundLocalError` outside the Trellis path.
5. Original-MXFP4 prompt tails pad only the token-independent MoE invocation
   to the profiled M=256 capacity, then trim the output. Decode stays M=1.
6. SparkInfer PCIe oneshot now dispatches world size 16 in both plain and
   fused all-reduce+add+RMSNorm kernels. The underlying ABI was already sized
   for `kMaxRanks=16`; the Python whitelist and CUDA dispatch cases were
   missing. The current K3 server uses the plain operation: breakable CUDA
   graphs disable vLLM's Inductor pipeline, so its `fuse_allreduce_rms` pass
   cannot currently wire the otherwise validated fused kernel into the model.

The exact model-free TP16 communication gate is:

```bash
python benchmarks/benchmark_pcie_oneshot_tp16.py \
  --world-size 16 --hidden-size 3584 \
  --warmup 100 --iterations 1000
```

Measured twice on this host for the 7,168-byte K3 decode row:

| Path | Maximum rank latency |
|---|---:|
| SparkInfer TP16 oneshot | 14.38 us |
| PyNCCL | 35.63 us |
| SparkInfer fused AR+add+RMSNorm | 17.60 us |

The plain collective is 2.48x faster than NCCL and both correctness checks
pass. Evidence is in
`/mnt/luke/k3-tp16-pcie-oneshot-model-free-20260802-rerun.jsonlog`.

## Decode benchmark

`benchmarks/kimi_k3_decode_stream.py` sends the exact first 256
token IDs of the canonical KLD window to `/v1/completions`, streams 384 output
tokens with EOS ignored, and calculates:

```text
(completion_tokens - 1) / (timestamp_last - timestamp_first)
```

The authoritative NCCL runs are:

```text
/mnt/luke/k3-tp16-exl3-onegrid-decode-run1-20260802T0953Z.json
/mnt/luke/k3-tp16-exl3-onegrid-decode-run2-20260802T0953Z.json
/mnt/luke/k3-tp16-exl3-onegrid-decode-run3-20260802T0953Z.json
```

They measured 31.2348, 31.2387, and 31.2128 tok/s. This is decode only, not
prefill or vLLM's aggregated rolling metric.

With the TP16 SparkInfer B12X path active, the authoritative runs are:

```text
/mnt/luke/k3-tp16-exl3-onegrid-b12x-decode-run1-20260802T1048Z.json
/mnt/luke/k3-tp16-exl3-onegrid-b12x-decode-run2-20260802T1048Z.json
/mnt/luke/k3-tp16-exl3-onegrid-b12x-decode-run3-20260802T1048Z.json
```

| Run | Decode tok/s | Decode interval |
|---:|---:|---:|
| 1 | 34.6999 | 11.0375 s |
| 2 | 34.6561 | 11.0514 s |
| 3 | 34.6440 | 11.0553 s |
| **Mean** | **34.6667** | |

The sample standard deviation is 0.02945 tok/s. All three streams contained
exactly 384 timed token events and coherent text, with no NUL bytes or
exclamation-mark collapse. The service log contains no error, traceback, or
fallback after the requests.

## KLD quality result

The complete 32-window x 2,048-token comparison for this exact quant is:

```text
/mnt/luke/kld/kimi-k3-exl3-3p09-mxfp4-44p0-tp16-fp8kv1m-kld-32x2048-mixed-v1-20260801
```

It contains 65,504 compared positions and 39.98 GiB of logits. Results:

| Metric | TP16 hybrid |
|---|---:|
| mean KL(reference || candidate) | 0.01715409 |
| median KL | 0.00017697 |
| p95 / p99 KL | 0.07146779 / 0.30196805 |
| maximum KL | 4.26138 |
| mean JS | 0.00407232 |
| top-1 agreement | 97.4353% |
| 95% window-bootstrap CI for mean KL | 0.01257895-0.02231703 |

This is better than the TP12 artifact (mean KL 0.0199675, top-1 97.183%).
The capture manifest used an earlier EP-enabled runtime, so these numbers prove
the artifact's quantization quality, not the current no-EP execution speed.
The canonical capture and comparison commands are documented in
[`kld-reference-logits-32x2048.md`](kld-reference-logits-32x2048.md), with
scripts under `models/kimi-k3/tools/`.

## Operational notes

- Use `/v1/completions` with token IDs for repeatable decode measurements;
  chat rendering is not part of the speed result.
- Startup logs must contain 92 layers x 16 ranks armed, one-grid execution,
  and no serial fallback/error.
- A non-256 prompt tail should not emit a new `size_m=<tail>` W4A16 compile
  after the API becomes ready.
- The 28-token prompt-tail smoke test completed over HTTP with no MoE compile,
  fallback, or error. Its metrics are in
  `/mnt/luke/k3-tp16-exl3-onegrid-b12x-tail28-smoke.json`.
- The process recorded in `/mnt/luke/k3-tp16-exl3-onegrid.pid` was left
  running after the tests.
