# Kimi K3 full MXFP4 TP16/DCP8

Status date: 2026-08-02. This page records the working full, original
`moonshotai/Kimi-K3` MXFP4 checkpoint on 16 RTX PRO 6000 Blackwell GPUs with
TP16, DCP8, InstantTensor, SparkInfer communication/projection kernels, and a
physical one-million-token FP8 KV cache. No EXL3, NF3, expert parallelism,
pipeline parallelism, or repacked checkpoint is used.

## Validated result

| Item | Result |
|---|---:|
| Checkpoint | original `moonshotai/Kimi-K3` snapshot `2496450e...` |
| TP / DCP | 16 / 8 |
| Weight format | checkpoint-native ModelOpt MXFP4 experts, BF16 non-experts |
| Weight loader | InstantTensor |
| InstantTensor weight iteration | 167.35 s |
| Total model load through runner setup | 193.07 s |
| Model memory | 90.99 GiB/rank |
| KV allocation | 1.75 GiB/rank, 28 GiB over 16 ranks |
| Usable KV cache | 1,059,851 tokens |
| Maximum model length | 1,048,576 tokens |
| Maximum-length concurrency | 1.01x |
| Short completed decode | 34.8409 tok/s, 568 tokens |
| Long regression decode | 34.5360 tok/s, 4,305 tokens |

The speed is token emission only:

```text
(completion_tokens - 1) / (timestamp_last_token - timestamp_first_token)
```

It does not include prefill or TTFT. The 568-token request had 0.2974 s TTFT,
a 16.2740 s decode window, and a coherent two-paragraph final answer. The
4,305-token test crossed the historical deterministic corruption at token
4,204: it had zero non-finite logprobs and its longest literal `!` run was one.

## Published branches

No PR was opened. Both branches are based on Luke's `dev/gg-k3` work.

| Repository | Branch / head | Contents |
|---|---|---|
| vLLM | [`codex/kimi-k3-full-mxfp4-dcp8-20260802`](https://github.com/local-inference-lab/vllm/tree/codex/kimi-k3-full-mxfp4-dcp8-20260802) @ `0f3eb381a` | Full-MXFP4 DCP8 launcher, InstantTensor profile, TP-local K3 projection, dense Triton MLA DCP dispatch, recurrent-state replication, empty-shard sanitization, and fast validation harnesses |
| SparkInfer | [`codex/kimi-k3-full-mxfp4-dcp8-20260802`](https://github.com/local-inference-lab/sparkinfer/tree/codex/kimi-k3-full-mxfp4-dcp8-20260802) @ `01446e9` | TP16 PCIe oneshot, K3 H6/K128 MLA query projection, DCP8 A2A+LSE reduction, invalid-partial handling, and compact-I MoE tail fix |

Important new commits are:

```text
vLLM
0f3eb381a bench(kimi): add fast DCP8 validation harnesses
8a0a8cee0 fix(mla): sanitize empty DCP shard partials
852812c20 fix(dcp): replicate recurrent state across context ranks
c0c68a055 feat(dcp): enable SparkInfer A2A for dense Triton MLA
467d9ad89 fix(dcp): normalize MLA LSE before SparkInfer dispatch
41f991ce2 feat(kimi): TP-shard MLA latent input projection
f2d54f949 serve: add Kimi K3 full MXFP4 DCP8 1M profile

SparkInfer
01446e9 fix(moe): mask compact native FC2 scratch tails
543f693 fix(dcp): ignore invalid empty-shard partials
246dddd feat(mla): fuse Kimi K3 TP16 query projection
e8b6417 bench(pcie): add exact Kimi K3 DCP8 A2A validator
db40e90 feat(pcie): support TP16 oneshot all-reduce
```

## Reproduce

The vLLM branch contains the production launcher
`serve-kimi-k3-full-mxfp4-dcp8-1m.sh`. In the current Docker overlay:

```bash
PYTHON_BIN=/opt/venv/bin/python \
PYTHONPATH=/mnt/luke/sparkinfer-k3-full-mxfp4-dcp8 \
MODEL=/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970 \
SERVED_MODEL_NAME=Kimi-K3-MXFP4-DCP8-1M-tail-fix \
/mnt/luke/vllm-k3-full-mxfp4-dcp8/serve-kimi-k3-full-mxfp4-dcp8-1m.sh
```

The launcher fixes the validated contract:

```text
tensor_parallel_size=16
decode_context_parallel_size=8
dcp_comm_backend=a2a
dcp_kv_cache_interleave_size=1
max_model_len=1048576
max_num_seqs=1
max_num_batched_tokens=256
kv_cache_dtype=fp8
kv_cache_memory_bytes=1879048192
load_format=instanttensor
GPU_MEMORY_UTILIZATION=0.985
cudagraph=PIECEWISE, capture size [1]
VLLM_TRITON_MLA_STATIC_KV_SPLITS=8
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
```

The current validated instance was left running in container `cbc3ead9480b`
on port 8000. Its log is:

```text
/mnt/luke/k3-full-mxfp4-dcp8-tail-fix.log
```

Healthy startup markers include:

```text
Loading weights took 167.35 seconds
Model loading took 90.99 GiB memory and 193.072243 seconds
GPU KV cache size: 1,059,851 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.01x
Using B12X PCIe DCP collectives (world_size=8, max_batch_size=64,
  heads=48, query_head_dim=576, output_head_dim=512)
Warmed up 1 fused MLA BF16/MXFP8 query variants
Application startup complete
```

## What was fixed

### TP16 MLA query projection

K3 TP16 has six local MLA heads and `q_nope` width 128. SparkInfer's fused
BF16 query projection previously qualified only the DeepSeek-style K=192
geometries, so K3 fell back to staged BF16 operations. The new H6/K128 and
H8/K128 specializations fuse projection, RoPE append, and output conversion.
For H6/K128/M1, the isolated operation measured 5.568 us versus 9.152 us for
the staged path, 1.64x faster and bitwise identical in the tested output.

### DCP8 dense MLA communication

K3 has no sparse indexer or GLM-style CKV path. Its dense Triton MLA decode
now returns local normalized output plus LSE and uses SparkInfer's PCIe DCP
A2A+LSE reduction for batches up to 64. The exact geometry is eight DCP ranks,
six local heads, 48 gathered heads, query width 576, and output width 512.
Larger prefill uses the bounded AG+RS policy from the launcher.

An empty local DCP shard legitimately has `LSE=-inf`; Triton may leave its
partial output undefined. Both vLLM and SparkInfer now sanitize/skip those
contributors instead of evaluating IEEE `0 * NaN`. KDA/Mamba recurrent state
is marked DCP-replicated because every context rank must advance its own
TP-sharded head state for every token.

### K3 MXFP4 MoE corruption

The observed `!!!!`/NaN failure was ultimately inside the SparkInfer native
W4A16 small-M direct MoE kernel, not Triton MLA. On TP16, K3's routed
intermediate width is 192. The kernel reserves a 256-BF16 FC2 chunk, writes
only 192 values, but the narrow FC2 path read all 256. A zero weight did not
protect poisoned tail values because `0 * NaN = NaN`.

The proper fix masks the 64-BF16 unwritten tail inside the M=1 and M=2-8
microkernel paths. It adds no host memset, extra kernel launch, checkpoint
conversion, or fallback. The regression test pre-fills the tail with NaNs and
checks M=1/2/8 against an independent K3 H3584/I192 reference.

## Fast development and validation

Do not reload the 1.56-TB logical tensor set for every kernel iteration.
Use the linked four-layer checkpoint builder:

```bash
python benchmarks/make_kimi_k3_linked_truncated_checkpoint.py \
  --source /root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970 \
  --output /mnt/luke/models/Kimi-K3-MXFP4-trunc4-linked \
  --layers 4
```

It links source shards instead of copying weights. The resulting metadata is
about 2 MiB and InstantTensor iterates its weights in about 6.1 seconds. It
reproduced the MoE failure after 18 decode tokens before the fix and completed
128 forced decode tokens with finite logprobs after the fix.

Model-free DCP checks are:

```bash
python benchmarks/validate_kimi_k3_dcp8_a2a.py
python benchmarks/validate_kimi_k3_triton_mla_dcp8.py
python benchmarks/validate_kimi_k3_dense_dcp8_dispatch.py
```

The exact Triton MLA FP8-KV DCP8 validator passed eager BF16/FP32-LSE and CUDA
graph cases; worst maximum absolute error against the direct full-context
reference was 0.000999. The SparkInfer A2A validator passed eager batches
1/2/8/64 plus graph batch 1, including NaN outputs paired with invalid LSE.

Final checks recorded for this branch:

```text
SparkInfer compact-tail K3 tests: 3 passed (M=1/2/8, NaN-poisoned tail)
SparkInfer query projection suite: 65 passed
vLLM KV cache spec registry: 39 passed
full model: 256/256 finite logprobs smoke
full model: 4,305-token deterministic regression, 0 non-finite logprobs
```

## Capacity interpretation

DCP8 reduces the dense MLA token-history allocation to one eighth of DCP1.
The launcher therefore needs 1.75 GiB/rank rather than the measured 14
GiB/rank DCP1 allocation for approximately the same one-million-token cache.
Across 16 ranks that is 28 GiB instead of 224 GiB, a 196-GiB physical saving.
KDA recurrent state remains replicated by design; it is sequence state, not a
token-history KV cache.
