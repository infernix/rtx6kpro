# Kimi K3 full MXFP4 TP16 with Inferact DSpark-7

Status date: 2026-08-03. This is the first validated speculative-decode
profile for the original `moonshotai/Kimi-K3` MXFP4 checkpoint on 16 RTX PRO
6000 Blackwell GPUs. It uses the MLA-native
[`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark)
draft, SparkInfer dense MLA for both target and draft, and InstantTensor for
the target loader.

The target is unchanged full MXFP4. This profile is not the discarded NF3
hybrid, an EXL quant, EP, PP, or a padded checkpoint. The only padding is a
runtime-only zero padding of local MLA query heads from six/four to the
SparkInfer kernel tile of eight; the returned heads are sliced back.

## Result

With locked host clocks, three warmed 512-token greedy chat decodes produced
a median 74.100 decode tok/s. The same target without speculative decoding
measured 38.821 tok/s, so DSpark delivered 1.909x throughput (+90.9%). The
median accepted block length was 2.876 tokens.

The cost is context capacity. DSpark currently requires DCP1 and adds five
dense draft layers, so this profile reduces the maximum model length from the
DCP8 profile's physical 1M cache to 8,192 tokens.

## Immutable artifacts

| Component | Immutable reference |
|---|---|
| vLLM branch | [`codex/kimi-k3-dspark-inferact-20260803`](https://github.com/local-inference-lab/vllm/tree/codex/kimi-k3-dspark-inferact-20260803) |
| Validated vLLM commit | `eae1b41e01cdbca1ed23e7d25138edbf50dd1fb2` |
| HH full-MXFP4 base | `b3b8a189e4010343b2d6b83451ed3c41d8b10ff9` |
| SparkInfer branch | [`codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803`](https://github.com/local-inference-lab/sparkinfer/tree/codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803) |
| SparkInfer commit | `f39c6bf26be9d92b65d1f031819289c8c1f084a1` |
| Target snapshot | `moonshotai/Kimi-K3` at `2496450e92e425c886db095102a52a6682ca3970` |
| Draft snapshot | `Inferact/Kimi-K3-DSpark` at `cf6b8244620e7ea4b0651d214f28e89eac75bed6` |
| Draft weights | 7,124,633,450 bytes; SHA-256 `f9972a636d92a11994cdcfc88fd4c5b5d50d6eb2a89af016031593b8c65c2053` |
| Runtime image | `voipmonitor/vllm:kimi-k3-hh-dense-mla-dcp8-it-local-full-20260803` |
| Runtime image ID | `sha256:f453030864542a91babcbb3565ad185a87ff171a937fe18bf309f72e2270a3dc` |
| InstantTensor | `0.1.9+consumer1` |

The runtime image is the same pinned HH image used by the non-speculative
profile. vLLM and SparkInfer are bind-mounted source overlays; the DSpark
changes therefore do not require rebuilding the multi-hundred-gigabyte image.

## Why this draft

The Inferact checkpoint matches Kimi K3 directly: it has five dense draft
layers attached to target layers `[2, 23, 47, 71, 89]`, 64 draft attention
heads, and predicts a fixed seven-token block. At TP16 this gives four local
draft heads per rank. It is an MLA-native vLLM checkpoint and therefore fits
the existing K3 target stack better than the available Qwen/GQA/SGLang-
oriented alternatives.

The model card's acceptance figures were collected on a different GB300 and
FlashInfer stack. Treat the acceptance numbers in this page as the applicable
measurements for this RTX PRO 6000/SparkInfer deployment.

## Implementation required for TP16

The base HH branch already contained the K3 model implementation, DSpark
model classes, FlashKDA prefill, sharded BF16 target projections, and
SparkInfer dense MLA. Four integration gaps prevented an end-to-end TP16 run:

1. Kimi K3 target attention has 96 heads, or six local heads at TP16. The
   draft has four. SparkInfer dense MLA launches heads in tiles of eight. The
   adapter now uses capture-stable buffers to zero-pad 6 -> 8 and 4 -> 8,
   then slices output and LSE back to their real head counts.
2. The draft's seven non-causal proposal rows and the target's causal
   eight-row verification block cannot be passed to SparkInfer's one-query
   decode primitive as one request. The metadata builder flattens them to
   independent single-token rows. Target verification receives the exact
   sequence-length ramp from `final_length - 7` through `final_length`; draft
   siblings all see only the committed prefix.
3. `VLLM_KIMI_SHARD_QKV_A=1` changes each fused target Q/KV-A projection from
   replicated to TP-sharded layout. DSpark context-KV fusion now extracts the
   local KV portions, performs one fused local GEMM, gathers once, and restores
   layer-major order. This retains the intended 72.7% projection-FLOP saving.
4. vLLM warms speculative recurrent KDA before production KV allocation.
   Warmup now uses one correctly shaped temporary state page when the live
   cache is not bound yet.

The branch includes unit coverage for all four contracts. A real SparkInfer
CUDA oracle also compared padded and unpadded attention. BF16 target-like
attention had maximum output error `1.2207e-4`; FP8 draft-like attention had
maximum output error `2.4414e-4`. The causal eight-row flatten test had output
maximum error `1.2207e-4`, LSE maximum error `9.5367e-7`, and cosine similarity
`0.99999577`.

## Final runtime contract

```text
target: original moonshotai/Kimi-K3 compressed-tensors MXFP4
draft: Inferact/Kimi-K3-DSpark, five dense layers, seven proposals
loader: InstantTensor
tensor parallel / decode context parallel: 16 / 1
target and draft attention backend: B12X_MLA (SparkInfer dense MLA)
KDA prefill backend: FlashKDA
target and draft KV dtype: FP8 E4M3
physical KV allocation: 500,000,000 bytes/rank
physical KV capacity: 8,894 tokens
maximum model length: 8,192
maximum sequences: 1
maximum batched tokens: 2,048
prefix caching: disabled
CUDA graphs: PIECEWISE, capture sizes [1, 8]
KDA f_a and target QKV-A/MoE/router projection sharding: enabled
PCIe all-reduce: B12X
PyTorch allocator: expandable_segments:True
served model: Kimi-K3-MXFP4-HH-DSpark7
```

The scheduler exposes 2,042 ordinary batched tokens because six slots are
reserved for the speculative block. The difference from 2,048 is negligible.
The final process leaves only about 0.55-0.64 GiB free per GPU, so do not grow
the cache or graph set without first measuring the resulting peak.

## Reconstruct

Pin both source overlays:

```bash
git clone --filter=blob:none --single-branch \
  --branch codex/kimi-k3-dspark-inferact-20260803 \
  https://github.com/local-inference-lab/vllm.git \
  /mnt/luke/vllm-k3-dspark
git -C /mnt/luke/vllm-k3-dspark checkout \
  eae1b41e01cdbca1ed23e7d25138edbf50dd1fb2

git clone --single-branch \
  --branch codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803 \
  https://github.com/local-inference-lab/sparkinfer.git \
  /mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest
git -C /mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest checkout \
  f39c6bf26be9d92b65d1f031819289c8c1f084a1
```

Download the exact draft revision into the normal Hugging Face cache:

```bash
huggingface-cli download Inferact/Kimi-K3-DSpark \
  --revision cf6b8244620e7ea4b0651d214f28e89eac75bed6
```

Create or restore the `kimi-k3-hh` container as described by the
[base HH runbook](hh-dense-mla-full-mxfp4-dcp8-2026-08-03.md). It must bind
`/mnt/luke` and `/root/.cache/huggingface` at the same paths. Link the image's
compiled vLLM extensions into the vLLM source overlay as described there.

Optionally lock the validated clocks:

```bash
nvidia-smi -lgc 3000,3090
nvidia-smi -lmc 14001,14001
```

Start the service. All validated parameters are defaults in the launcher:

```bash
docker exec -d \
  -e PYTHON_BIN=/opt/venv/bin/python \
  -e SPARKINFER_DIR=/mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest \
  kimi-k3-hh bash -lc '
exec /mnt/luke/vllm-k3-dspark/serve-kimi-k3-full-mxfp4-dspark7.sh \
  >/mnt/luke/k3-dspark-full-mxfp4-tp16-dcp1-8k-kv500m.log 2>&1
'
```

Run the fast preflight without loading target weights:

```bash
docker exec \
  -e KIMI_DSPARK_PREFLIGHT_ONLY=1 \
  -e PYTHON_BIN=/opt/venv/bin/python \
  -e SPARKINFER_DIR=/mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest \
  kimi-k3-hh \
  /mnt/luke/vllm-k3-dspark/serve-kimi-k3-full-mxfp4-dspark7.sh
```

Expected preflight markers are `K3DSparkConfig`, `SparkInfer target/draft TP16
MLA contract: OK`, and `K3 target native-op preflight: OK`.

## Expected startup evidence

```text
target Loading weights took 163.85 seconds
draft Loading weights took 0.80 seconds
Model loading took 91.94 GiB/rank and about 188.8 seconds
GPU KV cache size: 8,894 tokens
Maximum concurrency for 8,192 tokens per request: 1.09x
target: local_heads=6, kernel_heads=8, page_size=768
draft: local_heads=4, kernel_heads=8, page_size=768
Graph capturing finished in 8 secs, took 0.20 GiB
Application startup complete
```

Resolve the container IP or query from inside the container; port 8000 is not
published by the existing rollback container:

```bash
docker exec kimi-k3-hh curl -f http://127.0.0.1:8000/health
docker exec kimi-k3-hh curl -s http://127.0.0.1:8000/v1/models | jq \
  '.data[0] | {id, max_model_len}'
```

## Decode validation

The measured requests used `/v1/chat/completions`, temperature zero, a fixed
technical prompt, 512 forced output tokens, and clocks locked as above.
Throughput is derived from vLLM's output-token-time counter rather than TTFT
or aggregate prefill throughput.

| Run | Wall throughput | Decode throughput | Mean accepted length |
|---:|---:|---:|---:|
| 1 | 69.289 tok/s | 73.221 tok/s | 2.844 |
| 2 | 71.444 tok/s | 74.100 tok/s | 2.876 |
| 3 | 73.724 tok/s | 76.244 tok/s | 2.954 |
| median | 71.444 tok/s | 74.100 tok/s | 2.876 |

After committing, the service was restarted from the exact pinned source and
two additional forced 512-token smoke runs measured 72.826 and 72.464 wall
tok/s. Their Prometheus time-per-output-token observations sum to
`0.02643678` seconds, equivalent to 75.65 decode tok/s after averaging the two
observations. This exact-commit restart therefore showed no regression.

The non-speculative full-MXFP4 TP16/DCP8 baseline was 38.821 tok/s. DCP sizes
differ because vLLM's DSpark implementation is not currently compatible with
decode context parallelism; the comparison nevertheless uses the same target
weights and host clocks.

Two additional correctness runs exercised longer decode:

- 2,048 completion tokens remained coherent and finite; the request stopped
  only because it reached that diagnostic limit.
- A separate request generated 1,870 completion tokens and ended naturally
  with `finish_reason=stop`. It remained coherent with no repeated `!`,
  garbling, CUDA error, nonfinite assertion, or traceback.

The raw completion endpoint may continue checkpoint pretraining-like text.
That is expected base-completion behavior; validate instruction following via
the chat endpoint and the Kimi chat template.

## Tests

Run the targeted suite inside the image environment:

```bash
docker exec -w /mnt/luke/vllm-k3-dspark kimi-k3-hh bash -lc '
/opt/venv/bin/python -m pytest -q \
  tests/models/test_dspark_mla.py \
  tests/v1/attention/test_b12x_mla.py \
  tests/model_executor/test_kimi_k3_triton_warmup.py \
  tests/transformers_utils/test_dspark_mla_config.py
'
```

Validated result: 37 passed. Ruff, `bash -n`, and `git diff --check` also pass.

## Rollback

Stop only the active API process, retain the container and caches, check out
the previously pinned HH source commit, and follow the
[physical-1M non-speculative runbook](full-mxfp4-tp16-dcp8-1m-chunk2048-2026-08-03.md).
That returns to full MXFP4 TP16/DCP8, 1,054,602 physical KV tokens, and the
38.821 tok/s measured decode baseline.
