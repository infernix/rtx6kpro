# Kimi-K2.6 v7 on 8x RTX PRO 6000 Blackwell

Status: measured 2026-05-29. This page records the Kimi-K2.6 v7 speed sweep
using the CUDA 13.2.1 GLM/Kimi image with the B12X PR8 small-M overlay.

The v7 sweep uses the same fastest MTP profile as the Kimi v5 standard-greedy
page: `festr2/kimi-k2.6-eagle3-mla-fp8`, Eagle3, `standard` rejection, and
`greedy` draft sampling.

## Image And Checkpoints

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:glm51-v7-awq-mxfp8-b12x-pr8-smallm-cu132-20260528` |
| Image digest | `sha256:1eab72d9c83a9f0a82420c8be2bab5b0266c502fb7a2400a3941382c59b34e66` |
| Target model | `/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58` |
| Draft model | `festr2/kimi-k2.6-eagle3-mla-fp8` |
| Served name | `Kimi-K2.6` |
| Target attention | `TRITON_MLA` |
| Draft attention | `TRITON_MLA` |
| KV cache | `fp8` |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=1` |
| PCIe allreduce | `VLLM_ENABLE_PCIE_ALLREDUCE=0` |

Image labels:

```text
vllm:       https://github.com/voipmonitor/vllm/tree/codex/glm51-v6-awq-mxfp8-clean-rebase-20260528
vllm sha:   2f5db31f9bcddf8d0cdd4d52f012759f50f37875
b12x:       https://github.com/voipmonitor/b12x/tree/codex/glm51-v6-awq-mxfp8-pr8-smallm-20260528
b12x sha:   fbb76ca3a91491c8f26a2edf729540414323e55b
flashinfer: 8eb61546e82169759801c7895537f3c09ec423f9
cuda:       13.2.1
cublas:     13.4.1.2-1
cudnn:      9.22.0.52-1
nccl:       local-inference-lab/nccl-canonical canonical/cu132-nccl2304-amd-noxml
```

## Runtime Profiles

Use these profile values unless doing a strict A/B:

| Profile | `DCP` | `MTP` | `GPU_MEM` | Notes |
|---|---:|---:|---:|---|
| DCP1 + MTP | 1 | 1 | 0.90 | standard+greedy speculative profile |
| DCP1 no-MTP | 1 | 0 | 0.94 | target-only baseline, larger KV |
| DCP2 + MTP | 2 | 1 | 0.90 | standard+greedy speculative profile |
| DCP2 no-MTP | 2 | 0 | 0.94 | target-only baseline, larger KV |
| DCP4 + MTP | 4 | 1 | 0.90 | standard+greedy speculative profile |
| DCP4 no-MTP | 4 | 0 | 0.94 | target-only baseline, larger KV |
| DCP8 + MTP | 8 | 1 | 0.90 | long-context profile, but this sweep had incomplete cc32/128k output |
| DCP8 no-MTP | 8 | 0 | 0.94 | target-only baseline, larger KV |

If you need an exact MTP on/off A/B at identical memory pressure, set
`GPU_MEM=0.90` for both.

## MTP Configuration

```text
{"model":"festr2/kimi-k2.6-eagle3-mla-fp8","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"standard","draft_sample_method":"greedy"}
```

For `MTP=0`, no speculative config is passed.

## Benchmark Command

This sweep was run on helper host `10.229.14.14` with the same image and model
cache layout. The helper host uses a one-CPU system and different switches, so
small transport differences versus the main machine are possible. Raw artefacts
were copied back to the main host after the run.

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 5317 \
  --model Kimi-K2.6 \
  --concurrency 1,32 \
  --contexts 0,128k \
  --duration 30 \
  --skip-prefill \
  --max-tokens 2048 \
  --kv-budget <server KV tokens> \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor \
  --output /root/bench-results/v6-v7-speed-sweep-20260529/kimi/dcp<DCP>/mtp<MTP>/result.json
```

`llm_decode_bench.py` version on the helper host was `0.4.15`. `acc` is the
average speculative acceptance rate reported by server metrics for that cell.
For `MTP=0`, acceptance is `0.000` by definition. `N/A` means the cell was
skipped, did not fit, or produced no clean measured output.

## Standard+Greedy Limited Sweep Results

Result directory:

```text
/root/bench-results/v6-v7-speed-sweep-20260529/kimi/
```

### DCP 1

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 491,344 | 99.4 | 0.000 | 1133.6 | 0.000 | 51.0 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 342,672 | 142.6 | 0.315 | 1301.4 | 0.382 | 63.9 | 0.419 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 2

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 982,688 | 78.0 | 0.000 | 981.9 | 0.000 | 60.4 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 685,344 | 114.4 | 0.600 | 1098.8 | 0.412 | 71.9 | 0.404 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 4

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 1,965,376 | 75.1 | 0.000 | 931.8 | 0.000 | 60.9 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 1,370,688 | 122.5 | 0.362 | 830.8 | 0.397 | 65.9 | 0.495 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 8

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 3,930,752 | 82.4 | 0.000 | N/A | N/A | 64.7 | 0.000 | N/A | N/A | cc32 produced only 2 client tokens and no completion; rerun before publishing as steady-state |
| 1 | 0.90 | 2,741,376 | 113.2 | 0.365 | N/A | N/A | N/A | N/A | N/A | N/A | cc32 and 128k/c1 produced no measured output; rerun required |

## Notes And Risks

- `VLLM_ENABLE_PCIE_ALLREDUCE=0` is set explicitly, matching the v5 Kimi runbook.
- The DCP8 rows are not clean enough to use as primary publishable numbers. Use
  the DCP1/2/4 rows for v7 comparison until DCP8 is rerun.
- Persistent cache mounts matter. Keep `/cache/jit`, Triton, TorchInductor,
  CUTE DSL, and vLLM cache directories mounted across restarts to avoid repeated
  compile/autotune cost.
- `NCCL_GRAPH_FILE`, `NCCL_GRAPH_DUMP_FILE`, and
  `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` should be unset, not set to empty strings.
