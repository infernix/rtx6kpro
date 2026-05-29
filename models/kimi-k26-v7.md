# Kimi-K2.6 v7 on 8x RTX PRO 6000 Blackwell

Status: measured locally on 2026-05-29. This page is the Kimi-K2.6 v7 recipe
for the CUDA 13.2.1 GLM/Kimi image with the B12X PR8 small-M direct overlay.

The v7 sweep uses the same fastest MTP profile as the Kimi v5 standard-greedy
page: `festr2/kimi-k2.6-eagle3-mla-fp8`, Eagle3, `standard` rejection, and
`greedy` draft sampling. All measurements below were rerun on the local 8x RTX
PRO 6000 Blackwell host; earlier helper-host numbers from `10.229.14.14` were
discarded.

## Image

```bash
voipmonitor/vllm:cu132-vllm2f5db31f9bcd-b12xfbb76ca3a914
```

Image digest:

```text
sha256:1eab72d9c83a9f0a82420c8be2bab5b0266c502fb7a2400a3941382c59b34e66
```

Image metadata:

| Component | Revision |
|---|---|
| CUDA base | `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` |
| CUDA | `13.2.1` |
| cuBLAS runtime package | `libcublas-13-2 13.4.0.1-1` |
| cuBLAS dev package | `libcublas-dev-13-2 13.4.0.1-1` |
| cuDNN package | `libcudnn9-cuda-13 9.22.0.52-1` |
| cuDNN dev package | `libcudnn9-dev-cuda-13 9.22.0.52-1` |
| PyTorch | `2.12.0+cu132` |
| vLLM repo | `https://github.com/voipmonitor/vllm.git` |
| vLLM branch | `codex/glm51-v6-awq-mxfp8-clean-rebase-20260528` |
| vLLM commit | `2f5db31f9bcddf8d0cdd4d52f012759f50f37875` |
| B12X branch | `codex/glm51-v6-awq-mxfp8-pr8-smallm-20260528` |
| B12X commit | `fbb76ca3a91491c8f26a2edf729540414323e55b` |
| B12X overlay | `pr8-smallm-direct` |
| FlashInfer | `flashinfer-ai/flashinfer`, branch `main`, commit `8eb61546e82169759801c7895537f3c09ec423f9` |
| NCCL | `local-inference-lab/nccl-canonical`, branch `canonical/cu132-nccl2304-amd-noxml`, version `2.30.4` |
| vLLM package | `0.20.2+glm51v6.cu132.20260528` |
| B12X package | `0.15.3` |
| FlashInfer package | `0.6.12+cu132` |

Verify after pulling:

```bash
IMAGE=voipmonitor/vllm:cu132-vllm2f5db31f9bcd-b12xfbb76ca3a914

docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' | python3 -m json.tool
docker run --rm "$IMAGE" /usr/local/bin/verify-glm-kimi-cu132
docker run --rm --entrypoint /bin/bash "$IMAGE" -lc \
  'python - <<PY
import importlib.metadata as md
for p in ["vllm", "torch", "flashinfer-python", "b12x"]:
    print(p, md.version(p))
PY'
```

## Target And Draft

| Item | Value |
|---|---|
| Target model | `/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58` |
| Draft model | `festr2/kimi-k2.6-eagle3-mla-fp8` |
| Served name | `Kimi-K2.6` |
| Target attention | `TRITON_MLA` |
| Draft attention | `TRITON_MLA` |
| Target KV cache | `fp8` |
| Draft KV cache | `fp8` |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=1` |
| PCIe allreduce | `VLLM_ENABLE_PCIE_ALLREDUCE=0` |

## MTP Policy

The v7 speculative path is verifier-backed and lossless in vLLM. The measured
default is standard+greedy because it is the fastest Kimi v5/v7 MTP profile in
the current local tests.

```json
{
  "model": "festr2/kimi-k2.6-eagle3-mla-fp8",
  "method": "eagle3",
  "num_speculative_tokens": 3,
  "draft_attention_backend": "TRITON_MLA",
  "draft_kv_cache_dtype": "fp8",
  "rejection_sample_method": "standard",
  "draft_sample_method": "greedy"
}
```

For `MTP=0`, no speculative config is passed.

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
| DCP8 + MTP | 8 | 1 | 0.90 | starts, but decode requests crash in this v7 image; see results |
| DCP8 no-MTP | 8 | 0 | 0.94 | target-only baseline, larger KV |

If you need an exact MTP on/off A/B at identical memory pressure, set
`GPU_MEM=0.90` for both.

## Docker Run

Create the launcher:

```bash
cat >/tmp/run-kimi-k26-v7 <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:cu132-vllm2f5db31f9bcd-b12xfbb76ca3a914}"
NAME="${NAME:-kimi-k26-v7}"
PORT="${PORT:-8402}"
DCP="${DCP:-4}"
MTP="${MTP:-1}"
GPU_MEM="${GPU_MEM:-0.90}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/vllm-kimi-k26-v7}"
MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58}"

SPEC_CONFIG='{"model":"festr2/kimi-k2.6-eagle3-mla-fp8","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"standard","draft_sample_method":"greedy"}'

mkdir -p \
  "${CACHE_ROOT}/cutlass_dsl" \
  "${CACHE_ROOT}/jit" \
  "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/torchinductor" \
  "${CACHE_ROOT}/vllm"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

mtp_disable=0
if [[ "${MTP}" == "0" ]]; then
  mtp_disable=1
fi

exec docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' \
  --ipc=host --network host --privileged \
  --name "${NAME}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${CACHE_ROOT}/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "${CACHE_ROOT}/jit:/cache/jit" \
  -v "${CACHE_ROOT}/triton:/root/.cache/triton" \
  -v "${CACHE_ROOT}/torchinductor:/root/.cache/torchinductor" \
  -v "${CACHE_ROOT}/vllm:/root/.cache/vllm" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e USE_NCCL_XML=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e PORT="${PORT}" \
  -e MODEL="${MODEL_PATH}" \
  -e SERVED_MODEL_NAME=Kimi-K2.6 \
  -e TP_SIZE=8 \
  -e DCP_SIZE="${DCP}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEM}" \
  -e MAX_MODEL_LEN=262144 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_NUM_SEQS=128 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=512 \
  -e ATTENTION_BACKEND=TRITON_MLA \
  -e KV_CACHE_DTYPE=fp8 \
  -e LOAD_FORMAT=fastsafetensors \
  -e ENABLE_PREFIX_CACHING=1 \
  -e ENABLE_CHUNKED_PREFILL=1 \
  -e ENABLE_ASYNC_SCHEDULING=1 \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "$(cat <<RUN_EOF
set -euo pipefail
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
spec_args=()
if [[ "${mtp_disable}" != "1" ]]; then
  spec_args+=(--speculative-config '${SPEC_CONFIG}')
fi
exec /opt/venv/bin/vllm serve '${MODEL_PATH}' \
  --served-model-name Kimi-K2.6 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port '${PORT}' \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size '${DCP}' \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --gpu-memory-utilization '${GPU_MEM}' \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --enable-flashinfer-autotune \
  --max-cudagraph-capture-size 512 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  "\${spec_args[@]}"
RUN_EOF
)"
EOF

chmod +x /tmp/run-kimi-k26-v7
```

Example starts:

```bash
# DCP4 + MTP, default v7 throughput profile.
DCP=4 MTP=1 GPU_MEM=0.90 PORT=8402 /tmp/run-kimi-k26-v7

# DCP4 no-MTP baseline with larger KV cache.
DCP=4 MTP=0 GPU_MEM=0.94 PORT=8402 /tmp/run-kimi-k26-v7
```

Check readiness and KV cache:

```bash
curl -fsS http://127.0.0.1:8402/v1/models | jq .
docker logs kimi-k26-v7 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency' | tail -n 5
```

## Benchmark Command

The local sweep was run from:

```text
/root/bench-results/kimi-v7-local-matrix-clean-cu132-vllm2f5db31-b12xfbb76ca-20260529/
```

Each DCP/MTP combination was started in a fresh container. The runner verifies
that port `8402` is served by the intended container before measuring, to avoid
stale-server contamination from older local runs.

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 8402 \
  --model Kimi-K2.6 \
  --concurrency 1,32 \
  --contexts 0,128k \
  --duration 30 \
  --skip-prefill \
  --max-tokens 8192 \
  --temperature 0 \
  --kv-budget <server KV tokens> \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor \
  --output /root/bench-results/kimi-v7-local-matrix-clean-cu132-vllm2f5db31-b12xfbb76ca-20260529/dcp<DCP>/mtp<MTP>/result.json
```

`llm_decode_bench.py` version was `0.4.24`. `acc` is the average speculative
acceptance rate reported by server metrics for that cell. For `MTP=0`,
acceptance is `0.000` by definition. `N/A` means the cell was skipped or did
not fit. `crash` means the server started, but the measured request killed the
engine, so no throughput value is valid.

## Standard+Greedy Local Sweep Results

### DCP 1

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 491,344 | 89.4 | 0.000 | 1002.9 | 0.000 | 45.6 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 342,672 | 134.3 | 0.473 | 1226.6 | 0.417 | 62.9 | 0.346 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 2

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 982,688 | 79.3 | 0.000 | 891.4 | 0.000 | 54.7 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 685,344 | 131.8 | 0.423 | 1089.4 | 0.397 | 77.7 | 0.402 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 4

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 1,965,376 | 75.9 | 0.000 | 836.6 | 0.000 | 55.0 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 1,370,688 | 115.5 | 0.421 | 994.3 | 0.424 | 67.1 | 0.460 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 8

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 3,930,752 | 75.1 | 0.000 | 698.9 | 0.000 | 61.1 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 2,741,376 | crash | N/A | crash | N/A | crash | N/A | N/A | N/A | Server starts, but decode request kills engine; see notes |

## DCP8 MTP Crash

DCP8+MTP is not recorded as `0 tok/s` because the benchmark did not measure a
slow server. The server starts and reports a usable KV cache, but the first
decode request kills the engine.

Two local checks were run:

| Run | GPU mem | KV cache tokens | Outcome |
|---|---:|---:|---|
| Full sweep | 0.90 | 2,741,376 | Initial 128k request hit CUDA OOM / engine death |
| Headroom retry | 0.86 | 1,652,736 | No OOM, but first decode request hit CUDA illegal memory access |

The headroom retry failed in `TRITON_MLA` DCP metadata construction:

```text
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
vllm/v1/attention/backends/utils.py:get_dcp_local_seq_lens
decode_context_parallel_size=8
speculative_config=SpeculativeConfig(method='eagle3', model='festr2/kimi-k2.6-eagle3-mla-fp8', num_spec_tokens=3)
```

This makes DCP8+MTP a correctness/runtime issue in the v7 stack, not a valid
throughput result. DCP8 no-MTP is valid.

## Notes And Risks

- The earlier helper-host `10.229.14.14` measurements are intentionally not used
  here; local-only results above are the current v7 numbers.
- A stale local MTP server previously contaminated one no-MTP check on port
  `8402`. The clean sweep removed older Kimi v7 containers and verified the
  intended container before each measurement.
- The local benchmark uses `--max-tokens 8192` and `--temperature 0`. This is a
  better sustained decode setting than the older 2048-token default, but it is
  not directly comparable to old 2048-token wiki rows without rerunning them.
- `VLLM_ENABLE_PCIE_ALLREDUCE=0` is set explicitly, matching the v5 Kimi
  runbook.
- Persistent cache mounts matter. Keep `/cache/jit`, Triton, TorchInductor,
  CUTE DSL, and vLLM cache directories mounted across restarts to avoid repeated
  compile/autotune cost.
- `NCCL_GRAPH_FILE`, `NCCL_GRAPH_DUMP_FILE`, and
  `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` should be unset, not set to empty strings.
