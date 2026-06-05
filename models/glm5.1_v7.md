# GLM-5.1 v7 on 8x RTX PRO 6000 Blackwell

Status: smoke-validated locally on 2026-06-05. This page records the GLM-5.1
runtime on the same `dev/nameless-ascent` B12X/vLLM image used for the
DeepSeek-V4-Flash v1 page.

## Image

```bash
voipmonitor/vllm:cu132-vllm05cc7ba-b12x60a63d8-fi-e8d3131-ds4glm-20260605
```

Image ID:

```text
sha256:0dbf3ecffe7ea69874b107169a33ce70a558db1896a6e14d56a19c48a9d613a8
```

Source revisions:

| Component | Revision |
|---|---|
| CUDA | `13.2.1` |
| cuBLAS | `13.4.1.2-1` |
| cuDNN | `9.22.0.52-1` |
| NCCL | `2.30.4` |
| PyTorch | `2.12.0+cu132` |
| vLLM branch | `codex/nameless-ascent-66b2a76-upstream-pr1-20260605` |
| vLLM commit | `05cc7ba06` |
| B12X commit | `60a63d8cc5cb9eb5022304af79e6abc5c2cca576` |
| FlashInfer commit | `e8d31317bedb4efd52559a2234f4cb9e83428cb9` |

Relevant runtime fixes:

| Area | Fix |
|---|---|
| DCP sparse MLA | For DCP>1, B12X sparse MLA reports no full CUDA graph support, forcing piecewise graph mode and avoiding long-context corruption. |
| Spec decode | Draft attention receives DCP-local sequence lengths. |
| OpenAI serving | `CompletionOutput` import is fixed for chat serving. |
| ModelOpt sparse metadata | Sparse routed expert metadata is inferred for ModelOpt mixed checkpoints. |

## Runtime

Launcher:

```text
/opt/vllm/serve-glm51.sh
```

Validated model path:

```text
/mnt/glm51-luke-nvfp4-mtp-nvfp4routed-symlink
```

The tested profile used:

```text
TP_SIZE=8
ATTENTION_BACKEND=B12X_MLA_SPARSE
MOE_BACKEND=auto
KV_CACHE_DTYPE=fp8
B12X_W4A16_TC_DECODE=1
B12X_MOE_FORCE_A16=0
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_USE_B12X_FP8_GEMM=1
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
```

## Docker Run

```bash
cat >/tmp/run-glm51-v7 <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:cu132-vllm05cc7ba-b12x60a63d8-fi-e8d3131-ds4glm-20260605}"
NAME="${NAME:-glm51-v7}"
PORT="${PORT:-5329}"
DCP_SIZE="${DCP_SIZE:-4}"
MTP="${MTP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MODEL="${MODEL:-/mnt/glm51-luke-nvfp4-mtp-nvfp4routed-symlink}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-130000}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-64}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

disable_mtp=0
if [[ "${MTP}" == "0" ]]; then
  disable_mtp=1
fi

exec docker run -d --gpus all \
  --name "${NAME}" \
  -p "${PORT}:${PORT}" \
  -v /mnt:/mnt \
  -v /cache:/cache \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e PORT="${PORT}" \
  -e MODEL="${MODEL}" \
  -e MTP_MODEL="${MODEL}" \
  -e SERVED_MODEL_NAME=GLM-5.1 \
  -e TP_SIZE=8 \
  -e DCP_SIZE="${DCP_SIZE}" \
  -e GLM51_DISABLE_MTP="${disable_mtp}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e MAX_NUM_SEQS=64 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  -e KV_CACHE_DTYPE=fp8 \
  -e ATTENTION_BACKEND=B12X_MLA_SPARSE \
  -e MOE_BACKEND=auto \
  -e B12X_MOE_FORCE_A16=0 \
  -e B12X_W4A16_TC_DECODE=1 \
  "${IMAGE}" \
  bash -lc 'cd /opt/vllm && ./serve-glm51.sh'
EOF
chmod +x /tmp/run-glm51-v7
```

Examples:

```bash
# DCP4 + MTP.
DCP_SIZE=4 MTP=1 GPU_MEMORY_UTILIZATION=0.90 PORT=5329 /tmp/run-glm51-v7

# DCP4 no-MTP baseline.
DCP_SIZE=4 MTP=0 GPU_MEMORY_UTILIZATION=0.93 PORT=5329 /tmp/run-glm51-v7

# DCP8 + MTP.
DCP_SIZE=8 MTP=1 GPU_MEMORY_UTILIZATION=0.90 PORT=5329 /tmp/run-glm51-v7
```

Do not force `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS`. Leave it unset so scratch
planning can choose a DCP-safe value.

## Smoke Validation

Validation command:

```bash
python3 /mnt/test.py --port 5329
python3 /mnt/test.py --port 5329 -L -c10000
```

For `-L -c10000`, the first completed iteration was used as the validation
sample and the watchdog was stopped before the next loop iteration.

| Profile | KV cache tokens | 130k max concurrency | Short result | 10k-context result |
|---|---:|---:|---|---|
| DCP1 + MTP | 459,670 | 3.54x | coherent, 0 CJK, 104.1 tok/s | coherent, 0 CJK, 84.9 tok/s |
| DCP4 no-MTP | not recorded | not recorded | coherent, 0 CJK | coherent, 0 CJK, 10.9 tok/s |
| DCP4 + MTP | 1,838,681 | 14.14x | coherent, 0 CJK | coherent, 0 CJK, 27.8 tok/s |
| DCP8 + MTP | 3,677,362 | 28.29x | coherent, 0 CJK, 29.8 tok/s | coherent, 0 CJK, 34.7 tok/s |

DCP4 + MTP OOMed during long prefill at `GPU_MEMORY_UTILIZATION=0.93`.
`GPU_MEMORY_UTILIZATION=0.90` passed.

## DCP Graph Note

The corruption root cause for DCP sparse MLA was full CUDA graph replay. The
current branch makes B12X sparse MLA return no full-graph support when
`decode_context_parallel_size > 1`; vLLM then logs that
`FULL_AND_PIECEWISE` is not supported and falls back to `PIECEWISE`. DCP1 still
uses the normal full/piecewise graph path.

This is a correctness guard, not a final graph-stable DCP implementation.

## Pending

The page currently records correctness and smoke throughput. A full GLM v7
decode matrix should be added after DS4 Flash v1 image publishing is complete.
