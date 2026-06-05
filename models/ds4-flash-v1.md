# DeepSeek-V4-Flash v1 on 8x RTX PRO 6000 Blackwell

Status: measured locally on 2026-06-05. This page records the current
`dev/nameless-ascent` B12X/vLLM image for DeepSeek-V4-Flash TP2/TP4 and the
first clean MTP-on/off decode sweep.

## Image

```bash
voipmonitor/vllm:cu132-vllm05cc7ba-b12x60a63d8-fi-e8d3131-ds4glm-20260605
```

Image ID:

```text
sha256:0dbf3ecffe7ea69874b107169a33ce70a558db1896a6e14d56a19c48a9d613a8
```

Image metadata:

| Component | Revision |
|---|---|
| CUDA | `13.2.1` |
| cuBLAS | `13.4.1.2-1` |
| cuDNN | `9.22.0.52-1` |
| NCCL | `2.30.4`, `local-inference-lab/nccl-canonical` |
| PyTorch | `2.12.0+cu132` |
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `codex/nameless-ascent-66b2a76-upstream-pr1-20260605` |
| vLLM commit | `05cc7ba06` |
| Luke branch source | `https://github.com/local-inference-lab/vllm/tree/dev/nameless-ascent` at `66b2a7688c753b160a2856f41e069560fddce8fb` |
| B12X repo | `https://github.com/lukealonso/b12x.git` |
| B12X commit | `60a63d8cc5cb9eb5022304af79e6abc5c2cca576` |
| FlashInfer repo | `https://github.com/flashinfer-ai/flashinfer.git` |
| FlashInfer commit | `e8d31317bedb4efd52559a2234f4cb9e83428cb9` |

Important fixes in this image:

| Area | Fix |
|---|---|
| DS4 MTP | `serve-ds4-flash.sh` defaults to `use_local_argmax_reduction:false`; the old `true` setting produced corrupted short outputs and CJK leakage. |
| Loader | DS4 Flash defaults to `--load-format safetensors`; current public snapshot has safetensors shards, not usable InstantTensor artifacts. |
| DCP sparse MLA | DCP sparse MLA disables full CUDA graph replay and uses piecewise graphs to avoid long-context corruption for DCP>1. |

Verify:

```bash
IMAGE=voipmonitor/vllm:cu132-vllm05cc7ba-b12x60a63d8-fi-e8d3131-ds4glm-20260605

docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' | python3 -m json.tool
docker run --rm "$IMAGE" bash -lc \
  'grep -n use_local_argmax_reduction /opt/vllm/serve-ds4-flash.sh;
   /opt/vllm/.venv/bin/python -c "import importlib.metadata as md; [print(p, md.version(p)) for p in (\"vllm\",\"torch\",\"flashinfer-python\",\"b12x\")]"'
```

## Runtime

Default launcher:

```text
/opt/vllm/serve-ds4-flash.sh
```

Default model:

```text
deepseek-ai/DeepSeek-V4-Flash
```

Default MTP config when `VLLM_ENABLE_MTP=1`:

```json
{
  "method": "mtp",
  "num_speculative_tokens": 2,
  "draft_sample_method": "probabilistic",
  "moe_backend": "b12x",
  "use_local_argmax_reduction": false
}
```

The spec config can be overridden without editing the image:

```bash
DS4_SPEC_CONFIG_JSON='{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x","use_local_argmax_reduction":false}'
```

## Docker Run

```bash
cat >/tmp/run-ds4-flash-v1 <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:cu132-vllm05cc7ba-b12x60a63d8-fi-e8d3131-ds4glm-20260605}"
NAME="${NAME:-ds4-flash-v1}"
PORT="${PORT:-5329}"
TP_SIZE="${TP_SIZE:-4}"
MTP="${MTP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.875}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-130000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-4096}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

exec docker run -d --gpus all \
  --name "${NAME}" \
  -p "${PORT}:${PORT}" \
  -v /mnt:/mnt \
  -v /cache:/cache \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e PORT="${PORT}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e DCP_SIZE=1 \
  -e VLLM_ENABLE_MTP="${MTP}" \
  -e LOAD_FORMAT=safetensors \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  "${IMAGE}" \
  bash -lc 'cd /opt/vllm && ./serve-ds4-flash.sh'
EOF
chmod +x /tmp/run-ds4-flash-v1
```

Examples:

```bash
# TP4 + MTP, default benchmark profile.
TP_SIZE=4 MTP=1 PORT=5329 /tmp/run-ds4-flash-v1

# TP4 no-MTP.
TP_SIZE=4 MTP=0 PORT=5329 /tmp/run-ds4-flash-v1

# TP2 + MTP.
TP_SIZE=2 MTP=1 PORT=5329 /tmp/run-ds4-flash-v1
```

Readiness:

```bash
curl -fsS http://127.0.0.1:5329/v1/models | jq .
docker logs ds4-flash-v1 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Graph capturing finished|Application startup complete' | tail -20
```

## Correctness Smoke

`python3 /mnt/test.py --port 5329` and
`python3 /mnt/test.py --port 5329 -c10000` were run for TP2/TP4 MTP on/off.
All four combinations returned coherent English output with `0` CJK characters
after disabling local argmax reduction for MTP.

| Profile | Short gen tok/s | 10k-context gen tok/s | CJK |
|---|---:|---:|---:|
| TP2 no-MTP | 108.7 | 115.6 | 0 |
| TP2 MTP | 203.3 | 207.4 | 0 |
| TP4 no-MTP | 126.7 | 136.3 | 0 |
| TP4 MTP | 250.0 | 257.6 | 0 |

The broken configuration was:

```json
{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x","use_local_argmax_reduction":true}
```

It reproduced corrupted Python output and CJK leakage on short smoke tests.

## Decode Sweep

Command used for every row:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 5329 \
  --model DeepSeek-V4-Flash \
  --skip-prefill \
  --contexts 0k \
  --concurrency 1,16,64 \
  --duration 30 \
  --max-tokens 8192 \
  --display-mode plain \
  --output OUT.json
```

Results are under:

```text
/root/bench-results/ds4-flash-nameless-20260605/
```

Aggregate decode tok/s:

| TP | MTP | cc1 | cc16 | cc64 |
|---:|---:|---:|---:|---:|
| 2 | off | 110.6 | 682.3 | capacity-limited, effective 16/64 |
| 2 | on | 190.3 | 879.8 | capacity-limited, effective 16/64 |
| 4 | off | 129.8 | 966.7 | not fit / skipped |
| 4 | on | 241.7 | 1341.0 | not fit / skipped |

Per-request tok/s:

| TP | MTP | cc1 | cc16 | cc64 |
|---:|---:|---:|---:|---:|
| 2 | off | 110.6 | 42.6 | capacity-limited |
| 2 | on | 190.3 | 55.0 | capacity-limited |
| 4 | off | 129.8 | 60.4 | not fit / skipped |
| 4 | on | 241.7 | 83.8 | not fit / skipped |

Raw JSON:

```text
/root/bench-results/ds4-flash-nameless-20260605/tp2-nomtp-cc1-16-64.json
/root/bench-results/ds4-flash-nameless-20260605/tp2-mtp-cc1-16-64.json
/root/bench-results/ds4-flash-nameless-20260605/tp4-nomtp-cc1-16-64.json
/root/bench-results/ds4-flash-nameless-20260605/tp4-mtp-cc1-16-64.json
```

## Notes

`cc64` is not a useful headline for this profile because `MAX_NUM_SEQS=16` and
the runner either skips or records it as capacity-limited. Use cc1 and cc16 as
the comparable kernel/scheduler signal for this page.

The server logs show B12X PCIe oneshot allreduce for TP groups and PYNCCL for
EP groups. FlashInfer autotune is enabled, but this DS4 B12X path logs that no
FlashInfer compute kernels are active during the warmup.
