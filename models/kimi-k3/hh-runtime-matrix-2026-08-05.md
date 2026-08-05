# Kimi K3 HH TP16 runtime matrix

## TL;DR

Use one immutable Kimi K3 image for all validated HH profiles:

```text
voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-20260805@sha256:2407e74fba03d5074acc3d2a7b1a1340bb21905fcacc560186bb3322c4e8f125
```

The image pins vLLM PR #238 at
`b7e203d0bb6e1456b858277a388726f93f0d1ff6` and SparkInfer PR #118 at
`024a7607ee7e025921b02e140f83f01687165bcf`.

Profile selection:

| Requirement | Profile | Measured result |
|---|---|---:|
| Highest validated CC1 throughput | `dcp16-dspark` | about 106-115 tok/s at short context |
| Exact target checkpoint, no speculative decoding | `dcp16-no-dspark` | 47.118 tok/s CC1; 204.559 tok/s aggregate CC8 |
| Exact target checkpoint with DCP8 | `dcp8-no-dspark` | 38.054 tok/s CC1 median |
| Exact target weights plus BF16 DSpark | `dcp16-dspark-full` | functional reference profile; smaller scheduler budget |
| Historical DCP8 speculative profile | `dcp8-dspark` | 99.65 tok/s CC1, but target shared experts are MXFP8 |

"Exact target checkpoint" means that routed experts stay in the checkpoint's
MXFP4 representation and dense weights stay BF16. DCP, fused communication, and
the projection transport optimizations are lossless layout/runtime changes; they
do not add weight quantization.

## Source and Docker provenance

| Component | Reference |
|---|---|
| vLLM | [PR #238](https://github.com/local-inference-lab/vllm/pull/238), commit `b7e203d0bb6e1456b858277a388726f93f0d1ff6` |
| SparkInfer | [PR #118](https://github.com/local-inference-lab/sparkinfer/pull/118), commit `024a7607ee7e025921b02e140f83f01687165bcf` |
| Docker recipe | [build/kimi-k3-hh-runtime-20260805](https://github.com/local-inference-lab/blackwell-llm-docker/tree/build/kimi-k3-hh-runtime-20260805), commit `e1755c5f7ec062b83ffd895217fc07dd58fbabf1` |
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Draft checkpoint | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` |

The Docker image contains the runtime source and compiled Kimi K3 compatibility
extensions, but no model weights. It inherits the CUDA 13.2/InstantTensor runtime
from the preserved HH image and pins both source trees by full commit SHA.

## Start a profile

Clone the Docker recipe branch once:

```bash
git clone --branch build/kimi-k3-hh-runtime-20260805 \
  https://github.com/local-inference-lab/blackwell-llm-docker.git \
  blackwell-llm-docker-kimi-k3
cd blackwell-llm-docker-kimi-k3
```

The helper mounts the host Hugging Face cache, publishes port 8000, enables all
16 GPUs, and always sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

```bash
./run-kimi-k3-hh.sh dcp16-dspark
./run-kimi-k3-hh.sh dcp16-dspark-full
./run-kimi-k3-hh.sh dcp16-no-dspark
./run-kimi-k3-hh.sh dcp16-no-dspark-batch8
./run-kimi-k3-hh.sh dcp8-no-dspark
./run-kimi-k3-hh.sh dcp8-dspark
```

Override the host cache, host port, container name, or image as needed:

```bash
HF_CACHE=/models/hf PORT=8001 NAME=kimi-k3-dcp8 \
  ./run-kimi-k3-hh.sh dcp8-no-dspark
```

Equivalent direct launch for DCP8 target-only:

```bash
KIMI_IMAGE='voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-20260805@sha256:2407e74fba03d5074acc3d2a7b1a1340bb21905fcacc560186bb3322c4e8f125'

docker run --rm \
  --name kimi-k3-hh-dcp8-no-dspark \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt label=disable \
  -p 8000:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint /opt/kimi-k3-hh/vllm/serve-kimi-k3-full-mxfp4-dcp8-1m-no-dspark-optimized.sh \
  "${KIMI_IMAGE}"
```

Wait for readiness:

```bash
docker logs -f --tail 100 kimi-k3-hh-dcp8-no-dspark
curl -f http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | jq
```

## Profile details

### DCP16 with DSpark and selective KDA MXFP8

Launcher:
`serve-kimi-k3-full-mxfp4-dspark7-dcp16-1m-kda-mxfp8.sh`.

- Target routed experts: checkpoint-native MXFP4.
- Target shared experts: BF16.
- Target online conversion: 69 low-sensitivity KDA input projections to
  MXFP8/Marlin, saving about 1.36 GiB/rank.
- Draft: five BF16 Inferact DSpark layers, seven proposals, TP-sharded Markov
  head, replicated 32K draft-KV tail.
- Target KV: FP8, 1,057,049 physical tokens.
- Scheduler: 4096 tokens.
- Prefill medians at 8K/32K/64K: 3,451/3,589/3,475 tok/s.
- Decode: about 106-115 tok/s at short context, 110.7 tok/s at 100K, and
  49.7 tok/s at 200K.
- KLD versus the original full-MXFP4 reference: 0.003183 over 65,504
  positions; top-1 agreement 98.826%.

This is the throughput default when speculative decoding and the measured KDA
precision tradeoff are acceptable.

### DCP16 without DSpark

Launchers:

- CC1: `serve-kimi-k3-full-mxfp4-dcp16-1m-no-dspark.sh`.
- Up to eight concurrent requests:
  `serve-kimi-k3-full-mxfp4-dcp16-1m-no-dspark-batch8.sh`.

This path has no online weight quantization. It uses the fused paired projection
transport, precomputed top-k routing, deferred hierarchical collective
consumption, BF16x2 transport, persistent small-M MoE barriers, and the fast DCP
A2A dispatch from PRs #238/#118.

Measured with 512 forced decode tokens after warmup:

- CC1: 47.117707 tok/s.
- CC8: 204.559323 tok/s aggregate decode; 26.0351 tok/s median per request.
- Physical FP8 target KV: 1,460,937 tokens for a 1,048,576-token model limit.
- Scheduler: 2048 tokens.

### DCP8 without DSpark

Launcher:
`serve-kimi-k3-full-mxfp4-dcp8-1m-no-dspark-optimized.sh`.

This is the requested exact-checkpoint DCP8 control. It uses the same lossless
decode optimizations as DCP16. The dense `f_a` projections remain replicated by
default because that is faster at CC1. The resulting memory headroom is very
tight, so the validated launcher uses a 256-token scheduler budget and a
2048-row Triton MLA prefill workspace.

Validated 2026-08-05:

| Item | Result |
|---|---:|
| Physical FP8 KV | 1,054,602 tokens |
| Model limit | 1,048,576 tokens |
| CC1 decode runs | 36.749 / 38.054 / 38.354 tok/s |
| CC1 median / mean | 38.054 / 37.719 tok/s |
| InstantTensor model load | 191.55 s |
| Loaded model memory | 90.96 GiB/rank |
| Final free VRAM | about 138 MiB/rank |

The original scheduler/workspace settings missed startup by about 21 MiB/rank
during Triton autotuning. Reducing only the transient scheduler/workspace budget
allowed the same physical 1M cache and unmodified target checkpoint to start.
For larger prefill chunks, set `KIMI_SHARD_F_A=1` and increase
`MAX_NUM_BATCHED_TOKENS`; that trades extra gathers during decode for transient
memory and is not the CC1 speed profile measured above.

### DCP16 full-target DSpark reference

Launcher:
`serve-kimi-k3-full-mxfp4-dspark7-dcp16-1m-hierarchical-bf16x2.sh`.

This keeps all target dense weights BF16 and target routed experts MXFP4 while
using the BF16 DSpark draft. It uses a 2048-token scheduler and a 1,299,000,000
byte/rank target cache allocation. Use it for exact-target speculative A/B tests;
the selective-KDA profile has the stronger validated throughput/prefill envelope.

### Historical DCP8 DSpark profile

Launcher: `serve-kimi-k3-full-mxfp4-dspark7-dcp8-1m.sh`.

It reached 99.65 tok/s median CC1 and a physical 1M target cache, but it converts
the target shared experts to MXFP8 to fit BF16 DSpark plus a replicated 32K draft
tail. It is retained for reproduction, not recommended as the accuracy default.

## Loader and attention backends

- Weight loading is `--load-format instanttensor`; the full target tensor load is
  roughly three minutes on this host, not the 16-minute safetensors path.
- Dense MLA decode uses SparkInfer B12X and the fused Kimi K3 cache/KDA ops.
- KDA prefill remains Triton. The archived FlashKDA SM120 work reduced persistent
  module memory to about 0.119 GiB/rank, but was 0.37% slower than Triton in the
  equal-MBT4096 8K end-to-end test. Its source and measurements are preserved in
  [archive/hh-kimi-k3-flashkda-sm120-20260804](https://github.com/local-inference-lab/vllm/tree/archive/hh-kimi-k3-flashkda-sm120-20260804).

## Decode benchmark method

The CC1 numbers use one 136-token coding prompt, `temperature=0`,
`ignore_eos=true`, a 128-token warmup, and a 512-token measured completion. The
reported decode rate excludes prefill and the first generated token:

```text
(completion_tokens - 1) / (last_token_time - first_token_time)
```

Machine-readable results are in
[`benchmarks/hh-runtime-matrix-20260805.json`](benchmarks/hh-runtime-matrix-20260805.json).
The stream benchmark helper supports `--allowed-token-id` for a deterministic
hidden-state/routing A/B control.

## Rollback images

These earlier immutable images remain available independently of PR #238/#118:

- Stable pre-HH full-MXFP4 DCP8:
  `voipmonitor/vllm:kimi-k3-full-mxfp4-dcp8-it-stable-20260803@sha256:2c99435142dd10f85834eaf4c490cb3d4095318152f0cc4fb38c7623d7edb7ac`.
- HH dense-MLA DCP8:
  `voipmonitor/vllm:kimi-k3-hh-dense-mla-dcp8-it-20260803@sha256:499c405bb849f9e8fad920ddd90053af60090b592906f34a64bca8a6481a5ce0`.
- Previous PR #238/#118 DSpark DCP16 snapshot:
  `voipmonitor/vllm:kimi-k3-hh-dspark-dcp16-pr238-pr118-20260804@sha256:a13b98f7c420c32c776872896a30c53867aa9dbab1b895c03d5ff15984179722`.
