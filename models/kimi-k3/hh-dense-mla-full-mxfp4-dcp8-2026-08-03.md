# Kimi K3 HH + SparkInfer dense-MLA full-MXFP4 runtime

Status date: 2026-08-03. This page reconstructs the validated successor to
the [pre-HH stable checkpoint](stable-full-mxfp4-dcp8-checkpoint-2026-08-03.md).
It serves the original `moonshotai/Kimi-K3` checkpoint on 16 RTX PRO 6000
Blackwell GPUs using TP16, DCP8, SparkInfer native dense MLA, native MXFP4
experts, FP8 KV, a physical one-million-token cache, and InstantTensor.

This is not the EXL3 hybrid checkpoint. It does not use NF3, expert
parallelism, pipeline parallelism, or model-level MoE padding.

## Pinned artifacts

| Component | Immutable reference |
|---|---|
| Docker image | `voipmonitor/vllm:kimi-k3-hh-dense-mla-dcp8-it-20260803` |
| Docker registry digest | `sha256:499c405bb849f9e8fad920ddd90053af60090b592906f34a64bca8a6481a5ce0` |
| Registry image ID | `sha256:17fb0819b6385b6640a456c5b3f4fd756b372c87fc702dd8a1d6b7e012170264` |
| Runtime base digest | `sha256:2c99435142dd10f85834eaf4c490cb3d4095318152f0cc4fb38c7623d7edb7ac` |
| Full local commit tag | `voipmonitor/vllm:kimi-k3-hh-dense-mla-dcp8-it-local-full-20260803` |
| Full local commit ID | `sha256:f453030864542a91babcbb3565ad185a87ff171a937fe18bf309f72e2270a3dc` |
| vLLM | [`codex/kimi-k3-hh-dense-mla-dcp8-20260803`](https://github.com/local-inference-lab/vllm/tree/codex/kimi-k3-hh-dense-mla-dcp8-20260803) at `6f1bcaa05ec603aba1e4b926c71aa8d4dcd8f05d` |
| vLLM review | [`local-inference-lab/vllm#232`](https://github.com/local-inference-lab/vllm/pull/232), merged |
| vLLM HH base | `dev/heraldic-harbinger` at `bce8a43539e3f0db8e366a600142a236ad4d4904` |
| SparkInfer | [`codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803`](https://github.com/local-inference-lab/sparkinfer/tree/codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803) at `f39c6bf26be9d92b65d1f031819289c8c1f084a1` |
| SparkInfer production fix | `a84463014bba9933e69c67da0f8a983f9b1e149f` (the following commit adds benchmark tools only) |
| SparkInfer review | [`local-inference-lab/sparkinfer#116`](https://github.com/local-inference-lab/sparkinfer/pull/116), merged |
| SparkInfer base | `master` at `77154c105f441777355df1817ab660a8151fb294` |
| InstantTensor | version `0.1.9+consumer1` |
| InstantTensor wheel SHA256 | `1077d0b8fe0d97ee4b759018e1dc8b801fd8bdc65802a8ff00f391043f6fbabf` |
| Model | `moonshotai/Kimi-K3` snapshot `2496450e92e425c886db095102a52a6682ca3970` |

The pinned branches preserve the exact validated pair. vLLM PR #232 contains
the HH integration, Kimi DCP chunked-prefill fix, and bounded-workspace
profile. SparkInfer PR #116 contains the native E8M0 aligned-FC1 correctness
fix. Both reviews were merged.

Docker commit does not include bind mounts. The image is the runtime
environment; the two pinned source branches and the model snapshot are also
required. Labels named `local-inference-lab.vllm.commit` and
`local-inference-lab.sparkinfer.commit` identify the exact source overlays.
Older `local-inference.*` labels describe the original base image.

The 97,759,866,504-byte full local commit is retained under the local-only tag
above. Its single writable layer also captured non-runtime compile caches and
temporary process files; Docker Hub repeatedly returned HTTP 502 partway
through that monolithic upload. The published image is instead rebuilt by
[`docker/hh-dense-mla-runtime.Dockerfile`](docker/hh-dense-mla-runtime.Dockerfile)
on the already-published stable runtime digest. This is the actual deployment
boundary: HH vLLM and SparkInfer are bind-mounted source overlays. A sorted
`pip freeze` diff between that base and the running HH container was empty,
and both report InstantTensor `0.1.9+consumer1`. The published image therefore
contains the complete required runtime without uploading disposable caches.

## What changed from the rollback checkpoint

The HH branch expected `sparkinfer.attention.dense_mla`, which was absent from
the older SparkInfer overlay. Luke's `77154c1` SparkInfer base now contains the
planned native dense-MLA implementation. The vLLM integration:

- gathers six local TP16 query heads across DCP8 into the 48-head SparkInfer
  layout;
- attends against each rank's one-eighth KV shard with FP8 storage;
- combines the eight partial outputs through SparkInfer's PCIe DCP A2A/LSE
  reduction;
- keeps KDA prefill on Triton because loading FlashKDA's SM120 module consumes
  about 3.74 GiB/rank, which cannot fit beside the physical 1M cache;
- uses InstantTensor directly, without a second safetensors loading pass.

Kimi's fused prefill entry point originally called the non-DCP context path
even when `dcp_world_size > 1`. On the second scheduler chunk that path read a
rank-local KV shard as a complete cache and produced an illegal memory access.
vLLM commit `6058224b7` dispatches chunked context to
`_context_parallel_compute_prefill_context` under DCP and retains the old path
for DCP1.

The attention split is deliberate and unchanged from the HH base: decode uses
SparkInfer `B12X_MLA`, while dense MLA prefill uses vLLM FlashAttention FA2.
KDA prefill is a third, separate Triton path. No FlashInfer prefill selection
change is part of the branch.

The HH model had to shard the otherwise replicated BF16 QKV-A, routed-down,
routed-up, and router projections to fit the unmodified checkpoint. The
routed-up partial is combined with the shared-expert partial before one
all-reduce. The resulting full-model allocation is about 90.93 GiB/rank.

SparkInfer commit `a844630` also fixes a correctness bug in the generic aligned
FC1 path for native E8M0 MXFP4 weights. Kimi K3 has `K=3584` split into seven
aligned FC1 segments. The old branch treated K/32 E8M0 scales as packed K/16
E4M3 scales in that path, doubled the expert stride, and could read outside the
weight allocation for high expert IDs. The fix selects the native E8M0 loader
before the W4A16 E4M3 branch for both gate and up weights. It is covered by a
targeted oracle test and the arbitrary-route lifecycle harness.

## Validated runtime contract

```text
model format: checkpoint-native compressed-tensors MXFP4 + BF16 non-experts
load format: instanttensor
TP / DCP: 16 / 8
attention backend: B12X_MLA (SparkInfer native dense MLA)
DCP backend: a2a (SparkInfer PCIe collectives for size-1 decode)
DCP KV interleave: 1
maximum model length: 1,048,576
maximum sequences: 1
maximum batched tokens: 256
MLA chunked-prefill workspace: 32,768 tokens
KV cache dtype: FP8 E4M3
KV cache allocation: 1,860,000,000 bytes/rank
GPU memory utilization: 0.985
CUDA graph mode: PIECEWISE, capture size 1
KDA prefill backend: Triton
PyTorch CUDA allocator: expandable_segments:True
served model name: Kimi-K3-MXFP4-HH-DenseMLA-DCP8-1M
```

The native dense-MLA plan is:

```text
local heads: 6
effective heads after DCP8 gather: 48
page size: 768
maximum local cache tokens: 131,072
capture-static splits: 94
maximum decode rows: 1
```

Validated startup and capacity before clock tuning:

| Item | Result |
|---|---:|
| InstantTensor checkpoint traversal | 164.84 s |
| Full model construction | 188.69-189.02 s across ranks |
| Model allocation | 90.93 GiB/rank |
| Physical GPU KV cache | 1,054,602 tokens |
| 1M request concurrency | 1.01x |
| CUDA graph capture | about 2 s / 0.18 GiB |

## Long-prefill capacity gate

The original 64k MLA context workspace consumes 81 MiB/rank under DCP8:
`(65,536 + 65,536/8) * 576 * 2` bytes. It also permits up-projected K and V
temporaries for all 64k context tokens to coexist. With the full
1,860,000,000-byte KV allocation, a 64k request failed in FlashAttention's
temporary `pad(v)` after processing about 60k tokens. The failed allocation
was 84 MiB with only 81.25 MiB physically free. The base launcher already set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; repeating the test with
that setting confirmed this was real headroom, not allocator fragmentation.

Commit `6f1bcaa05` adds
`VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE`. The full-MXFP4 launcher defaults it
to 32,768 tokens. Under DCP8 the persistent workspace becomes 40.5 MiB and the
expanded context operands are bounded to half their previous row count.
FlashAttention still computes the same exact attention and vLLM merges the
additional context partial with online softmax; only the number of exact
chunks changes.

The E2E capacity run retained all 1,054,602 physical KV tokens and completed a
direct-token-ID 65,536-token request:

| Gate | Result |
|---|---:|
| 64k TTFT | 81.3695 s |
| 64k effective prefill | 805.413 tok/s |
| Completion tokens | 1 |
| Post-gate 1,024-token decode | 38.3467 tok/s |
| Decode TTFT | 0.2645 s |

The decode result matches the prior unlocked 38.1146 tok/s median within run
variance. Output remained coherent and the server log contained no CUDA,
OOM, nonfinite, assertion, or traceback error. Raw reports are
[`prefill-64k-full-mxfp4-dcp8-1m-ws32k-20260803.json`](benchmarks/prefill-64k-full-mxfp4-dcp8-1m-ws32k-20260803.json)
and
[`full-hh-ws32k-decode-1024.json`](benchmarks/full-hh-ws32k-decode-1024.json).

The old stable median was 34.9848 decode tok/s. Three warmed 1,024-token runs
on the new HH stack produced:

```text
37.824386 tok/s
39.935691 tok/s
38.114578 tok/s
mean:   38.624885 tok/s
median: 38.114578 tok/s
```

That median is 8.95% above the rollback floor. An additional run during coarse
GPU profiling reached 40.1199 tok/s. All outputs were coherent and the old
repeated-`!`/nonfinite failure did not recur.

The final full-model run used the optional clock profile and the same fixed
256-token input for all three forced 1,024-token decode requests:

```text
40.948979 tok/s
41.401056 tok/s
41.166727 tok/s
mean:   41.172254 tok/s
median: 41.166727 tok/s
```

That is 17.67% above the rollback median and 8.01% above the unlocked HH
median. An earlier clock-locked trio ran concurrently with Docker compression
and a large registry upload and measured only 38.016-38.316 tok/s (38.269
median); it is retained as host-load evidence, not used as the final gate.

The final InstantTensor checkpoint pass took 166.29 seconds; all ranks
completed model construction in 191.98-192.62 seconds at 90.93 GiB per rank.
It retained a 1,054,602-token physical cache and 1.01x concurrency for a
1,048,576-token request. The server log contained no CUDA, nonfinite,
assertion, or traceback error after the smoke test and all measured runs.

## Reconstruct the source overlays

Create the exact paths used by the container:

```bash
git clone --filter=blob:none --single-branch \
  --branch codex/kimi-k3-hh-dense-mla-dcp8-20260803 \
  https://github.com/local-inference-lab/vllm.git \
  /mnt/luke/vllm-k3-hh-dense-mla-dcp8
git -C /mnt/luke/vllm-k3-hh-dense-mla-dcp8 checkout \
  6f1bcaa05ec603aba1e4b926c71aa8d4dcd8f05d

git clone --single-branch \
  --branch codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803 \
  https://github.com/local-inference-lab/sparkinfer.git \
  /mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest
git -C /mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest checkout \
  f39c6bf26be9d92b65d1f031819289c8c1f084a1
```

The validated runtime directories matched these checkouts byte-for-byte for
tracked production files. Only `.ruff_cache` entries differed.

Link the installed vLLM binary extensions into the source overlay after the
container starts:

```bash
docker exec kimi-k3-hh bash -lc '
set -euo pipefail
src=/mnt/luke/vllm-k3-hh-dense-mla-dcp8/vllm
wheel=/opt/venv/lib/python3.12/site-packages/vllm
for binary in "$wheel"/*.so; do
  ln -sfn "$binary" "$src/$(basename "$binary")"
done
mkdir -p "$src/vllm_flash_attn"
for binary in "$wheel"/vllm_flash_attn/*.so; do
  ln -sfn "$binary" "$src/vllm_flash_attn/$(basename "$binary")"
done
'
```

The Kimi-specific activation, cache, and KDA companion extensions are built
by the branch preflight when absent. Do not bypass the launcher's dense-MLA and
extension checks; failing before a 1.4-TiB load is intentional.

## Start the Docker environment

Use the immutable registry digest:

```bash
docker pull voipmonitor/vllm@sha256:499c405bb849f9e8fad920ddd90053af60090b592906f34a64bca8a6481a5ce0

docker run -d \
  --name kimi-k3-hh \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --security-opt label=disable \
  --restart unless-stopped \
  -p 127.0.0.1:22012:22 \
  -v /mnt/luke:/mnt/luke \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/luke-docker/ssh:/root/.ssh \
  -v /root/luke-docker/ssh-host:/etc/ssh-host:ro \
  voipmonitor/vllm@sha256:499c405bb849f9e8fad920ddd90053af60090b592906f34a64bca8a6481a5ce0
```

The same image can be reconstructed locally from the checked-in Dockerfile:

```bash
docker build --pull=false \
  -f models/kimi-k3/docker/hh-dense-mla-runtime.Dockerfile \
  -t voipmonitor/vllm:kimi-k3-hh-dense-mla-dcp8-it-20260803 \
  models/kimi-k3/docker
```

The original container remains `cbc3ead9480b`. Port 8000 is not published;
resolve the current bridge address after each container creation:

```bash
docker inspect kimi-k3-hh --format \
  '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
```

Verify InstantTensor before loading:

```bash
docker exec kimi-k3-hh /opt/venv/bin/python -c \
  'import importlib.metadata as m; print(m.version("instanttensor"))'
docker exec kimi-k3-hh sha256sum \
  /root/k3-serve/instanttensor-0.1.9+consumer1-cp312-cp312-linux_x86_64.whl
```

Expected values are `0.1.9+consumer1` and the SHA256 in the artifacts table.

## Optional low-duty-cycle clock profile

Short single-sequence decode alternates brief 99% SM bursts with idle gaps.
Default dynamic clocks gave a 4-layer median of 526.868 tok/s. Locking graphics
clocks to 3000-3090 MHz gave 539.649 tok/s (+2.43%). Locking the memory clock
also added about 1% in a separate run, although `nvidia-smi` continued to
report 13,365 MHz on this driver.

Apply the speed profile on the host:

```bash
nvidia-smi -lgc 3000,3090
nvidia-smi -lmc 14001,14001
```

This raises idle draw substantially (roughly 80-105 W/GPU instead of about
10-25 W/GPU). Reset both locks with:

```bash
nvidia-smi -rgc
nvidia-smi -rmc
```

Clock locking is host state, not part of the Docker image.

## Launch the full model

```bash
docker exec -d \
  -e PYTHON_BIN=/opt/venv/bin/python \
  -e PYTHONPATH=/mnt/luke/vllm-k3-hh-dense-mla-dcp8:/mnt/luke/sparkinfer-k3-hh-dense-mla-dcp8-latest \
  -e SERVED_MODEL_NAME=Kimi-K3-MXFP4-HH-DenseMLA-DCP8-1M \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  kimi-k3-hh bash -lc '
exec /mnt/luke/vllm-k3-hh-dense-mla-dcp8/serve-kimi-k3-full-mxfp4-dcp8-1m.sh \
  >/mnt/luke/k3-hh-full-mxfp4-dcp8-dense-mla.log 2>&1
'
```

The launcher fixes or validates all critical settings, including the four
memory-saving projection shards, native B12X MoE, native dense MLA, SparkInfer
DCP A2A, size-1 graph capture, and InstantTensor. Healthy markers are:

```text
SparkInfer dense MLA preflight: .../sparkinfer/attention/dense_mla/__init__.py
HH Kimi-K3 fused cache-op preflight: OK
HH Kimi-K3 fused KDA decode preflight: OK
HH Kimi-K3 fused SiTU activation preflight: OK
Model loading took 90.93 GiB and about 189 seconds
Using B12X PCIe DCP collectives (world_size=8 ...)
B12X dense K3 MLA plan: ... splits=94
GPU KV cache size: 1,054,602 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.01x
Application startup complete
```

Verify:

```bash
server_ip=$(docker inspect kimi-k3-hh --format \
  '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
curl -s "http://${server_ip}:8000/v1/models" | jq \
  '.data[0] | {id, max_model_len}'
curl -f "http://${server_ip}:8000/health"
```

## 64k prefill capacity gate

This sends exact token IDs, requests one streamed output token, and measures
TTFT. Prefix caching must remain disabled for repeated measurements:

```bash
tools=models/kimi-k3/tools
python "$tools/kimi-k3-prefill-stream.py" \
  --url "http://${server_ip}:8000" \
  --model Kimi-K3-MXFP4-HH-DenseMLA-DCP8-1M \
  --token-file "$tools/decode-baseline-256-token-ids.json" \
  --sizes 65536 \
  --warmups 0 \
  --runs 1 \
  --output /tmp/kimi-k3-prefill-64k.json
```

Require `usage_prompt_tokens=65536`, one completion token, HTTP success, and
no CUDA/OOM/nonfinite/traceback entry in the server log.

## Decode gate

Run one short smoke request followed by three forced 1,024-token requests.
The checked-in token file makes the 256-token prefill identical between
checkpoints. The helper measures only first-to-last streamed-token time, so
TTFT and prefill are excluded:

```bash
tools=models/kimi-k3/tools
out=/tmp/kimi-k3-hh-decode
mkdir -p "$out"

python "$tools/kimi-k3-decode-stream.py" \
  --url "http://${server_ip}:8000" \
  --model Kimi-K3-MXFP4-HH-DenseMLA-DCP8-1M \
  --token-file "$tools/decode-baseline-256-token-ids.json" \
  --prompt-tokens 256 \
  --max-tokens 256 \
  --output "$out/smoke.txt" \
  --metrics "$out/smoke.json"

for run_id in 1 2 3; do
  python "$tools/kimi-k3-decode-stream.py" \
    --url "http://${server_ip}:8000" \
    --model Kimi-K3-MXFP4-HH-DenseMLA-DCP8-1M \
    --token-file "$tools/decode-baseline-256-token-ids.json" \
    --prompt-tokens 256 \
    --max-tokens 1024 \
    --output "$out/run${run_id}.txt" \
    --metrics "$out/run${run_id}.json"
done
```

Require coherent text, no nonfinite/CUDA error in the server log, and median
decode throughput no lower than 34.9848 tok/s. The comparable HH target is
38.1146 tok/s or better without clock locking. The final clock-profile gate
passed at 41.1667 tok/s median with the host otherwise idle.

## Profiling and rejected tuning knobs

The reproducible model-free tools are in the SparkInfer branch:

```text
benchmarks/benchmark_kimi_k3_dense_mla_splits.py
benchmarks/reproduce_kimi_k3_tp16_direct_lifecycle.py
benchmarks/benchmark_pcie_hierarchical_tp12.py --world-size 16
```

The dense-MLA sweep proved that the 94-split plan is the correct long-context
choice. At a global 1M context it took 0.118 ms/layer, versus 0.154 ms for 64
splits, 0.195 ms for 32, and 0.190 ms for 16. One to sixteen splits save only
about 4 microseconds/layer at very short context and become much worse as the
context grows, so the production plan remains 94.

The native M=1 MXFP4 MoE graph measured about 0.0246 ms/layer and replayed
byte-exactly. The TP16 hierarchical all-reduce sweep confirmed the existing
geometry is optimal: 16 blocks for 3,584 BF16 elements (16.50 microseconds) and
32 blocks for 7,168 elements (18.53 microseconds), about twice as fast as NCCL.

A 4-layer Torch/CUDA profile is retained at:

```text
/mnt/luke/k3-hh-4l-torch-profile-20260803/
```

It identified the three TP all-reduces per transformer layer as the main
remaining decode cost. Dense MLA and MoE are no longer the dominant kernels.
Disabling the shared-expert auxiliary stream reduced the clock-locked 4-layer
median from 539.649 to 527.899 tok/s, so production keeps the stream enabled.
Nsight GPU performance counters are unavailable to this container/host due to
`ERR_NVGPUCTRPERM`; Torch profiler, CUDA events, and `nvidia-smi dmon` were used
instead.

Future optimization should target GEMV/TP16 collective overlap or a true
hierarchical fused GEMV+all-reduce path. It should not reduce dense-MLA splits,
switch back to Triton MLA, disable the shared-expert stream, or replace the
validated custom all-reduce with NCCL.

## Rollback

If the HH stack regresses:

1. terminate only the active vLLM API process with `SIGTERM`;
2. reset clocks if the speed profile was applied;
3. restore the image, source branches, and command from the
   [pre-HH checkpoint page](stable-full-mxfp4-dcp8-checkpoint-2026-08-03.md);
4. require its 1,059,851-token cache marker and three-run decode gate.

Do not mix the rollback vLLM checkout with this SparkInfer branch or vice
versa. The two checkpoints are complete pairs.
