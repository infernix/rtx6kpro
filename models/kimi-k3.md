# Kimi K3 TP16 MXFP4/NF3 on 16x RTX PRO 6000 Blackwell

Status: working and output-validated on 2026-07-29. This page is the handoff
for the Kimi K3 quantization, vLLM/SparkInfer integration, InstantTensor loader,
CUDA-clean forkserver startup, DCP1 decode optimization, and the separate
TP16/DCP16 one-million-token capacity profile.

No TP8/PP2 fallback is used anywhere in this runbook. All configurations use
Tensor Parallelism (TP) 16.

## Result At A Glance

| Item | Validated result |
|---|---|
| Model | `moonshotai/Kimi-K3` revision `2496450e92e425c886db095102a52a6682ca3970` |
| Quant | 65,945 routed experts unchanged in source MXFP4, 16,487 in stock NF3 |
| Routed-expert size | 4.049992721 bpw, 1,378,385,068,032 bytes, 1,283.72 GiB |
| Serving layout | TP16, one process per GPU, no PP |
| DCP1 latency profile | 32,768 max length, 34,105 physical FP8 KV tokens, 512,000,000 KV bytes/GPU |
| DCP1 pure decode | 36.801049 / 36.796794 / 36.793361 tok/s; mean 36.797068 tok/s |
| DCP1 time per token | 27.18 ms/token |
| DCP16 capacity profile | exactly 1,048,576 physical FP8 KV tokens, max length 1,048,576 |
| Weight load, final restart | InstantTensor 168.60 s; full model load/postprocess 212.22 s |
| Graph | PIECEWISE decode graph, 2 s capture, 0.29 GiB/GPU |
| Startup process method | CUDA-clean `forkserver`; worker startup phase 167 s -> 39 s |
| Live local service at handoff | container `kimi-k3-dcp1-speed`, `http://127.0.0.1:5670` |

The current DCP1 service is a latency profile. The earlier DCP16 run is the
validated 1M-context capacity profile. DCP greater than one has not yet been
decode-speed tuned with the final graph path.

## Published Source Handoff

No pull requests were opened and no default branch was modified. Development
continues on these branches:

| Component | Branch / commit | Purpose |
|---|---|---|
| vLLM, exact tested r9 tree | [`build/kimi-k3-r9-dcp1-20260729` at `e8154512af`](https://github.com/local-inference-lab/vllm/tree/e8154512afea7ab2d196af12f049d6fbf0b4a77a) | Exact Gilded Gnosis r9 reconstruction used by the validated image |
| vLLM, current K3 development | [`dev/gg-k3` at `2d5fcd100a`](https://github.com/local-inference-lab/vllm/tree/2d5fcd100af4c676e29bdc33d7fdc6d030594b30) | Same K3 loader/runtime/fork improvements ported as a fast-forward onto current `dev/gg-k3` |
| SparkInfer | [`dev/gg-k3` at `64a9097`](https://github.com/local-inference-lab/sparkinfer/tree/64a90970621c3da8a28b84e8ef5f06d7d4260de3) | Exact r9 integration plus K3 hybrid decode kernels and benchmark harnesses |
| kquant | [`dev/gg-k3-4p05` at `2f9cb74`](https://github.com/local-inference-lab/kquant/tree/2f9cb7464345588a6f7960245dacbeb654de1817) | Revision-provenance fixes and the exact 4.05-bpw allocation/verification records |
| InstantTensor | [`dev/gg-k3-consumer-event` at `b0f9534`](https://github.com/voipmonitor/InstantTensor/tree/b0f9534db44b7c11eac171519af38450eea90d77) | Consumer-stream CUDA-event patch on official v0.1.9 commit `f5a445e` |

The exact-tested vLLM branch and the `dev/gg-k3` port are intentionally both
kept. The production image came from the r9 reconstruction, while the port is
the correct place for Luke and other developers to continue the K3 branch.

## Why The Physical Checkpoint Is 2,737.39 GiB

The composed directory is a zero-copy view over two complete physical stores:

1. the original Kimi K3 checkpoint, which still contains every source MXFP4
   expert and all non-expert tensors;
2. the 4.05-bpw expert overlay, which contains another copy of every selected
   expert tensor, either unchanged MXFP4 or converted NF3.

The filesystem therefore contains 991,812 physical tensors occupying
2,737.386 GiB. The composed safetensors index selects only 497,220 tensors and
1,390.268 GiB:

| Selected part | Bytes |
|---|---:|
| Routed-expert overlay | 1,378,385,068,032 |
| Original non-expert tensors | 114,404,258,816 |
| Total selected by the composed index | 1,492,789,326,848 |

vLLM does not load 2.737 TiB. The index-aware loader reads the selected
1.390 TiB logical model. The large physical total is the cost of retaining the
base checkpoint plus a non-destructive expert overlay; it is not the per-run
weight footprint.

The source MXFP4 experts are not repacked to W4A8 storage. Kept experts remain
bitwise-identical MXFP4 with 4-bit E2M1 values and E8M0 K32 scales. The runtime
uses BF16 activations through the SparkInfer W4A16 interface. Consequently the
claimed 33.33% W4A8 storage expansion does not apply to this artifact.

## Quantization Decision

The first kquant prototype targeted 3.20 bpw with MXFP4, NF3, and NF2. It could
not be used as emitted because NF2 was accounted as 2.25 bpw but the writer
still physically used its 3-bit packing path. The runtime also had no matching
NF2 kernel. The selected production plan is therefore only two tiers:

- retain as many experts as possible in original MXFP4;
- demote the remaining experts to the existing stock NF3 format;
- no NF2, no refitted codebook, and no dense W4A8 repack.

The exact allocation keeps 65,945 of 82,432 routed experts in MXFP4 (80.00%)
and uses NF3 for 16,487 (20.00%). Allocation is layer-sensitive, not a uniform
20% demotion per layer.

`kquant verify --samples 256` produced:

| Check | Result |
|---|---:|
| Kept MXFP4 samples bitwise exact | 203 / 203 |
| NF3 samples deterministically reproduced | 53 / 53 |
| NF3 mean MSE | 2.5136177e-5 |
| NF3 maximum MSE | 2.7672388e-5 |
| Byte accounting | exact |

All 92 MoE layers were packed into 321 safetensors shards in 1,165.2 s. A
separate structural pass checked 494,592 overlay tensors, every header extent,
and every index mapping.

## Reproduce The Quant And Composite Checkpoint

This step needs the complete original checkpoint and about 1.29 TiB additional
free space for the expert overlay. It deliberately preserves the source.

```bash
git clone --branch dev/gg-k3-4p05 \
  https://github.com/local-inference-lab/kquant.git /root/vllm/kimi/kquant-k3
cd /root/vllm/kimi/kquant-k3

REV=2496450e92e425c886db095102a52a6682ca3970
HF_REPO=/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3
OUT=/root/vllm/kimi/kquant-k3/out-tp16-4p05
OVERLAY=/root/vllm/kimi/Kimi-K3-MXFP4-NF3-4p05

python3 -m kquant.cli \
  --out "$OUT" --cache-dir "$HF_REPO" --revision "$REV" \
  rank --formats nf3 --target-bpw 4.05 \
  --traffic bias --traffic-alpha 1.0 --w2 raw --min-keep 0

python3 -m kquant.cli \
  --out "$OUT" --cache-dir "$HF_REPO" --revision "$REV" \
  pack --device cuda --batch-size 32 --dest "$OVERLAY"

install -m 0644 "$OUT/allocation.json" "$OVERLAY/allocation.json"

python3 -m kquant.cli \
  --out "$OUT" --cache-dir "$HF_REPO" --revision "$REV" \
  verify --dest "$OVERLAY" --samples 256
```

The branch contains the exact `allocation.json`, its stock statistics bundle
links, and `verify.json`. Re-running `rank` should reproduce the assignment;
use the committed allocation directly if the goal is only to resume packing.

Compose a standard Hugging Face view without copying shards:

```bash
BASE="$HF_REPO/snapshots/$REV"
COMPOSITE=/root/vllm/kimi/Kimi-K3-MXFP4-NF3-4p05-hf

python3 /root/rtx6kpro/scripts/compose-kimi-k3-kquant-checkpoint.py \
  --base "$BASE" \
  --overlay "$OVERLAY" \
  --destination "$COMPOSITE"
```

The composer removes original expert keys from the base index, adds overlay
keys, writes the hybrid bit map into both quantization configs, and symlinks
the selected shards. Do not build the composite by concatenating two indexes;
stale base expert keys otherwise leak into NF3 parameters.

## Runtime Changes

### vLLM

The main integration changes are:

- consume kquant's mixed MXFP4/NF3 tensor names and per-layer expert map;
- stream NF3 repacking and release temporary warmup allocations;
- enforce the exact `tensor -> absolute shard` mapping for both safetensors and
  InstantTensor so a stale base copy cannot override an overlay tensor;
- shard routed latent up/down projections and router work across TP16 rather
  than retaining the initial replicated implementation;
- handle K3's physically padded `A_log[128]` as 96 logical heads, then select
  the local six-head TP16 slice;
- keep embeddings and `lm_head` outside the hybrid linear quant method;
- use `-1` as the inactive route sentinel and zero inactive tier outputs;
- mark KDA attention as an eager breakpoint during PIECEWISE graph capture;
- support an explicit CUDA-clean `forkserver` worker method.

The original K3 branch did replicate the routed latent projections/router.
That was a bring-up implementation, not an architectural necessity. Sharding
them removes the TP16 replication cost while keeping the expert-local layout.

### SparkInfer

K3 TP16 has local expert intermediate width 192. GLM's N256 tile is invalid
for this geometry. The working K3 configurations use FC1 K128/N64 and FC2
K64/N128. The SparkInfer branch also contains:

- K3 SiTU activation in the micro path;
- correct native E8M0 K32 scale decode for MXFP4;
- a direct M=1 MXFP4 path instead of correctness-padding every request to M=9;
- one-grid hybrid MXFP4/NF3 execution to avoid two tier-launch sequences;
- safe inactive-route handling;
- standalone real-shape and prewarm benchmark harnesses.

The isolated real-shape kernel matched the FP32 oracle at cosine 0.999997 and
measured 0.0852 ms for native M=1 versus 0.4339 ms for padded M=9, a 5.09x
kernel-level improvement.

## InstantTensor Patch

Official InstantTensor v0.1.9 synchronized and cloned each yielded tensor in
Python. K3 exposes roughly 497,000 selected tensors, so the per-tensor path
dominated startup even though storage bandwidth was sufficient.

The published patch changes the contract as follows:

- vLLM passes its consumer CUDA stream and requests `copy=False`;
- InstantTensor records a persistent CUDA event on that consumer stream;
- AIO and BUFFERED producer streams wait for the event before reusing the
  zero-copy ring buffer;
- CUFILE retains a conservative CPU event wait;
- vLLM filters the composed index before InstantTensor opens I/O and coalesces
  contiguous selected byte ranges.

Source and directly applicable diff:
[`voipmonitor/InstantTensor@b0f9534`](https://github.com/voipmonitor/InstantTensor/commit/b0f9534db44b7c11eac171519af38450eea90d77).

The validated wheel was:

```text
instanttensor-0.1.9-cp312-cp312-linux_x86_64.whl
sha256 5ad0561a85e1ddc4e5c9efc186cdbe6c106c844f2d185a8257d783bd89f6ca0e
```

Build it from the published branch:

```bash
git clone --branch dev/gg-k3-consumer-event \
  https://github.com/voipmonitor/InstantTensor.git \
  /root/vllm/worktrees/instanttensor-k3
cd /root/vllm/worktrees/instanttensor-k3
./checkout_submodules.sh
python3.12 -m pip wheel --no-deps -w wheelhouse .
sha256sum wheelhouse/instanttensor-0.1.9-cp312-cp312-linux_x86_64.whl
```

Exact validation included a 4.298-GB real shard, forcing the 4-GiB ring to
wrap, then comparing retained first/middle/last destination tensors bitwise
against safetensors. The TP16 BUFFERED loader harness reached 23.3046 GB/s for
68.7608 GB logical input. Full weight-load measurements ranged from 125.31 to
173.55 s; the production restart took 168.60 s versus 377.19 s with the prior
safetensors path.

This is a large improvement, but not the end state. The remaining bottleneck
is Python/model assignment of 497,220 small tensors; it is why this K3 load is
still much slower than a GLM checkpoint with far fewer tensors.

Run the included shard harness with:

```bash
python3 scripts/benchmark-instanttensor-shard.py \
  --verify /path/to/one/real/model-shard.safetensors
```

## CUDA-Clean Forkserver

Plain `fork` is not usable: the parent carries CUDA runtime state and children
fail during `_cuda_init()` with a poison-fork error. Plain `spawn` is safe, but
on TP16 every rank repeatedly imported the heavy vLLM worker stack and ranks
appeared at roughly ten-second intervals.

The initial GG forkserver support was incomplete in three ways:

1. `vllm/envs.py` rejected `forkserver` as an allowed value;
2. the API forkserver preloaded only the EngineCore path;
3. EngineCore created a second worker forkserver with preload `['__main__']`.

The final implementation accepts explicit `forkserver`, preloads executor and
GPU worker modules in both process layers, and moves `kernel_warmup` behind a
lazy import. Importing `multiproc_executor` and `gpu_worker` therefore leaves
`torch.cuda.is_initialized() == False`.

| Startup segment | Before | After |
|---|---:|---:|
| EngineCore init -> start loading weights | 167 s (`spawn`) | 39 s (`forkserver`) |
| EngineCore init -> API ready | 407 s | 271 s |

All 16 ranks entered distributed initialization within one second in the final
run. Keep these safety rules:

- never change this launcher to ordinary `fork`;
- every module added to forkserver preload must remain CUDA-clean on import;
- if ranks return to ten-second spacing, inspect the second worker forkserver;
- if `Cannot re-initialize CUDA in forked subprocess` appears, bisect preload
  imports for a new module-level CUDA initialization.

## CUDA Graph Correctness

The apparent approximately 38 tok/s monolithic FULL decode graph was invalid:
output text was corrupt. A first PIECEWISE attempt was also corrupt because 69
stateful KDA layers were captured through Python metadata and recurrent-cache
mutation, even though isolated raw KDA kernels were graph-exact.

The valid solution marks both MLA and KDA attention boundaries eager during
capture while leaving projections and MoE in graph segments. Output was
checked at 64, 96, and 512 generated tokens, including a coherent Euclid proof,
before performance was accepted.

At 952,000,000 KV bytes/GPU, graph capture OOMed because an additional 192-MiB
MLA workspace had only 169 MiB free. The DCP1 speed profile therefore uses
512,000,000 bytes and max length 32,768. It yields 34,105 physical KV tokens,
leaves room for the 0.29-GiB graph pool, and captures in two seconds.

## Build The Exact Tested Image

The following sequence starts from the requested r9 image and the published
exact-tested source pins:

```bash
git clone --branch build/kimi-k3-r9-dcp1-20260729 \
  https://github.com/local-inference-lab/vllm.git \
  /root/vllm/worktrees/vllm-k3-r9

git clone --branch dev/gg-k3 \
  https://github.com/local-inference-lab/sparkinfer.git \
  /root/vllm/worktrees/sparkinfer-k3-r9

cd /root/vllm/worktrees/vllm-k3-r9

docker build \
  -f /root/rtx6kpro/docker/kimi-k3/Dockerfile.r9-base \
  -t local/vllm:kimi-k3-kquant-4p05-tp16-20260729 .

docker build \
  -f Dockerfile.kimi-k3-kv-allocator \
  -t local/vllm:kimi-k3-kquant-4p05-tp16-production-20260729 .

docker build \
  --build-context sparkinfer=/root/vllm/worktrees/sparkinfer-k3-r9 \
  --build-context instanttensor-wheel=/root/vllm/worktrees/instanttensor-k3/wheelhouse \
  -f Dockerfile.kimi-k3-dcp1-speed \
  -t local/vllm:kimi-k3-kquant-dcp1-speed-20260729 .
```

The validated final local image ID was:

```text
sha256:0d947fd4f9207effab5feca6c07a366ff3c46f72eff42ca4ee81393345488a91
```

The first rebuild after changing SparkInfer may compile the CuTe kernels. Keep
the persistent JIT cache mounted; deleting it makes iteration unnecessarily
slow. A different InstantTensor build can legitimately change the final image
ID, so verify the source commits and wheel SHA as well as the tag.

## Install The Launch Artifacts

The launchers expect the model and cache under `/root/vllm/kimi`:

```bash
install -m 0755 scripts/run-kimi-k3-kquant-tp16-dcp.sh /root/vllm/kimi/
install -m 0755 scripts/run-kimi-k3-dcp1-speed.sh /root/vllm/kimi/
install -m 0755 scripts/validate-kimi-k3-dcp.sh /root/vllm/kimi/
install -m 0755 scripts/benchmark-kimi-k3-pure-decode.py /root/vllm/kimi/
install -m 0644 scripts/kimi-k3-chat-template.jinja /root/vllm/kimi/
```

The custom K3 tokenizer ignores the placeholder Jinja content and applies its
own segment-aware XTML chat encoding. The non-empty file is still required by
vLLM's chat-template resolution guard.

## Launch And Validate DCP1 Decode

```bash
/root/vllm/kimi/run-kimi-k3-dcp1-speed.sh
docker logs -f kimi-k3-dcp1-speed

KIMI_EXPECT_MAX_MODEL_LEN=32768 \
  /root/vllm/kimi/validate-kimi-k3-dcp.sh

python3 /root/vllm/kimi/benchmark-kimi-k3-pure-decode.py \
  --url http://127.0.0.1:5670 \
  --model Kimi-K3-MXFP4-NF3-4p05 \
  --max-tokens 96 --warmups 1 --runs 3
```

Important DCP1 settings are TP16/DCP1, `max_num_seqs=1`, maximum 32 batched
tokens, FP8 KV, BF16 Mamba cache, prefix caching disabled, InstantTensor
BUFFERED, PIECEWISE graph at capture size one, and forkserver workers.

Pure decode is measured as:

```text
(completion_tokens - 1) / (last text-token timestamp - first text-token timestamp)
```

This excludes request queueing, tokenization, prefill, and time to first token.
Do not report `completion_tokens / request wall time` as decode speed.

| Configuration | Pure decode tok/s | ms/token |
|---|---:|---:|
| Original correct path | 4.844763 | 206.41 |
| Tuned eager path | 6.315500 | 158.34 |
| Final output-valid graph path | 36.797068 | 27.18 |

The final result is 7.595x / +659.52% over the original correct path and
5.826x / +482.65% over tuned eager.

## Reproduce The TP16/DCP16 1M Capacity Profile

The exact capacity run used the production image before the final DCP1 graph
overlay. It is a correctness/capacity proof, not a tuned DCP16 decode result:

```bash
KIMI_IMAGE=local/vllm:kimi-k3-kquant-4p05-tp16-production-20260729 \
KIMI_CONTAINER=kimi-k3-kquant-tp16-dcp16-1m \
KIMI_DCP_SIZE=16 \
KIMI_MAX_MODEL_LEN=1048576 \
KIMI_KV_CACHE_BYTES=952000000 \
KIMI_LOAD_FORMAT=safetensors \
KIMI_EXECUTION_MODE=eager \
KIMI_WORKER_MULTIPROC_METHOD=spawn \
  /root/vllm/kimi/run-kimi-k3-kquant-tp16-dcp.sh

KIMI_EXPECT_MAX_MODEL_LEN=1048576 \
KIMI_PORT=5670 \
  /root/vllm/kimi/validate-kimi-k3-dcp.sh
```

The server reported exactly `GPU KV cache size: 1,048,576 tokens` and maximum
concurrency 1.00x. Model footprint was 91.65 GiB/rank; after cache allocation
`nvidia-smi` showed about 97,245 MiB used per GPU. Maximum batched tokens 32 is
required: an arbitrary second prefill shape previously triggered CUDA error
999 in the SparkInfer MXFP4 path. Four sequential correctness requests passed
after enforcing 32-token chunks.

Do not enable the DCP1 graph settings with the 952-MB DCP16 memory ledger
without re-profiling. DCP16 shares the context state and fits 1M; DCP1 instead
uses its freed context memory for the valid low-latency graph.

## Validation Completed

- kquant: 14 tests passed; allocation provenance and byte accounting checked;
- exact image loader/hybrid tests: 14 passed;
- breakable graph tests: 20 passed;
- current `dev/gg-k3` port: 12 passed, 2 GPU-only tests skipped in CPU CI;
- Python compile, shell syntax, patch apply, and `git diff --check` passed;
- InstantTensor real ring-wrap comparison passed bitwise;
- DCP1 responses were deterministic and coherent at 64, 96, and 512 tokens;
- live restart loaded weights in 168.60 s, `/health` returned 200, and warmed
  non-thinking `1+1` returned exactly `2`.

## Remaining Work

1. Reduce the 497,220-tensor Python/model assignment overhead after
   InstantTensor I/O. The storage path is no longer the dominant limitation.
2. Tune DCP greater than one only after preserving the DCP1 correctness and
   pure-decode measurement method.
3. Run longer quality evaluation for the layer-sensitive NF3 allocation. The
   current checks prove packing determinism and runtime correctness, not full
   model-quality equivalence.
4. Revalidate full E2E serving from the forward-ported `dev/gg-k3` image; the
   published exact-tested r9 branch remains the release reference meanwhile.

Do not accept a speed number from a graph configuration unless generated text
has also been inspected. The failed graph variants were fast and wrong.
