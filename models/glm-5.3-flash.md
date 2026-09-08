# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke in the
<a href="https://discord.com/channels/1466898002793857221/1476263308242714718/1543077243398393927">community Discord</a>.</em></p>

This page is the stable deployment and performance reference for
GLM-5.3-Flash on RTX PRO 6000 Blackwell. The qualified serving artifact is
Jovian Judgement Community — 20260831-r8.

Terminology used below:

- The vLLM inference engine `vLLM` integrates the B12X kernel/backend stack
  `B12X`.
- 4-bit floating-point `FP4`, NVIDIA FP4 `NVFP4`, 8-bit floating-point `FP8`,
  and Microscaling FP8 `MXFP8` name numeric formats.
- Multi-Token Prediction `MTP`, Decode Context Parallelism `DCP`, Multi-head Latent Attention `MLA`,
  Mixture of Experts `MoE`, and Time To First Token `TTFT` name model or serving
  operations.
- Compute Unified Device Architecture `CUDA`, NVIDIA Collective Communications Library `NCCL`,
  Peripheral Component Interconnect Express `PCIe`, and Streaming Multiprocessor 120 `SM120`
  name the runtime and hardware interfaces.

The runbook serves `local-inference-lab/GLM-5.3-Flash-NVFP4` on four NVIDIA
RTX PRO 6000 Blackwell GPUs. The Jovian Judgement Community image
supports ordinary decode, three-token Multi-Token Prediction (MTP), and a
seven-token DFlash2 draft loaded from
`local-inference-lab/GLM-5.3-Flash-DFlash2`.

The commands use Hugging Face model names and named Docker volumes. They do not
require local checkpoint paths or source-code bind mounts.

## Status

| Field | Value |
|---|---|
| Runtime status | **qualified** for Tensor Parallelism 4 (TP4) with Decode Context Parallelism 1 (DCP1) in all three serving modes |
| Additional qualification | **qualified** for TP4/DCP4 DFlash2 prefill with full compressed-key/value (CKV) gathering |
| TP3 candidate overlay | **research-only validated** 2026-09-08: TP3/EP3/DCP1 chain reconstructed on the R27 sources, six serving-configurations passed; receipts bound to `infernix/vllm@sha256:1871c461…`. See [TP3 three-GPU candidate](#tp3-three-gpu-candidate-research-overlay-2026-09-08). |
| Hardware | four RTX PRO 6000 Blackwell Workstation Edition GPUs, PCIe 5.0 x16, stock clocks |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4` |
| Target update policy | resolve the Hugging Face `main` branch at startup; no runtime revision pin |
| Target routed experts | ModelOpt NVFP4, B12X 4-bit-weight/4-bit-activation (W4A4) |
| Target key-value cache | FP8 compressed Multi-head Latent Attention (MLA) |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2` |
| DFlash2 update policy | resolve the Hugging Face `main` branch at startup; no runtime revision pin |
| DFlash2 weights | pre-serialized ModelOpt MXFP8; no online weight quantization |
| Cache page geometry | independent 512-token target and recurrent-state pages |
| Scheduler limit | `MAX_NUM_BATCHED_TOKENS=4096` |
| Concurrent prefill cadence | `PREFILL_SCHEDULE_INTERVAL=8`; active only while eligible decode work is running |
| CUDA graphs | target and speculative decode are captured; Gated Delta Network (GDN) prefill is eager |
| Qualification date | 2026-08-31 |
| Canonical DFlash2 locator validation | **qualified** on 2026-08-31 with identical MXFP8 weights |

## Docker artifact

Use the digest for byte-identical runtime layers:

```text
voipmonitor/vllm:jovian-judgement-community-20260831-r8
voipmonitor/vllm@sha256:827a64ce0cea267aad843b3d521a47d742a6e78b502eaec7c05b4ae8bf403194
```

The embedded source-lock SHA-256 is
`5ca5c81dff7dcf1b864dbb11f230df9f6276ca46e88c3b2a81ebec15e09dd21a`.
Both checkpoint locators follow their Hugging Face `main` branches. The image
digest therefore fixes the runtime but intentionally does not pin mutable model
revisions.

## Source contract

The vLLM package tree is
`4daf199faeab2208c975433973adf5cefce6d2ce`. It is composed from
`local-inference-lab/vllm` branch `dev/jovian-judgement` at
`0b67266a0f37d6146a8403fb8482403c62f412d5` and the following non-draft pull
request heads:

| Pull request | Resulting behavior |
|---|---|
| [vLLM #515](https://github.com/local-inference-lab/vllm/pull/515) `3bcb90163d9f` | Retains CUDA-graph profiling resources until teardown. |
| [vLLM #516](https://github.com/local-inference-lab/vllm/pull/516) `db0f14444e8c` | Makes the B12X profiling warmup lifetime-safe. |
| [vLLM #517](https://github.com/local-inference-lab/vllm/pull/517) `92d807af333b` | Gathers full C4 CKV for DCP prefill. |
| [vLLM #530](https://github.com/local-inference-lab/vllm/pull/530) `3609a3db4986` | Uses FlashKDA recurrent checkpoints for GLM-5.3 prefill. |
| [vLLM #531](https://github.com/local-inference-lab/vllm/pull/531) `b8edca554d21` | Reuses immutable B12X C4 indexer plans. |
| [vLLM #532](https://github.com/local-inference-lab/vllm/pull/532) `71054201ae23` | Parallelizes C4 pool writes and bounds visible pages. |
| [vLLM #533](https://github.com/local-inference-lab/vllm/pull/533) `d6ace9116f9c` | Preserves replicated DFlash2 cache geometry under DCP. |
| [vLLM #535](https://github.com/local-inference-lab/vllm/pull/535) `5f8e00d6c33a` | Separates target and recurrent cache pages at 512 tokens. |
| [vLLM #536](https://github.com/local-inference-lab/vllm/pull/536) `9f27029f55fd` | Checkpoints target GDN state for MTP prefill. |
| [vLLM #537](https://github.com/local-inference-lab/vllm/pull/537) `236032509531` | Compacts MTP prefill outputs before Mixture-of-Experts (MoE) compute. |
| [vLLM #539](https://github.com/local-inference-lab/vllm/pull/539) `d59cea8e55f6` | Preserves the embedding of a valid MTP token at position zero. |
| [vLLM #546](https://github.com/local-inference-lab/vllm/pull/546) `2412a6f34ab0` | Applies the configured prefill cadence to non-data-parallel engines, validates the scheduler step-counter contract, and avoids empty pipeline-parallel steps when decode is temporarily ineligible. |

The reproducible vLLM composition commit is
`voipmonitor/vllm@f311938fab3e7ffccfc0713d5518025c3c811725` with tree
`f5e05de47a9e9ff77c73f623347675a11b7a0d02`. Pull request #513 is not part of
this package tree.

The B12X package tree is
`6de9871d15dab093340695518fec0f744289e676`. It is composed from B12X
`master@fc1d4b68f7a5b0cfdb88bf06abccd869f5c589d5` and two non-draft
performance pull requests:

| Pull request | Scope | Qualified kernel result |
|---|---|---:|
| [B12X #259](https://github.com/local-inference-lab/b12x/pull/259) `e5509cc95b8f` | Selects a wider SM120 TF32 multi-head-connection projection for hidden size 4096 and 2,304–3,583 prefill rows. This operation is shared by all three target-serving modes. | 31.67% projection-throughput increase at 3,072 rows |
| [B12X #260](https://github.com/local-inference-lab/b12x/pull/260) `b68c197262d5` | Increases device-bounded shared candidate capacity and omits unused score output in the paged C4 top-k selector. This operation is shared by all three target-prefill modes. | 2.01% selector-throughput increase at 4,080 query rows |

Those percentages qualify the named fixed-work kernels, not end-to-end server
throughput. The reproducible B12X composition commit is
`voipmonitor/b12x@6255090a03b12c3f7d552102a02fac0b542fb8c9` with tree
`0bb58d0dcc10e29e00ff9850c0d719fca1aba5ad`.

## Runtime backends

| Operation | Selected implementation |
|---|---|
| Target sparse MLA attention and C4 indexer | B12X |
| Target GDN prefill | FlashKDA recurrent checkpoints |
| Target GDN decode | B12X live-tensor KDA when eligible; automatic resolver retains the Triton fallback |
| Target routed experts | B12X NVFP4 W4A4 |
| Target linear layers | B12X |
| Tensor-parallel all-reduce | B12X PCIe first; PyNCCL outside the B12X dispatch range |
| MTP attention | B12X |
| MTP MXFP8 experts | Humming |
| DFlash2 MXFP8 linear layers | `B12xMxfp8LinearKernel` |
| DFlash2 fused context key/value projection | `B12xMxfp8LinearKernel` |
| DFlash2 local attention | FlashAttention 2 |
| Sampling | FlashInfer |

DeepGEMM and TileLang are installed dependencies but are not selected for the
target, MTP, or DFlash2 hot paths in this serving contract.

`CUDAGRAPH_MODE=FULL` is requested. The GLM GDN backend supports uniform-batch
decode capture but not full prefill capture, so vLLM resolves target execution
to `FULL_DECODE_ONLY`. MTP and DFlash2 decode graphs are captured. Prefill is
eager.

## Start one of the three DCP1 modes

Set the immutable image, the four desired physical GPUs, and a unique container
name. The qualification host used GPUs 4, 5, 6, and 7; `0,1,2,3` below is a
portable four-GPU example.

```bash
IMAGE=voipmonitor/vllm@sha256:827a64ce0cea267aad843b3d521a47d742a6e78b502eaec7c05b4ae8bf403194
GPU_DEVICES=0,1,2,3
```

Select exactly one mode:

```bash
# Ordinary decode without speculative tokens.
NAME=jovian-judgement-nomtp-dcp1
MODE_ARGS=(-e SPECULATOR=mtp -e MTP=0)
```

```bash
# Three-token built-in MTP.
NAME=jovian-judgement-mtp3-dcp1
MODE_ARGS=(-e SPECULATOR=mtp -e MTP=3)
```

```bash
# DFlash2 with its trained default of seven draft tokens.
NAME=jovian-judgement-dflash2-dcp1
MODE_ARGS=(
  -e SPECULATOR=dflash2
  -e NUM_SPECULATIVE_TOKENS=7
  -e DFLASH_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2
  -e DFLASH_MODEL_REVISION=
)
```

Run the selected mode:

```bash
docker rm -f "$NAME" 2>/dev/null || true

docker run -d \
  --name "$NAME" \
  --init \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v jovian-judgement-vllm-cache:/cache \
  -v jovian-judgement-huggingface-cache:/root/.cache/huggingface \
  -e PORT=5001 \
  -e TP=4 \
  -e DCP=1 \
  -e MAX_MODEL_LEN=262144 \
  -e MAX_NUM_SEQS=16 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e PREFILL_SCHEDULE_INTERVAL=8 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=128 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -e B12X_PCIE_ALLREDUCE=1 \
  -e NCCL_MIN_NCHANNELS=32 \
  -e NCCL_MAX_NCHANNELS=32 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e OMP_NUM_THREADS=2 \
  "${MODE_ARGS[@]}" \
  "$IMAGE"
```

For DCP4 full-CKV target prefill in any of the three serving modes, replace
`-e DCP=1` with:

```bash
  -e DCP=4 \
  -e DCP_CKV_GATHER=1 \
```

The launcher enables full-CKV gathering automatically when `DCP` is greater
than one; the explicit `DCP_CKV_GATHER=1` documents the selected behavior.
No-spec, MTP, and DFlash2 share this target-prefill mechanism. The DCP4
performance result below qualifies the DFlash2 mode specifically.

## Verify startup

```bash
curl -fsS http://127.0.0.1:5001/health

docker logs "$NAME" 2>&1 | grep -E \
  'speculative_config|B12X PCIe|B12xMxfp8|HUMMING|FlashAttention version 2|split GLM-5.3 cache pages|GPU KV cache size|Graph capturing finished|Application startup complete'

curl -fsS http://127.0.0.1:5001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-NVFP4","messages":[{"role":"user","content":"Reply with exactly READY."}],"temperature":0,"max_tokens":64}'
```

Expected common markers include B12X NVFP4 MoE, B12X PCIe all-reduce, and
512-token target plus recurrent pages. MTP adds `HUMMING` for its MXFP8
experts. DFlash2 adds `B12xMxfp8LinearKernel` and FlashAttention version 2.

## Measured performance

The DCP1 table was measured on physical GPUs 4–7 with PCIe 5.0 x16 links. Rows
use stock clocks unless their mode explicitly identifies the research-only
`VRAM +6000` profile. Every server used TP4, DCP1, FP8 target KV, 512/512 cache
pages, B12X PCIe all-reduce, 32 minimum and maximum NCCL channels, and
`MAX_NUM_BATCHED_TOKENS=4096`. These isolated benchmark cells were measured on
the same kernel/backend composition as the r8 image. Pull request #546 changes
only scheduler fairness while prefill and decode requests overlap; its cadence
path is inactive during isolated CC1 decode or standalone prefill.
The target repository's `main` branch has the same 49 runtime files as the
qualified target revision; only its model card differs. A 2026-08-31 smoke test
loaded the byte-identical MXFP8 draft weights through the canonical DFlash2
repository and completed speculative inference on GPUs 4–7.

The measured DFlash2 checkpoint contents have `model.safetensors` SHA-256
`c033e03d47c7d5608596c8fc4e9336a1fe086eb781c08fe031be2bdea1614e58`.
The community launcher follows both checkpoints' `main` branches, so a later
target or draft update requires fresh performance qualification and may not
reproduce the rows below.

The 32k prefill request contained exactly 32,320 supplied token IDs, used a
unique `cache_salt`, streamed one generated token, and measured client
time-to-first-token (TTFT). One unreported warmup preceded three samples; the
table reports their median. This is an end-to-end TTFT measurement. For
speculative modes it includes the work required to emit the first verified
token, not only isolated target-kernel time.

CC1 decode used `llm-decode-bench` 0.4.29, an empty-context decode cell,
`max_tokens=8192`, greedy sampling, a three-second warmup, and a 30-second
measurement. Three independent cells were measured and the table reports the
median. Accepted length and engine steps per second are server metrics.

| Mode | Speculative configuration | 32k TTFT | 32k prompt throughput | CC1 output | Sieve median | Engine rate | Accepted length |
|---|---|---:|---:|---:|---:|---:|---:|
| No speculative decode | none | 2.0786 s | **15,549 tok/s** | **139.4 tok/s** | not measured | — | — |
| MTP:3 | three built-in draft tokens | 2.1360 s | **15,131 tok/s** | **228.0 tok/s** | **287.87 tok/s** | 90.47 steps/s | 2.52 |
| DFlash2 MXFP8 | seven draft tokens | 2.1157 s | **15,276 tok/s** | **185.5 tok/s** | **339.73 tok/s** | 74.19 steps/s | 2.49 |
| DFlash2 MXFP8, VRAM +6000 (**research-only**) | seven draft tokens | 2.0508 s | **15,759.6 tok/s** | **200.87 tok/s** | not measured | 82.91 steps/s | 2.43 |

The `VRAM +6000` row changes only the memory-clock offset on GPUs 4–7. Stock
clocks remain the qualified deployment default; the
[research profile](#vram-6000-research-profile) records its raw samples and
clock state.

Raw CC1 samples were:

| Mode | Output tok/s samples | Engine steps/s samples | Accepted-length samples |
|---|---|---|---|
| No speculative decode | 139.53, 139.41, 139.38 | — | — |
| MTP:3 | 228.14, 226.01, 228.03 | 90.47, 90.47, 90.48 | 2.52, 2.50, 2.52 |
| DFlash2 MXFP8 | 192.94, 185.49, 178.23 | 74.18, 74.47, 74.19 | 2.60, 2.49, 2.40 |

DFlash2 raw output throughput varies with accepted length. Engine rate remained
within 0.4% across the three DFlash2 samples, so acceptance rather than target
execution speed explains the wider raw tok/s range.

### Sieve coding-prompt decode

The Sieve test is a prompt-specific, sequential CC1 measurement rather than a
sustained empty-context engine cell. It used `llm-decode-bench` 0.4.29 with the
prompt `Write a Python script that implements the Sieve of Eratosthenes.`, no
synthetic context, streaming output, the server/model temperature default, and
at most 2,000 generated tokens. Throughput is completion tokens divided by the
time from the first streamed token through the final stream event. Five
sequential samples ran on the exact image ID documented above, TP4/DCP1,
physical GPUs 4–7, and stock clocks.

| Mode | Generation tok/s samples | Median | Mean |
|---|---|---:|---:|
| MTP:3 | 292.02, 287.22, 290.74, 287.87, 278.03 | **287.87 tok/s** | 287.17 tok/s |
| DFlash2 MXFP8, seven drafts | 339.73, 332.98, 365.87, 325.04, 353.52 | **339.73 tok/s** | 343.43 tok/s |

DFlash2 was 18.01% faster by median on this prompt. This difference is not a
prompt-independent decode claim: Sieve output is highly predictable, and
speculative throughput changes with the accepted draft length. The sustained
CC1 table above remains the backend regression signal; the Sieve table captures
the interactive coding-prompt behavior users observe with `test.py`-style
clients.

The same image at TP4/DCP4 with DFlash2, seven draft tokens, and full-CKV
gather measured a 2.3803-second median TTFT, or **13,578 prompt tok/s**, for
the same 32,320-token request. DCP1 and DCP4 numbers are separate deployment
contracts and must not be compared as if only one kernel changed.

### Concurrent long-prefill fairness

The fairness qualification starts a TP4/DCP1 DFlash2 request with 4,094 prompt
tokens and 8,192 requested output tokens. After 2,048 output tokens, it submits
a second request with 65,535 prompt tokens and one requested output token. Both
requests use greedy sampling, seed zero, `ignore_eos=true`, a unique cache salt,
and stock clocks. The table reports the interval while the second request is
prefilling.

| Runtime | Prefill cadence | Decode target forwards | Decode throughput | Concurrent prefill throughput |
|---|---:|---:|---:|---:|
| r7 scheduler behavior | 1 | 34 | 43.2 tok/s | 14,166 prompt tok/s |
| r8 qualified artifact | 8 | 154 | **199.5 tok/s** | **10,752 prompt tok/s** |

The r8 request pair completed with exactly 8,192 decode output tokens and
65,535 prefill prompt tokens, DFlash accepted length 7.0, and no API error. A
standalone 65,535-token prefill on the r8 artifact measured **14,630 prompt
tok/s**. The cadence trades some mixed-load prefill throughput for decode
progress; it does not throttle prefill when no eligible decode request exists.

### VRAM +6000 research profile

Status: **research-only**. Stock clocks remain the qualified deployment
default. The exact Jovian Judgement Community image ID
`sha256:26e40eeb6506d7fcff64b0ac155dee4d08b702a455e689533ceb60efaa874f88`
was measured in TP4/DCP1 DFlash2 mode with seven draft tokens. Only physical
GPUs 4–7 received an NVML memory-clock voltage/frequency offset of `+6000`;
NVIDIA reported a 13,365 MHz memory clock under load. The offset was returned
to zero on all four GPUs after measurement.

The stock and overclocked cells used the same 32,320-token prefill request and
30-second CC1 method specified above. Each value is the median of three
samples.

| Metric | Stock clocks | VRAM +6000 | Change |
|---|---:|---:|---:|
| 32k TTFT | 2.1157 s | **2.0508 s** | -3.07% |
| 32k prompt throughput | 15,276.1 tok/s | **15,759.6 tok/s** | +3.17% |
| CC1 output | 185.49 tok/s | **200.87 tok/s** | +8.30% |
| Engine rate | 74.19 steps/s | **82.91 steps/s** | +11.75% |
| Accepted length | 2.49 | 2.43 | -2.57% |

The overclocked TTFT samples were 2.04717, 2.05081, and 2.05165 seconds. Its
CC1 output samples were 205.29, 200.13, and 200.87 tok/s; engine-rate samples
were 82.91, 82.93, and 82.78 steps/s; accepted-length samples were 2.48, 2.41,
and 2.43. Engine rate is the cleaner execution-speed comparison because
speculative output throughput also changes with accepted length.

The CC1 client command was:

```bash
python3 llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 5001 \
  --model GLM-5.3-Flash-NVFP4 \
  --concurrency 1 \
  --contexts 0 \
  --duration 30 \
  --max-tokens 8192 \
  --temperature 0 \
  --decode-warmup-seconds 3 \
  --skip-prefill \
  --no-hw-monitor \
  --show-capacity-limited-values \
  --no-resume \
  --output jovian-judgement-cc1.json
```

## TP3 three-GPU candidate (research overlay, 2026-09-08)

Status: **research-only validated**. Everything above documents the qualified
four-GPU (TP4) Jovian Judgement deployment and stays the qualified serving
path. The section below pins a three-GPU TP3 candidate built on the R27 source
composition, reconstructed as a minimal mergeable port (`-min` branches on the
`infernix` forks) with every surviving commit tracked per the ledger in
[infernix/vllm#1](https://github.com/infernix/vllm/issues/1)
("Canonical merge set (start here)").

### TP3 source contract and artifact

```text
infernix/vllm:glm53-r27-tp3-min-recipe01c67936a364-r6
infernix/vllm@sha256:1871c46156aaac7a785286feea0533d290e777a91f193e23a1f699fdf1ffdcc1
```

- Recipe branch `research/glm53-r27-tp3-min-overlay` @
  `068293ffaca79782b3e873c3c46909a6946791e0`
  ([infernix/blackwell-llm-docker](https://github.com/infernix/blackwell-llm-docker/tree/research/glm53-r27-tp3-min-overlay));
  parent image
  `voipmonitor/vllm:jovian-judgement-community-20260906-r27@sha256:a298fe1cd207…`
  (the R27 runtime documented by the master-model page; the r8 artifact of
  this page is not in that lineage).
- vLLM `-min` branch `research/glm53-tp3-r27-min` @
  `1ad233f31a2c932ca2ab86935ef604ad5983e4bb` (tree `e39e3279aef2…`) —
  27 commits / 30 files, +3,586/−153 vs the R27 base
  `63a82f8d323e…`
  ([compare](https://github.com/infernix/vllm/compare/63a82f8d...1ad233f3)).
  Head fix: `fix(glm53): type the boundary-checkpoint restore scalars`
  (`1ad233f3`) — the restore kernel got host Python ints where the traced
  body called `.to(tl.int64)`, so any prefix-cache hit that restored a
  boundary checkpoint killed `EngineCore`; every later request answered
  HTTP 500 until the fix.
- B12X `-min` branch @ `7e0d491111fc02b60f8e0aa27d0380336fa88424` (tree
  `cc8d7ec10ce5…`) — 21 commits, 29 files, +2,145/−648 vs base
  `e8ad299b174f…`
  ([compare](https://github.com/infernix/b12x/compare/e8ad299b...7e0d4911)).
  The head commit is import-order only (content-equal to the qualified bake),
  so it required no fresh requalification; ruff repo-default census stays
  229→229 and the branch's own touched-file select improves 1→0.
- Overlay lock: packaged `source.lock` `cf3b0f5524…`, TP3 launcher
  `6560f35d5f…`, verifier `62c18385cf…`; TP3 compiled-artifact fingerprint
  `cu133-torch213-glm53-r27-tp3-vllm1ad233f3-b12x7e0d4911-dense-ctx1m-seq8-bt8192`.
- Checkpoint revisions are pinned, not branch-guessed:
  target `local-inference-lab/GLM-5.3-Flash-NVFP4@46aaae8a8203…`,
  draft `local-inference-lab/GLM-5.3-Flash-DFlash2@dfa270d7eb8d…` (MXFP8).
  The older `huggingface-main` draft label was never a resolvable revision and
  made every speculative launch fail closed at `SpeculativeConfig` validation.

### TP3 launch commands — what differs from the four-GPU runbook

The TP3 chain runs its own strict launcher: every value below is enforced by
fail-closed `lock_env` gates — any caller-passed value that differs from the
locked one exits 2, and the two model revisions are locked (they cannot be
re-pointed from the command line).

Common parameters differ only where the runbook above had four GPUs:

```bash
IMAGE=infernix/vllm@sha256:1871c46156aaac7a785286feea0533d290e777a91f193e23a1f699fdf1ffdcc1
GPU_DEVICES=0,1,2
```

Select exactly one serving mode:

```bash
# TP3 without speculation
TP3_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=0)
# TP3 with built-in MTP, three draft tokens
TP3_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=3)
# TP3 with DFlash2, its trained seven draft tokens; the draft is pinned by revision in the image
TP3_ARGS=(-e SPECULATOR=dflash2 -e DFLASH_DEPTH=7)
```

Select exactly one cache mode (vram is the default):

```bash
CACHE_ARGS=(-e CACHE_MODE=vram)                                                    # GPU-resident KV
CACHE_ARGS=(-e CACHE_MODE=native -e NATIVE_KV_OFFLOADING_SIZE_GB=64)               # native DRAM KV offload tier
CACHE_ARGS=(-e CACHE_MODE=lmcache -e LMCACHE_L1_SIZE_GB=64 -e LMCACHE_L1_INIT_SIZE_GB=2)  # LMCache DRAM L1
```

Run:

```bash
docker rm -f "$NAME" 2>/dev/null || true
docker run -d \
  --name "$NAME" \
  --init \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --network host \
  --ipc host \
  -v glm53-r27-tp3-cache:/cache \
  -v glm53-r27-tp3-hf-cache:/root/.cache/huggingface \
  -e PORT=8000 \
  -e TP=3 \
  -e VLLM_MTP_NVFP4_LM_HEAD=1 \
  "${TP3_ARGS[@]}" "${CACHE_ARGS[@]}" \
  "$IMAGE"
```

| Knob | Four-GPU runbook (r8 page, above) | TP3 overlay (locked; do not pass) |
|---|---|---|
| GPUs / `TP` / `DCP` | 4 cards, TP4, DCP1 (or DCP4) | 3 cards, TP3/EP3, DCP1 |
| `PORT` | 5001 | 8000 |
| `MAX_MODEL_LEN` | 262,144 | 1,048,576 |
| `MAX_NUM_SEQS` | 16 | 8 |
| `MAX_NUM_BATCHED_TOKENS` | 4,096 | 8,192 |
| `PREFILL_SCHEDULE_INTERVAL` / fairness | 8 (inactive without decode pressure) | 8 alongside `FAIRNESS_ENGINE=none`, `PREFILL_COMPUTE_SHARE=none` |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | 128 | 16 (capture sizes 1/2/4/8/16) |
| `GPU_MEMORY_UTILIZATION` | 0.90 | 0.91 |
| NCCL channels | 32 min / 32 max | 16 min / 16 max |
| MTP NVFP4 proposal head | parent-image default | must pass `VLLM_MTP_NVFP4_LM_HEAD=1`; the overlay image bakes `0` and the launcher fails closed on it |
| KV / loader / backends | page's r8 contract | `KV_CACHE_DTYPE=fp8`, `LOAD_FORMAT=instanttensor`, `block-size 256`, B12X attention, MoE `auto` + MTP MoE `humming`, DFlash2 FlashAttention, KDA decode `b12x` / prefill `flashkda` |
| DFlash2 draft KV | main-branch resolution | pinned `dfa270d7eb8d…`, draft KV dtype `auto` (BF16 sliding-window) |

DFlash2 draft KV stays BF16 `auto` under the TP3 chain (sliding-window
bounded); `FAIRNESS_ENGINE=none` + `PREFILL_SCHEDULE_INTERVAL=8` ship as one
knob pair.

Fail-closed in the opposite direction too: the verifier's `REQUIRED_UNSET_ENV`
set in `overlays/glm53-r27/tests/verify_glm53_flash_nvfp4_runtime.py` must not
be provided at all —
`KV_CACHE_QUANT`, `VLLM_KV_CACHE_LAYOUT`, `VLLM_SSM_CONV_STATE_LAYOUT`,
`VLLM_DP_SIZE`, `VLLM_DP_RANK`, `VLLM_DP_RANK_LOCAL`, `VLLM_DP_MASTER_IP`,
`VLLM_DP_MASTER_PORT`, `GLM53_TARGET_BLOCK_SIZE`, `GLM53_MAMBA_BLOCK_SIZE`,
`NCCL_ALGO`, `NCCL_COLLNET_ENABLE`, `NCCL_NVLS_ENABLE`, `NCCL_SHM_DISABLE`,
`NCCL_PXN_DISABLE`, `NCCL_P2P_DIRECT_DISABLE`, `VLLM_PCIE_DMA_FP8`,
`B12X_PCIE_DMA_FP8` — so a reader who copies the four-GPU block above and adds
their usual NCCL/DP defaults without dropping them first exits 2.

### TP3 validation wave (2026-09-08, six passes)

Hardware: one RTX PRO 6000 Blackwell Workstation Edition host rented on
vast.ai (offer `42445861`: 4 GPUs, stock clocks, driver `580.82.09`, CUDA
`13.0` host on a `cu133` image via driver minor-version compatibility), 755
GiB RAM, NVMe-only weights, `/dev/shm` 188 GiB. Serving used GPUs 0–2 of 4 in
one container from the exact digest above; every pass restarted the server
fresh and recorded `/proc/<pid>/cmdline` plus env. Weights 186 GiB streamed
from Hugging Face in ~25 minutes; per-pass server startup 191–246 s including
the first-pass JIT compile on a cold `/cache/jit` fingerprint.

Method of this wave (its own instruments; not the `llm-decode-bench` CC1
cells above): each pass runs the frozen probe corpus, 5x same-prompt
determinism, a 1,000,035-token admission (end-to-end request time), a 4x
131,107-token concurrent prefill burst, and an identical-prompt prefix-cache
double pass. Receipts:
[receipts-vast-r6/](https://github.com/infernix/rtx6kpro/tree/docs/glm53-r27-tp3-min-20260909/benchmarks/data/glm53-r27-tp3-min-20260909/receipts-vast-r6)
on this branch.

The 1M-prefill column is single-request end-to-end elapsed (~8.3–8.8k tok/s
over ~121 s); the 4x131K burst column reports the per-request elapsed while
four requests run simultaneously, so the aggregate over that window is 4x the
tokens — it is also persisted-L2 warm on the two lmcache rows, as is the
prefix column's pass-1 lead. The vision probe only runs on DFlash2 passes, so
`n/a` elsewhere is by design. KV pool totals differ across cache modes only
because native/lmcache add a 64 GiB DRAM tier to the same accounting; they
are not same-shape sizes across modes.

| Pass (mode x cache) | 1M-prefill tok/s | 4x131K burst (s) | Prefix reuse | KV pool tokens | Corpus | Vision | Determinism | Startup |
|---|---:|---|---|---:|---|---|---|---:|
| dense / vram | 8,267 | 24.3–24.8 all ok | 27.4x (p2 0.22 s) | 3,097,517 | 5/5 | n/a | 5/5 | 231 s |
| mtp3 / vram | 8,376 | 17.0–24.0 all ok | 25.5x (p2 0.23 s) | 2,280,455 | 5/5 | n/a | 5/5 | 236 s |
| dflash2 / vram | 4,388 | 14.5 uniform | 14.5x (p2 0.40 s) | 2,078,852 | 4/5 | 1/2 | 5/5 | 246 s |
| dense / native | 8,754 | 11.9 uniform | 12.0x (p2 0.49 s) | 3,121,851 | 5/5 | n/a | 5/5 | 216 s |
| dense / lmcache | 8,591 | 2.1–2.5 | 1.02x, pass-1 already warm (0.53 s) | 3,563,566 | 5/5 | n/a | 5/5 | 191 s |
| dflash2 / lmcache | 8,571 | 0.9–1.9 | 1.31x, pass-1 already warm (0.54 s) | 2,465,073 | 4/5 | 1/2 | 5/5 | 196 s |

Honest reads:

- Prefix speedup ≈ 1.0 on the lmcache passes is a warm-lead artifact: the
  persisted L2 on the same host disk already held the probe KV, so pass 1
  itself is ~0.5 s. The vram/native rows are the honest cold comparisons.
- The dense/mtp3/native/lmcache prefill columns come from the 1,000,035-token
  admission (full long-context attention tail), so they must not be compared
  against this page's 32k TP4 cells or any other short-context figure.
  Mid-context reads from inside the wave (per-request 131K burst of the
  native pass) land in the same order as card-proportional scaling of the
  32k row; matched-context TP3 cells are not yet recorded here.
- DFlash2/vram pays a 2x 1M-prefill penalty (227.9 s vs 116.7 s with
  lmcache) that this wave does not attribute; DFlash2 also returns empty
  content for the `QAD borderline` graded row in both cache modes (4/5 in
  both), and the vision green row answers empty within its 96-token budget
  (red proves the vision path works). Attributions are open follow-ups.
- No decode ladder or draft-acceptance measurement ran in this wave; C1/C8
  columns above belong to the r8 four-GPU artifact only and are not claimed
  for TP3.

Cross-generation anchors measured on the same GPU class (previous R21 TP3
qualification receipt): KV pools 3,074,098 / 2,292,781 / 2,082,512 tokens
(ordinary / MTP3 / DFlash2) vs this wave's 3,097,517 / 2,280,455 / 2,078,852 —
every pool within ±1%; model load 629–651 s/rank then vs 191–246 s
pass-inclusive now; the only same-definition 1M datapoint in the R21 receipt
is DFlash2 retrieval at 198.2 s (5,046 tok/s) vs this wave's vram 227.9 s
(4,388 tok/s) and lmcache 116.7 s (8,571 tok/s). R21 pinned target
`378ca545…` and draft `aea0ac8a…`; this wave pins `46aaae8a…` and
`dfa270d7eb8d…`, so checkpoint drift rides along the codebase delta.

## Limitations

- The source pull requests in the source contract are open review units. Use
  the image digest until the required heads are merged and a merged-only image
  is separately qualified.
- Full prefill CUDA-graph capture is unsupported by the GLM GDN backend in this
  source tree. Decode remains graph-captured.
- B12X is the main target backend, but the complete runtime intentionally uses
  PyNCCL outside B12X all-reduce sizes, Humming for MTP experts,
  FlashAttention 2 for DFlash2 local attention, FlashKDA for target prefill,
  and FlashInfer for sampling.
- Raw speculative tok/s is prompt- and acceptance-dependent. Record engine
  steps per second and accepted length with every MTP or DFlash2 comparison.
- The TP3 candidate overlay binds branch heads and digests rather than merged
  pull requests (the `-min` branches are the merge proposals); receipts, the
  kept/dropped commit ledger, and the lint census live in
  [infernix/vllm#1](https://github.com/infernix/vllm/issues/1). Open items are
  listed in the [TP3 candidate section](#tp3-three-gpu-candidate-research-overlay-2026-09-08).
