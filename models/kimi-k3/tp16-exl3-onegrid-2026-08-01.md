# Kimi K3 TP16 MXFP4/EXL3 one-grid progress

Status date: 2026-08-01. This page records the development state before the
first full-artifact E2E load. All numerical and compiler gates below were run
without loading Kimi K3 as a model. The already-running full-MXFP4 server was
left untouched.

## Outcome so far

Kimi K3's 3,072-wide routed experts split to exactly 192 channels per rank at
TP16. The implemented path has no expert parallelism, expert padding, or
TP8/PP2 fallback:

- the retained experts stay in their original ModelOpt MXFP4 representation;
- demoted experts are quantized after TP sharding as EXL3-3 MCG;
- each rank-local 192-channel transform is H128 followed by an H64 tail;
- all 896 global expert IDs remain addressable, with a 32-bit descriptor
  `(tier << 16) | tier_local_id`;
- decode M=1/top-k=16 runs both tiers in one cooperative SparkInfer grid;
- prefill and larger M retain the serial correctness path for now.

This is the same broad one-grid strategy used by GLM's Grid188 path, but it is
not the same kernel: K3 needs original MXFP4 plus full-rotation EXL3, 896 route
prefixes, rank-local H64 tails, and per-layer tier counts.

The old artifact at
`/mnt/luke/models/Kimi-K3-EXL3-3p09-MXFP4-44p0-TP16-1M-20260801` is **not a
valid TP16-local quant**. It is a size-only 44% repack of a globally H128
quantized artifact (manifest schema 2) and lacks `tp_local_quantization`,
`compatible_tp_sizes`, `tp_local_intermediate_size`, and
`intermediate_hadamard_blocks`. The new loader rejects it rather than serving
silently incorrect rank slices.

## Published branches (no PRs)

| Repository | Branch / head | Purpose |
|---|---|---|
| SparkInfer | [`codex/kimi-k3-mxfp4-exl3-onegrid-20260801`](https://github.com/local-inference-lab/sparkinfer/tree/codex/kimi-k3-mxfp4-exl3-onegrid-20260801) @ `cf77fab` | One-grid kernel, exact 896-expert route packing, actual/synthetic harnesses; includes Luke's `origin/fix/w4a16-planning-capture-current-20260801` stack through `0191725` |
| vLLM | [`codex/kimi-k3-tp16-mxfp4-exl3-onegrid-20260801`](https://github.com/local-inference-lab/vllm/tree/codex/kimi-k3-tp16-mxfp4-exl3-onegrid-20260801) @ `9f91322e3` | Fail-closed TP-local loader, exact per-layer one-grid preparation and decode dispatch |
| kquant | [`codex/kimi-k3-tp16-local-exl3-onegrid-20260801`](https://github.com/local-inference-lab/kquant/tree/codex/kimi-k3-tp16-local-exl3-onegrid-20260801) @ `f45264c` | Quantize after TP16 sharding, atomic/resumable per-layer replacement, manifest/artifact validation and smoke harness |
| ExLlamaV3 fork | [`codex/kimi-k3-tp16-local-hadamard-20260801`](https://github.com/voipmonitor/exllamav3/tree/codex/kimi-k3-tp16-local-hadamard-20260801) @ `c1518ee` | H128+H64 EXL3 transforms and bounded tile scratch |

Do not merge the vLLM branch into `dev/gg-k3` until the full artifact passes
load, decode, long-generation, and KLD gates. No PR was created.

## Model-free gates already passed

### Quantizer and real expert

The real-expert smoke test opens only the packed weight and scale for
w1/w3/w2 of one expert (six tensors total):

```bash
export PYTHONPATH=/tmp/kquant-k3-onegrid:/tmp/exllamav3-k3-tp16
python scripts/smoke_exl3_tp_local_expert.py \
  --layer 1 --expert 7 --tp-size 16 --batch 32 \
  --temp-batch-size 128 --device cuda:0
```

Measured with the full model still resident on the GPU:

- 8.2-8.8 seconds at scratch batch 128, peak allocation 922 MiB;
- 11.0 seconds at scratch batch 64, peak 664 MiB;
- 36.5 seconds at scratch batch 16, peak 470.5 MiB;
- proxy errors: w1 `0.0174386`, w3 `0.0174443`, w2 `0.0174363`;
- merged ABI: w1/w3 trellis `[224,192,48]`, w2 `[192,224,48]`.

Scratch batch 256 correctly OOMed while the full model left only about 2 GiB
free. It is intended for the offline rebuild after the model is stopped.

### Actual-weight kernel closure

`benchmarks/check_kimi_k3_actual_tp16_expert.py` caches the approximately
one-expert quant output and then performs all subsequent kernel iterations
without checkpoint I/O. Rank 0 and rank 15 both passed:

- checkpoint tensors opened on cached runs: 0;
- extra peak allocation: 9.4 MiB;
- serial full-rotation Trellis vs mixed one-grid relative L2: `0.0`;
- CUDA graph replay: bit-exact with eager output.

The synthetic production-geometry harness also passed mixed, MXFP4-only, and
EXL3-only routes. Mixed relative L2 was `4.0473e-4`; graph replay drift was
zero. The measured ~0.046 ms synthetic graph time is a kernel harness number,
not an E2E model throughput claim.

### Every real per-layer split

The allocation contains 81 unique `(kept, EXL3)` counts across 92 MoE layers.
All 81 variants were compiled without opening weights:

```bash
python benchmarks/precompile_kimi_k3_mxfp4_exl3_onegrid.py \
  /mnt/luke/models/Kimi-K3-EXL3-3p09-MXFP4-44p0-TP16-1M-20260801/allocation-exl3.json
```

Result: 81/81 in 302.49 seconds, slowest 4.39 seconds, 33,920 bytes shared
memory, 145 registers/thread, and zero local-memory spill for every split
from `2+894` through `650+246`.

### Unit and loader tests

- kquant TP-local/artifact tests: 11/11;
- ExL H128+H64 transform tests: 3/3;
- vLLM hybrid configuration/geometry tests: 20/20;
- InstantTensor small-shard, index-aware selection, and oversize fallback
  tests: 3/3;
- SparkInfer actual expert and synthetic GPU closure harnesses: pass.

vLLM already contains `88721d90d` (`loader: bound InstantTensor GPU memory for
Kimi K3`). Use `--load-format instanttensor`; optional
`INSTANTTENSOR_BUFFER_SIZE` is expressed in bytes. No EXL-specific
InstantTensor patch was required because the loader streams physical tensors
by the model's shard index and consumes each tensor inline.

## Safe full TP16-local rebuild

There are only about 294 GiB free while the invalid artifact occupies about
1.2 TiB, so a second complete artifact cannot coexist. The new driver replaces
one completed layer at a time through a temporary safetensors file and atomic
rename. Sixteen concurrent temporary layer files fit in the available space;
verify this again immediately before starting.

The build must run only after the current model is intentionally stopped and
all 16 GPUs are free. From the quantization container with the four branch
sources checked out:

```bash
export KQUANT_EXL3_SHARED_SU=1
export PYTHONPATH=/tmp/kquant-k3-onegrid:/tmp/exllamav3-k3-tp16

python /tmp/kquant-k3-onegrid/scripts/pack_exl3_12gpu.py \
  --dest /mnt/luke/models/Kimi-K3-EXL3-3p09-MXFP4-44p0-TP16-1M-20260801 \
  --reuse-allocation \
  --replace-unmarked \
  --tp-size 16 \
  --num-workers 16 \
  --temp-batch-size 256 \
  --build-id k3-exl3-tp16-h128h64-v1-20260801
```

Each layer receives a completion marker bound to build ID, TP size, H128/H64
contract, shared-SU, codebook, shard size, and error-file size. A crash leaves
`build_complete=false`; rerunning the identical command skips only verified
layers and atomically rebuilds everything else. Packaging refuses an
incomplete build.

With 46,162 EXL3 experts, the low-VRAM measurements imply hours rather than
minutes for the real conversion. A conservative first estimate is 5-8 hours
on 16 free GPUs at scratch batch 256; measure the first completed layers before
publishing an ETA.

After all 92 markers validate:

```bash
python /tmp/kquant-k3-onegrid/scripts/package_exl3_serve_dir.py \
  /mnt/luke/models/Kimi-K3-EXL3-3p09-MXFP4-44p0-TP16-1M-20260801 \
  /mnt/luke/models/Kimi-K3-EXL3-3p09-MXFP4-44p0-TP16-1M-20260801-serve \
  --nonexpert /mnt/luke/models/Kimi-K3-mxfp8-nonexpert
```

Run `kquant.artifact.validate_exl3_artifact()` before serving. It checks the
allocation, build markers, TP transform metadata, serve config/index, and all
safetensors headers without materializing payloads.

## Remaining E2E gates

1. Rebuild the actual TP16-local artifact with all GPUs free.
2. Validate and package it; start vLLM with the published vLLM/SparkInfer
   branches and `--load-format instanttensor`.
3. Confirm logs show the K3 EXL3 one-grid being armed and executed, with no
   serial fallback during M=1 decode.
4. Measure true decode tokens/second separately from prefill and leave the
   server running.
5. Run deterministic short prompts, the long Tetris/SMB corruption harness,
   and a generation beyond the prior deterministic failure token.
6. Capture the same KLD suite and compare it with the full-MXFP4 reference in
   `kld-reference-logits-32x2048.md`.
7. Measure actual KV capacity; the one-grid metadata/buffers add only roughly
   100 MiB per rank, but the 1M-token target is not considered proven until the
   full runtime profile completes.
