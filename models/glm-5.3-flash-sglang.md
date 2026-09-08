# GLM-5.3-Flash — SGLang program on RTX PRO 6000 (SM120)

Status: **research-only, hand-operated**. This page records the working
SGLang serving path for GLM-5.3-Flash-NVFP4 on the four-GPU RTX PRO 6000
Blackwell Workstation (SM120) surface that the four-GPU
[GLM-5.3-Flash](glm-5.3-flash.md) page covers with vLLM for the qualified
artifact. SGLang on SM120 is not a qualified deployment: it validated via a
six-patch compatibility stack and needs the operation notes below to survive.
The vLLM TP3 investigation ledger lives in
[infernix/vllm#1](https://github.com/infernix/vllm/issues/1).

## Source recipe

The validated bundle is the independently locked representation of a live
SM120 deployment:

- Repo: [0xSero/glm-5.3-flash-sglang-sm120](https://github.com/0xSero/glm-5.3-flash-sglang-sm120)
  (`compose.yaml`, `manifest.json`, six patches, multimodal chat template).
- SGLang cookbook page for the model family:
  [docs.sglang.io GLM-5.3-Flash](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash).
- The sm120 failure taxonomy and the measured work-configuration are in
  [sgl-project/sglang#37105](https://github.com/sgl-project/sglang/issues/37105).

## Why the defaults do not run on SM120

On RTX PRO 6000 (SM120, no TMEM/tcgen05, 102 KB shared memory per SM):

1. `deep_gemm` NameError: `configurer.py` excludes SM120 from deep_gemm, but
   `SGLANG_OPT_DEEPGEMM_HC_PRENORM` defaults True, so the MHC layernorm path
   calls an unbound `deep_gemm.tf32_hc_prenorm_gemm` → `NameError` at model
   init. Fix: env `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`.
2. TRT-LLM FMHA refuses the GPU: as auto-selection lands on `trtllm` for all
   Blackwell (`sm_major >= 10`) and `TllmGenFmhaRunner` only supports SM100.
   Renaming to `flashmla_sparse` does not help: the resolver redirects back to
   `trtllm`. **Explicit `tilelang` on both ends is the only working stock DSA
   path in, and it needs a tile retune on SM120** — stock
   `block_I=64/num_stages=2/threads=256` requests more dynamic shared memory
   (even 151.5 KB for BF16 KV) than SM120's ~100 KB allows; the validated
   override is `(block_I, num_stages, threads) = 32, 1, 128`.
3. The stack that ships Native-SM120 support without any kernel retune:
   FlashInfer [#4802](https://github.com/flashinfer-ai/flashinfer/pull/4802)
   (SM120 sparse-MLA runner) + the bundle's `flash_mla_sm120` Triton/PyTorch
   fallback module + the `dsa_backend.py` patch — this is the configuration
   recorded on this page. `flashinfer_cutedsl` is NOT usable here: FlashInfer's
   CuteDSL MoE hard-codes `supported_major_versions=[10]`.
4. On consumer/pro-workstation Blackwell (RTX PRO 6000, SM120), MoE runner must be
   `flashinfer_cutlass` — the SM120-native CUTLASS FP4 path in this image.

## Locked image and checkpoint

```text
lmsysorg/sglang@sha256:3a97bd50034ca60c6e6c86b8e36a73675d261f6a5eb71197796aee5175409290
```

That pinned digest is the bundle's lock (recognize the `glm-5.3-flash` tag
moves; reinstalling from the tag brought a different tree). The checkpoint
served from the recipe below is
`RadixArk/GLM-5.3-Flash-NVFP4` @ `f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35`
(190 GiB total, 51 files, ModelOpt 0.46.0 NVFP4 W4A4, abs-max scaling, group
size 16). The bundle's own lock references
`LibertAIDAI/GLM-5.3-Flash-NVFP4 @ 9e0d74e3cef1…` (182 GiB, 120 shards); the
~2.2 GiB/Rank checkpoint-size delta between the two matters for memory
planning (see the failure taxonomy).

The six patch files are byte-verified replacements at bundle target paths —
the `manifest.json` records the sha256 of each:

| Patch (source file) | Target path under `/sgl-workspace/sglang/python/sglang` | SHA-256 (16-char prefix) |
|---|---|---|
| `sglang-glm5_next-debug.py` | `srt/models/glm5_next.py` | `aeb9eb145958644d` |
| `sglang-deepseek_nextn-glm53.py` | `srt/models/deepseek_nextn.py` | `dce70ed739270215` |
| `sglang-quant-utils-sm120.py` | `srt/layers/quantization/utils.py` | `0814b359c0be04b7` |
| `sglang-modelopt-quant-sm120.py` | `srt/layers/quantization/modelopt_quant.py` | `10367ba1573c11e5` |
| `sglang-flash_mla_sm120-glm53.py` | `kernels/ops/attention/flash_mla_sm120.py` | `06c7f42a192997ab` |
| `sglang-dsa_backend-glm53.py` | `srt/layers/attention/dsa_backend.py` | `1b786b161e9279b7` |

Plus `chat-template-mm.jinja` installed at `/chat-template.jinja` (byte-exact
copy; it gives the multimodal conditional-generation template the text-only
default cannot render).

## Server launch (validated on 2026-09-08)

Environment (the bundle's exact lock):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_CUDA_ARCH_LIST=12.0a \
  FLASHINFER_CUDA_ARCH_LIST=12.0f \
  SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_TILELANG_INDEXER=1 \
  SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  SGLANG_OPT_FP8_WO_A_GEMM=0
```

Server:

```bash
cd /sgl-workspace
python3 -m sglang.launch_server \
  --model-path RadixArk/GLM-5.3-Flash-NVFP4 \
  --served-model-name glm-5.3-flash \
  --tp-size 4 --ep-size 4 \
  --context-length 1048576 \
  --quantization modelopt_fp4 \
  --attention-backend dsa \
  --dsa-prefill-backend flashinfer_sparse_mla \
  --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_cutlass \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --max-running-requests 8 \
  --mem-fraction-static 0.86 \
  --cuda-graph-max-bs-decode 8 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 5 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 --speculative-adaptive \
  --media-url-max-file-size-mb 1024 --enable-multimodal \
  --chat-template /chat-template.jinja \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --host 0.0.0.0 --port 30000
```

Deltas vs the bundle's locked CMD (all measured here, none in the bundle):

| Knob | Bundle lock | This run | Reason |
|---|---|---|---|
| `--mem-fraction-static` | 0.90 | **0.86** | RadixArk checkpoint is ~2.2 GiB/rank larger than the bundle's LibertAIDAI lock; 0.90 OOMs during `create_weights`, 0.88 still hits allocator exhaustion mid-flight (see taxonomy). 0.86 + the reduced chunk leaves allocator headroom. |
| `--chunked-prefill-size` / `--max-prefill-tokens` | 8192 | **4096** | The observed crash allocating 2.96 GiB in a single prefill-workspace burst is chunk-proportional; 4096 halves the burst and matches the published four-GPU scheduler budget on [glm-5.3-flash.md](glm-5.3-flash.md). |
| `--speculative-algorithm` | NEXTN | NEXTN | The model card's own value; upstream folds EAGLE into it for glm5_next. |
| `--port` | 30000 (host mapping 8000) | 30000, mapped per | Vast.ai maps must be requested at `create` (`--env '-p 30000:30000'`); the assigned host port is whatever vast names in `show instance` (50113 in the recorded run). |
| `--max-running-requests` | 8 | 8 | Ask; also note SGLang forces 48 when unset with speculation on, so the explicit 8 is required to pin the value. |

## Operation notes (verified the hard way)

- **Fail-closed sequencing**: kill and launch belong in DIFFERENT ssh sessions
  (or use a bracketed pattern `[s]glang` — a chain where `pkill -f "sglang
  …"` and the launch share one shell dies at pkill because the pattern
  matches the shell's own command line).
- **Port maps are create-time on vast.ai**: `update instance --env` parses its
  value as a JSON object (no `-p` items) and a recycle can also strips the
  default 6006/8888 mappings. Reproduce reachability by re-creating the
  instance with `--env '-p 30000:30000'`, then read the actual host port from
  `show instance` ports and give out that host:port. SSH gateway
  host/port/keys change on every recycle: launch with
  `-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null`.
- **The container filesystem is wiped on every image update/recycle**
  (verified twice: a `/workspace` copy did not survive either). Everything
  under `/root` is ephemeral: downloads re-run downstream receipts must go to
  the workstation/docs commit immediately. Re-downloading RadixArk takes
  ~9 min at ~370-950 MB/s unauthenticated.
- Readiness on NVMe measured ≈ 4m40s (launch → `fired up and ready to roll`)
  including fp8 KV pool sizing and CUDA graph capture; prefill CUDA graph is
  disabled by design against the KDA hybrid path
  (`Breakable CUDA graph is incompatible... disabling prefill CUDA graph`).

## Verified serving facts (2026-09-08)

- Startup line: `DSA with TP mode is active, dp_size=1, tp_size=4,
  attn_tp_size=4`; `Set DSA backends for fp8_e4m3 KV Cache:
  prefill=flashinfer_sparse_mla, decode=flashinfer_sparse_mla`;
  `max_running_requests=8`; engine-reported `max_total_num_tokens=3,167,360`.
- Speculative decoding engages: decode-stat lines show **`accept len` 2.93–
  3.22** across samples with the 5/1/6 adaptive NEXTN configuration (drafts
  are accepted at ~3 of a possible 6 tokens per verify step).
- End-to-end proof from a different network host: `GET /health` → 200; one
  `POST /v1/chat/completions` with `chat_template_kwargs.reasoning_effort=
  "low"` returned exact expected content, `finish_reason=stop`.
- More substantial completions: a 149-token prose completion at the
  checkpoint's generation defaults (temperature 1.0, top_p 0.95),
  `finish_reason=stop`.

## Client usage notes

- Per the bundle README: under default (max) reasoning effort the visible
  `content` field stays empty while the budget burns `reasoning_content` —
  do not pass a small `max_tokens`; either let the model stop naturally or
  set `chat_template_kwargs.reasoning_effort = "low"`.
- Sampling: the checkpoint's `generation_config.json` defaults temperature
  1.0 / top_p 0.95 (the recipes keep them when the deterministic behavior is
  not required).
- Tools: send `tools` + `tool_choice` in requests; SGLang needs no separate
  enable flag with the `glm47` parser.
