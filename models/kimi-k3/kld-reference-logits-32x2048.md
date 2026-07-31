# Kimi K3 full-MXFP4 KLD reference logits (32x2048)

## Result

The canonical Kimi K3 quantization reference is available at:

- Hugging Face dataset: [festr2/kimi-k3-full-mxfp4-kld-reference-32x2048](https://huggingface.co/datasets/festr2/kimi-k3-full-mxfp4-kld-reference-32x2048)
- local artifact on the original host: `/mnt/luke/kld/kimi-k3-full-mxfp4-kld-32x2048-mixed-v1-20260731`

It contains 32 independent 2048-token windows, 65,504 scored next-token
positions, and full logits over Kimi K3's 163,840-token vocabulary. Each
window is one F32 safetensors file with shape `[2047, 163840]`; the logit
payload is 42,928,712,256 bytes (39.98 GiB).

The primary comparison is:

```text
KL(full original MXFP4 reference || candidate)
```

This reference answers whether a new EXL or other quantized checkpoint changes
the served full-MXFP4 model's distribution. It is not a BF16 ground-truth
reference: the source checkpoint itself stores the original experts in MXFP4.

## Why 32 independent windows

The earlier one-window reference was a useful kernel corruption gate but too
small for model-quality decisions. A 512-token sliding stride would repeatedly
score closely related contexts and overstate the effective sample size. This
suite instead takes evenly spaced, non-overlapping windows from three pinned
sources:

| Domain | Windows | Dataset revision |
|---|---:|---|
| prose | 16 | `Salesforce/wikitext` `wikitext-2-raw-v1` test, `b08601e04326c79dfdd32d625aee71d232d685c3` |
| code | 8 | `openai/openai_humaneval` test, `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` |
| instruction | 8 | `databricks/databricks-dolly-15k` train, `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` |

The canonical suite-token hash is:

```text
a6856e1d0504fd00d13c67a5515c081f349088664d7ea0894dc4d15db2c7d209
```

`suite-manifest.json` records each source construction, dataset fingerprint,
token-stream start, token JSON hash, and first/last token IDs. The exact token
IDs are stored under `tokens/`. Always submit those IDs directly to the
completions endpoint. Do not reconstruct prompts with a chat template.

## Source model and runtime identity

The reference used the unmodified full original Kimi K3 MXFP4 checkpoint:

| Item | Value |
|---|---|
| checkpoint inside container | `/root/k3-serve/model` |
| model type | `kimi_k3` / `KimiK3ForConditionalGeneration` |
| checkpoint index SHA-256 | `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd` |
| config SHA-256 | `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213` |
| tokenizer config SHA-256 | `5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e` |
| TP / DCP | TP16 / DCP1 |
| MoE path | SparkInfer B12X native MXFP4 W4A16 |
| load format | InstantTensor `0.1.9+consumer1` |
| vLLM commit | `55e46aeb4c6e8a2dab15f640929d65124a6d41df` (`dev/gg-k3`) |
| SparkInfer commit | `64a90970621c3da8a28b84e8ef5f06d7d4260de3` (`dev/gg-k3`) |
| static MLA splits | 8 |
| DCP indexer shards | 0 |

The server was already loaded when the suite was captured. Its exact command
was:

```bash
VLLM_KLD_CAPTURE_DIR=/mnt/luke/kld/kimi-k3-full-mxfp4-wikitext2-ctx2048-ref-20260731T2242Z/capture-chunks \
VLLM_TRITON_MLA_STATIC_KV_SPLITS=8 \
VLLM_DCP_INDEXER_SHARDS=0 \
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve \
  /root/k3-serve/model \
  --served-model-name Kimi-K3 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 16 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.982 \
  --compilation-config '{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1]}' \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --load-format instanttensor
```

`gpu-memory-utilization=0.988` and `max-num-batched-tokens=512` were not usable
for this diagnostic: the full-vocabulary all-gather needed another 160 MiB
while the rank had only 107 MiB free. The validated capture values above leave
enough headroom and chunk the prompt into at most 256 rows.

## vLLM capture hook

The K3 vLLM build does not support `SamplingParams.return_prompt_logits`.
The reference therefore uses the small environment-gated patch in
[`tools/vllm-kld-capture.patch`](tools/vllm-kld-capture.patch), applied to
`vllm/v1/worker/gpu_model_runner.py` at the commit above.

The hook:

- runs only when `VLLM_KLD_CAPTURE_DIR` is set;
- runs only on global rank 0;
- is triggered by a request containing `prompt_logprobs=1`;
- captures raw prompt logits immediately after `compute_logits`, before
  log-softmax and the API top-k reduction;
- transfers the BF16/FP16 tensor to CPU before widening it to F32;
- writes contiguous 256-row safetensors chunks without retaining the complete
  1.25 GiB window on GPU;
- refuses to overwrite an existing chunk.

Apply it from the vLLM checkout:

```bash
git switch dev/gg-k3
git apply /path/to/rtx6kpro/models/kimi-k3/tools/vllm-kld-capture.patch
```

The patched runtime file had SHA-256
`244a105cb062d5042917f9cbb7c6df45e83ce8a63873c8796b13ed4582241a0b`.
This is a diagnostic hook, not required for normal serving.

## Reproduce the corpus

The checked-in tools are:

- [`prepare-kimi-k3-kld-suite.py`](tools/prepare-kimi-k3-kld-suite.py)
- [`capture-kimi-k3-kld-suite.py`](tools/capture-kimi-k3-kld-suite.py)
- [`finalize-kimi-k3-kld-suite.py`](tools/finalize-kimi-k3-kld-suite.py)
- [`compare-kimi-k3-kld-suite.py`](tools/compare-kimi-k3-kld-suite.py)

Prepare and pin the 32 token windows with the source model's tokenizer:

```bash
python models/kimi-k3/tools/prepare-kimi-k3-kld-suite.py \
  --model /root/k3-serve/model \
  --output-dir /mnt/luke/kld/kimi-k3-suite \
  --context-length 2048
```

The generated suite is valid only if its suite-token hash equals the canonical
hash above. Tokenizer drift is therefore detected before comparing logits.

## Capture a candidate

Start the candidate with the same capture patch and a fresh
`VLLM_KLD_CAPTURE_DIR`, then run:

```bash
python models/kimi-k3/tools/capture-kimi-k3-kld-suite.py \
  --url http://127.0.0.1:8000/v1/completions \
  --model Kimi-K3 \
  --suite-dir /mnt/luke/kld/kimi-k3-suite \
  --capture-dir /mnt/luke/kld/candidate/capture-chunks \
  --run-name candidate

python models/kimi-k3/tools/finalize-kimi-k3-kld-suite.py \
  --suite-dir /mnt/luke/kld/kimi-k3-suite \
  --capture-dir /mnt/luke/kld/candidate/capture-chunks \
  --run-name candidate \
  --output-dir /mnt/luke/kld/candidate/ref \
  --expected-vocab 163840
```

Each request uses the exact token-ID list with:

```python
body = {
    "model": "Kimi-K3",
    "prompt": json.loads(token_file.read_text()),  # exactly 2048 integers
    "max_tokens": 1,
    "temperature": 0,
    "seed": 1,
    "prompt_logprobs": 1,
}
```

The capture client is resumable. A completed window is recorded immediately;
rerunning the same command validates and skips it.

For a quick eight-window development gate, add `--stop-window 8` to both the
capture and finalize commands. Use all 32 windows before selecting or publishing
a quantization recipe.

## Download and compare

```bash
hf download festr2/kimi-k3-full-mxfp4-kld-reference-32x2048 \
  --repo-type dataset \
  --local-dir /mnt/luke/kld/kimi-k3-reference

python models/kimi-k3/tools/compare-kimi-k3-kld-suite.py \
  --reference-dir /mnt/luke/kld/kimi-k3-reference/ref \
  --candidate-dir /mnt/luke/kld/candidate/ref \
  --suite-manifest /mnt/luke/kld/kimi-k3-reference/suite-manifest.json \
  --output /mnt/luke/kld/candidate/kld-vs-full-mxfp4.json
```

For the quick gate, add `--stop-window 8` to the comparator and download
`logits_000` through `logits_007` plus both manifests and token files.

The comparator first verifies the suite hash and every per-window token hash.
It then streams small row blocks from safetensors and computes:

- mean, median, P95, P99, and maximum `KL(reference || candidate)`;
- Jensen-Shannon divergence;
- top-1 token agreement;
- per-window and per-domain KLD;
- a 10,000-resample 95% bootstrap confidence interval clustered by window.

The window, not the individual token, is the statistical sampling unit. Tokens
within a document and within one autoregressive prefix are correlated.

## Interpretation and stopping rule

Use this workflow:

1. Reject broken kernels or obviously bad quants on windows 0-7.
2. Compare viable candidates on all 32 windows.
3. Prefer candidates whose confidence intervals and per-domain results improve,
   not merely the global point estimate.
4. If two candidates differ by only a few thousandths, capture both again and
   run downstream task evaluations rather than growing the same corpus without
   limit.
5. A final winner may be checked on a second disjoint 32-window suite. This is
   more informative than extending this fixed suite to 100 similar windows.

The observed same-model runtime noise is not zero. Two earlier independent
captures of window 0 gave:

| Metric | Value |
|---|---:|
| `KL(run1 || run2)` | 0.0035014991 |
| `KL(run2 || run1)` | 0.0035850708 |
| JS | 0.0008669901 |
| top-1 agreement | 98.8276% |

The new 32-window capture's window 0 versus the earlier canonical run gave
`KL=0.0029474339`, `JS=0.0007258723`, and 99.0718% top-1 agreement. It is within
the expected runtime/kernel nondeterminism. Treat changes around 0.004 or less
as noise unless repeated measurements show otherwise.

Do not apply a universal KLD quality label copied from another model. KLD
depends on the reference, vocabulary, corpus, and serving numerics. Rank Kimi
K3 candidates only against this exact reference and report the complete setup.

## Integrity checks

The completed reference was validated as follows:

- 32 files and 65,504 scored rows present;
- every file re-opened with safetensors;
- only the `logits` key present;
- every shape exactly `[2047, 163840]` and dtype F32;
- all 32 complete file SHA-256 hashes recorded in `ref/manifest.json`;
- every logit file linked to the corresponding token JSON hash;
- window 0 independently recaptured and checked against the previous run;
- capture completed in 109.71 seconds total, 3.43 seconds per window on average.

The Hugging Face `ref/manifest.json` is authoritative for file integrity and
the exact runtime/checkpoint identity.
