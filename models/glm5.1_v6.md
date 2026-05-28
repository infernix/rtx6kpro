# GLM-5.1 v6 Quant Sweep on 8x RTX PRO 6000 Blackwell

Measured starting on 2026-05-28 on the local 8-GPU RTX PRO 6000 Blackwell
host.

This page tracks the GLM-5.1 v6 quantization sweep. The first recorded speed
row is pure AWQ with DCP4 and MTP enabled. More quant variants will be added as
they are measured.

Current live benchmark JSON:

```text
/root/benchmark_results.json
```

## Recorded Runtime

The first v6 speed row below was recorded as pure AWQ:

```text
model:     /root/.cache/huggingface/hub/models--QuantTrio--GLM-5.1-AWQ/snapshots/8f60817aa28023f2607850d1a1e51d21aa34817a
served:    GLM-5
port:      5317
TP/DCP:    TP8, DCP4
attention: B12X_MLA_SPARSE
MoE:       b12x
KV cache:  fp8
MTP:       enabled, greedy, num_speculative_tokens=3
target:    quantization=awq
draft:     quantization=awq
```

The benchmark JSON at `/root/benchmark_results.json` does not embed the
checkpoint path or Docker image, so the quant label above comes from the run
context, not from the JSON metadata itself.

For reproducing AWQ/AWQ-hybrid MTP now, use an image containing the serialized
AWQ NextN fix described below:

```text
glm51-v5-awq-mxfp8-hybrid-mtpfix:test
```

Runtime scripts should deliberately unset `NCCL_GRAPH_FILE` and
`VLLM_B12X_MLA_EXTEND_MAX_CHUNKS`. Do not set either to an empty or forced value
for DCP runs.

## MTP AWQ Loader Fix

Pure AWQ and AWQ/MXFP8 hybrid checkpoints can serialize the GLM NextN/MTP expert
weights as AWQ. The unpatched vLLM image can fail when MTP is enabled. The
failure is:

```text
ValueError: moe_backend='b12x' is not supported for unquantized MoE
```

Root cause: `SpeculativeConfig.hf_config_override()` only recognized serialized
GLM NextN FP4 experts. For AWQ-family checkpoints, layer 78 MTP expert weights
are serialized as AWQ:

```text
model.layers.78.mlp.experts.0.down_proj.qweight
model.layers.78.mlp.experts.0.down_proj.qzeros
model.layers.78.mlp.experts.0.down_proj.scales
```

The override did not recognize those AWQ NextN experts and added
`model.layers.78.*` to the MTP ignore list. That made the MTP MoE instantiate as
unquantized even though AWQ weights were present. The `mtpfix` image adds a
targeted detection for serialized GLM NextN AWQ experts, so MTP layer 78 keeps
its AWQ quantization.

This is a runtime/code fix, not a checkpoint-content fix.

## KLD Validation

Reference BF16 artefacts:

```text
prefill: /root/kld/glm51_bf16_ref_vllm_wikitext_ctx2048_s512_w1_b12xmlasparse_20260516
decode:  /root/kld/decode_teacher_bf16_ref_ctx2048_t17_20260525_085332.safetensors
```

Current comparison:

```text
Variant                    Prefill KL   Decode BF16->var   Decode var->BF16           JS
----------------------------------------------------------------------------------------
NVFP4-MTP A16                0.098751         0.00001563         0.00002087   0.00000420
Mixed MXFP8 L42-62           0.059316         0.00004714         0.00008693   0.00001457
AWQ                          0.054161         0.00007133         0.00012410   0.00002153
AWQ+MXFP8 L42-62             0.036489         0.00006228         0.00009273   0.00001818
```

Pure AWQ artefacts:

```text
prefill log:
/root/kld/awq_dcp4_prefill_kld_vs_bf16_ctx2048_w1_20260528_021113.log

decode tensor:
/root/kld/decode_teacher_awq_dcp4_ctx2048_t17_20260528_021551.safetensors

decode compare:
/root/kld/decode_teacher_awq_dcp4_ctx2048_t17_20260528_021551_vs_bf16_teacher_compare.json
```

AWQ+MXFP8 L42-62 artefacts:

```text
prefill log:
/root/kld/awq_mxfp8_l4262_dcp4_prefill_kld_vs_bf16_ctx2048_w1_20260528_094353.log

decode tensor:
/root/kld/decode_teacher_awq_mxfp8_l4262_dcp4_a16_teacher_decode_ctx2048_t17_20260528_095103.safetensors

decode compare:
/root/kld/decode_teacher_awq_mxfp8_l4262_dcp4_a16_teacher_decode_ctx2048_t17_20260528_095103_vs_bf16_teacher_compare.json
```

Interpretation: AWQ+MXFP8 L42-62 has the best measured prefill KL in this
comparison set. Decode KL/JS is better than pure AWQ, but worse than NVFP4 A16
and the earlier pure mixed-MXFP8 L42-62 decode numbers.

## Benchmark Command

The first v6 benchmark uses `llm_decode_bench.py v0.4.23` against the live API:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5317 \
  --model GLM-5 \
  --concurrency 1,2,4,8,16,32,64,128 \
  --contexts 0,16k,32k,64k,128k \
  --duration 30 \
  --display-mode plain \
  --output /root/benchmark_results.json
```

The recorded run is DCP4, MTP enabled, greedy draft sampling, pure AWQ, and
`gpu_memory_utilization=0.835`.

## Pure AWQ DCP4 MTP Greedy Result

Prefill scout speed:

```text
ctx    tokens   TTFT s   tok/s   N
-----------------------------------
8k      8,195     3.49   2,350   1
16k    16,230     6.63   2,447   1
32k    32,339    13.84   2,337   1
64k    64,550    28.53   2,263   1
128k  128,965    58.83   2,192   1
```

Aggregate sustained decode tok/s:

```text
ctx \ conc      1      2      4      8     16     32     64          128
----------------------------------------------------------------------------
0            72.8  129.1  207.0  303.8  445.4  592.8  709.1  ∅ (64/128)*
16k          70.2  119.1  191.8  274.0  386.7      ∅      ∅            ∅
32k          69.2  120.4  184.1  276.7      ∅      ∅      ∅            ∅
64k          67.0  114.6  174.5      ∅      ∅      ∅      ∅            ∅
128k         67.9  111.4      ∅      ∅      ∅      ∅      ∅            ∅
```

Client request latency, p50 / p90 ms:

```text
ctx \ conc             1             2      4      8     16     32     64    128
--------------------------------------------------------------------------------
0            28.4k/28.4k  31.6k/31.8k    -/-    -/-    -/-    -/-    -/-    -/-
16k          28.6k/28.6k  33.6k/33.6k    -/-    -/-    -/-      ∅      ∅      ∅
32k          29.8k/29.8k  34.3k/34.3k    -/-    -/-      ∅      ∅      ∅      ∅
64k          31.0k/31.0k           -/-    -/-      ∅      ∅      ∅      ∅      ∅
128k         30.5k/30.5k           -/-      ∅      ∅      ∅      ∅      ∅      ∅
```

Hardware summary:

```text
ctx    C    GPU avg/max   Mem avg   W avg/max   T max   CPU T   VRAM   PCIe rx/tx avg
-------------------------------------------------------------------------------------
0      1        89/91%       31%    1609/1646     57C     76C   97.5%    31292/30344
16k    1        90/92%       31%    1673/1696     55C     76C   97.5%    33708/32208
32k    1       89/100%       31%    1693/1785     57C     76C   97.5%    29767/29051
64k    1        90/94%       30%    1696/1715     59C     76C   97.5%    29112/28280
128k   1        90/93%       30%    1728/1754     62C     76C   97.5%    35092/34225
0      2        90/93%       34%    1727/1757     56C     76C   97.5%    33856/32990
0      4        92/94%       36%    1841/1877     55C     76C   97.5%    53259/52012
0      8        94/96%       39%    1984/2026     55C     77C   97.5%    44813/43362
0     16        95/96%       41%    2104/2120     56C     76C   97.5%    63746/62157
0     32       97/100%       40%    2241/2268     58C     77C   97.5%    81443/81149
0     64       98/100%       37%    2276/2299     58C     77C   97.5%    99227/98971
0    128       98/100%       39%    2354/2382     60C     77C   97.5%  108758/110018
16k    2        91/93%       35%    1873/1886     57C     76C   97.5%    33146/30601
16k    4        93/95%       38%    2077/2097     57C     76C   97.5%    49428/48110
16k    8        95/96%       41%    2333/2355     58C     77C   97.5%    40141/39249
16k   16        96/97%       47%    2683/2722     61C     77C   97.5%    56148/55173
32k    2        91/93%       35%    1886/1904     58C     76C   97.5%    32963/30438
32k    4        93/96%       38%    2099/2114     57C     76C   97.5%    48846/47107
32k    8        95/96%       40%    2326/2354     59C     76C   97.5%    40001/38908
64k    2        91/93%       35%    1892/1902     56C     76C   97.5%    32189/29792
64k    4        93/96%       37%    2096/2112     57C     76C   97.5%    47441/46364
128k   2        92/93%       34%    1891/1907     56C     77C   97.5%    29956/30284
```

`∅` means the cell was skipped or hidden because it did not fit in KV cache for
this run. `*` means the JSON marks the cell as capacity-limited or otherwise
not a clean steady-state comparison point. Exact request-level details remain
in `/root/benchmark_results.json` under `request_samples`.

Burst/E2E decode was not run for this entry.

## Pending Rows

Add future quant variants below this section using the same benchmark and KLD
format:

```text
Variant              Checkpoint / source                                      Notes
----------------------------------------------------------------------------------
Pure AWQ             QuantTrio/GLM-5.1-AWQ local snapshot                     speed row recorded above
AWQ+MXFP8 L42-62     /root/kld/checkpoints/GLM-5.1-AWQ-MXFP8-L42-62-20260528  KLD measured, speed pending
NVFP4-MTP            TBD                                                      pending v6 sweep row
Mixed MXFP8 L42-62   TBD                                                      pending v6 sweep row
```
