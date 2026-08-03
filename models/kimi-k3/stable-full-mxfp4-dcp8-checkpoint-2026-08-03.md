# Kimi K3 full MXFP4 stable Docker checkpoint and rollback

Status date: 2026-08-03. This is the immutable rollback point made immediately
before moving Kimi K3 development from the GG-derived integration to
`dev/heraldic-harbinger` and SparkInfer's native dense MLA implementation.

The checkpoint serves the original `moonshotai/Kimi-K3` MXFP4 weights on 16
RTX PRO 6000 Blackwell GPUs with TP16, DCP8, InstantTensor, FP8 KV, and a
physical one-million-token cache. It does not use EXL3, NF3, expert
parallelism, pipeline parallelism, or a converted model checkpoint.

## Immutable artifacts

| Component | Immutable reference |
|---|---|
| Docker image | `voipmonitor/vllm:kimi-k3-full-mxfp4-dcp8-it-stable-20260803` |
| Docker registry digest | `sha256:2c99435142dd10f85834eaf4c490cb3d4095318152f0cc4fb38c7623d7edb7ac` |
| Local image ID | `sha256:7bbc397b0cb0f0760d89f4e707a83661d807e03d1f91aacc670ab6e4905be42f` |
| vLLM | [`checkpoint/kimi-k3-full-mxfp4-dcp8-stable-20260803`](https://github.com/local-inference-lab/vllm/tree/checkpoint/kimi-k3-full-mxfp4-dcp8-stable-20260803) at `0f3eb381ac8046f5a535a47f677dd26e421effb5` |
| SparkInfer | [`checkpoint/kimi-k3-full-mxfp4-dcp8-stable-20260803`](https://github.com/local-inference-lab/sparkinfer/tree/checkpoint/kimi-k3-full-mxfp4-dcp8-stable-20260803) at `01446e9d217f130ad4f5c202941e0b2eae3fa044` |
| InstantTensor | [`checkpoint/kimi-k3-full-mxfp4-dcp8-stable-20260803`](https://github.com/voipmonitor/InstantTensor/tree/checkpoint/kimi-k3-full-mxfp4-dcp8-stable-20260803) at `e5f34e3ee956c99f840249f781fc1f3f10ab2264` |
| InstantTensor wheel | `instanttensor-0.1.9+consumer1-cp312-cp312-linux_x86_64.whl`, SHA256 `1077d0b8fe0d97ee4b759018e1dc8b801fd8bdc65802a8ff00f391043f6fbabf` |
| Model | `moonshotai/Kimi-K3` snapshot `2496450e92e425c886db095102a52a6682ca3970` |

No PR was opened for any checkpoint branch.

The Docker image is 96.76 GB unpacked. It contains CUDA 13.2.1, PyTorch
2.12.0+cu132, the patched NCCL 2.30.4 build, the complete `/opt/venv`, and the
installed InstantTensor consumer-event wheel. The same wheel is also present
inside the image at:

```text
/root/k3-serve/instanttensor-0.1.9+consumer1-cp312-cp312-linux_x86_64.whl
```

The image labels beginning with `ai.voipmonitor.*` are the authoritative
checkpoint source revisions. Older `local-inference.*` labels describe the
base image from which the container evolved.

## Validated runtime contract

```text
model format: checkpoint-native compressed-tensors MXFP4, BF16 non-experts
load format: instanttensor
TP / DCP: 16 / 8
DCP backend: a2a
DCP KV interleave: 1
maximum model length: 1,048,576
maximum sequences: 1
maximum batched tokens: 256
KV cache dtype: FP8
KV cache allocation: 1,879,048,192 bytes/rank (1.75 GiB/rank)
GPU memory utilization: 0.985
CUDA graph mode: PIECEWISE, capture size 1
served model name: Kimi-K3-MXFP4-DCP8-1M-tail-fix
```

Measured startup and capacity:

| Item | Result |
|---|---:|
| InstantTensor iteration | 164 s in the final rerun |
| Complete model load | 194.55 s |
| Model memory | 90.99 GiB/rank |
| GPU KV cache | 1,059,851 tokens |
| 1M-request concurrency | 1.01x |

The final one-shot SMB test generated 73,514 completion tokens with
`finish_reason=stop` in 2,102.15 s, or 34.9709 completion tok/s including its
short prefill. It did not reproduce the old repeated-`!`/NaN failure.

The controlled decode gate used a fixed 256-token prompt and 1,024 forced
tokens. True decode throughput was measured between the first and final token:

```text
run 1: 34.803699 tok/s
run 2: 34.984788 tok/s
run 3: 34.987855 tok/s
mean:  34.925447 tok/s
median: 34.984788 tok/s
```

Any successor stack must use this exact test before a speed claim. The
regression floor is the stable median, 34.9848 tok/s; compare medians over at
least three warmed runs.

## What the image does and does not contain

Docker commit does not include bind mounts. The model and source trees must be
provided separately:

```text
/mnt/luke                                  -> /mnt/luke
/root/.cache/huggingface                   -> /root/.cache/huggingface
/root/luke-docker/ssh                      -> /root/.ssh
/root/luke-docker/ssh-host                 -> /etc/ssh-host (read-only)
```

The first two are required for inference. The SSH mounts are needed only to
reproduce the original administrative container exactly.

The source tree used by the final process was content-checked against the two
checkpoint branches. All production Python files matched. The runtime copy
only lacked or had older versions of four validation-only files; this did not
affect serving. The git checkpoint branches contain the newer validation
harnesses.

## Reconstruct the source overlays

On the host, recreate the exact bind-mounted paths:

```bash
git clone --filter=blob:none --single-branch \
  --branch checkpoint/kimi-k3-full-mxfp4-dcp8-stable-20260803 \
  https://github.com/local-inference-lab/vllm.git \
  /mnt/luke/vllm-k3-full-mxfp4-dcp8

git clone --single-branch \
  --branch checkpoint/kimi-k3-full-mxfp4-dcp8-stable-20260803 \
  https://github.com/local-inference-lab/sparkinfer.git \
  /mnt/luke/sparkinfer-k3-full-mxfp4-dcp8
```

The source overlay needs the binary extensions from the image's installed
wheel. Create links after the container starts:

```bash
docker exec kimi-k3-stable bash -lc '
set -euo pipefail
src=/mnt/luke/vllm-k3-full-mxfp4-dcp8/vllm
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

The final runtime had these root extensions available:

```text
_C_stable_libtorch.abi3.so
_moe_C_stable_libtorch.abi3.so
_qutlass_C.abi3.so
_rust_tool_parser.abi3.so
cumem_allocator.abi3.so
fs_io_C.abi3.so
spinloop.abi3.so
```

and the FA2/FA3 extensions under `vllm/vllm_flash_attn/`.

## Start the checkpoint container

Use the verified registry digest rather than only the mutable tag:

```bash
docker pull voipmonitor/vllm@sha256:2c99435142dd10f85834eaf4c490cb3d4095318152f0cc4fb38c7623d7edb7ac

docker run -d \
  --name kimi-k3-stable \
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
  voipmonitor/vllm@sha256:2c99435142dd10f85834eaf4c490cb3d4095318152f0cc4fb38c7623d7edb7ac
```

The original container was named `luke` (`cbc3ead9480b`) and used bridge
networking. Port 8000 was intentionally not published; it was reached at the
container bridge address (`172.17.0.2` in that instance).

Confirm the InstantTensor artifact before serving:

```bash
docker exec kimi-k3-stable sha256sum \
  /root/k3-serve/instanttensor-0.1.9+consumer1-cp312-cp312-linux_x86_64.whl

docker exec kimi-k3-stable /opt/venv/bin/python -c \
  'import importlib.metadata as m; print(m.version("instanttensor"))'
```

Expected values are the SHA256 above and `0.1.9+consumer1`.

## Launch the full model

```bash
docker exec -d kimi-k3-stable bash -lc '
export PYTHON_BIN=/opt/venv/bin/python
export PYTHONPATH=/mnt/luke/vllm-k3-full-mxfp4-dcp8:/mnt/luke/sparkinfer-k3-full-mxfp4-dcp8
export SERVED_MODEL_NAME=Kimi-K3-MXFP4-DCP8-1M-tail-fix
echo $$ >/mnt/luke/k3-full-mxfp4-dcp8-current.pid
exec /mnt/luke/vllm-k3-full-mxfp4-dcp8/serve-kimi-k3-full-mxfp4-dcp8-1m.sh \
  >/mnt/luke/k3-full-mxfp4-dcp8-stable.log 2>&1
'
```

The launcher fixes the model snapshot path to:

```text
/root/.cache/huggingface/hub/models--moonshotai--Kimi-K3/
  snapshots/2496450e92e425c886db095102a52a6682ca3970
```

Healthy log markers are:

```text
load_format=instanttensor
quantization=compressed-tensors
tensor_parallel_size=16
decode_context_parallel_size=8
Using 'B12X' Mxfp4 MoE backend
Model loading took 90.99 GiB memory
GPU KV cache size: 1,059,851 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.01x
Application startup complete
```

Verify the API:

```bash
curl -s http://172.17.0.2:8000/v1/models | jq \
  '.data[0] | {id, max_model_len}'
```

Expected:

```json
{"id":"Kimi-K3-MXFP4-DCP8-1M-tail-fix","max_model_len":1048576}
```

## Re-run the decode gate

From this documentation checkout:

```bash
mkdir -p /tmp/kimi-k3-stable-baseline
for run in 1 2 3; do
  python models/kimi-k3/tools/kimi-k3-decode-stream.py \
    --url http://172.17.0.2:8000 \
    --model Kimi-K3-MXFP4-DCP8-1M-tail-fix \
    --token-file models/kimi-k3/tools/decode-baseline-256-token-ids.json \
    --prompt-tokens 256 \
    --max-tokens 1024 \
    --output /tmp/kimi-k3-stable-baseline/output-$run.txt \
    --metrics /tmp/kimi-k3-stable-baseline/metrics-$run.json
done
```

This is a completion endpoint test using token IDs directly, `temperature=0`,
`seed=1`, and `ignore_eos=true`. It avoids tokenizer/chat-template changes and
forces every run to emit the same number of tokens.

## Roll back from an experimental HH stack

1. Stop only the active vLLM API process with `SIGTERM`.
2. If CUDA worker contexts survive, restart only the container.
3. Restore the image by registry digest and the two checkpoint source branches.
4. Recreate the `.so` links.
5. Run the exact launch command above.
6. Require the capacity markers and all three decode runs before declaring the
   rollback complete.

Do not point `PYTHONPATH` at HH or SparkInfer `master` during rollback. The
stable order is vLLM checkpoint first, SparkInfer checkpoint second.

The final pre-HH log retained on the original host is:

```text
/mnt/luke/k3-full-mxfp4-dcp8-tail-fix-rerun-20260803T001731Z.log
```

The complete original implementation notes, correctness fixes, truncated-model
workflow, and DCP capacity analysis remain in
[the preceding full-MXFP4 TP16/DCP8 page](full-mxfp4-tp16-dcp8-2026-08-02.md).
