#!/usr/bin/env bash
set -euo pipefail

kimi_image="${KIMI_IMAGE:-local/vllm:kimi-k3-kquant-4p05-tp16-production-20260729}"
kimi_dcp_size="${KIMI_DCP_SIZE:-1}"
kimi_port="${KIMI_PORT:-5670}"
kimi_model_dir="/root/vllm/kimi/Kimi-K3-MXFP4-NF3-4p05-hf"
kimi_kv_cache_bytes="${KIMI_KV_CACHE_BYTES:-952000000}"
kimi_max_batched_tokens="${KIMI_MAX_BATCHED_TOKENS:-32}"
kimi_load_format="${KIMI_LOAD_FORMAT:-safetensors}"
kimi_instanttensor_backend="${INSTANTTENSOR_BACKEND:-BUFFERED}"
kimi_execution_mode="${KIMI_EXECUTION_MODE:-eager}"
kimi_k3_hybrid_decode="${KIMI_K3_HYBRID_DECODE:-1}"
kimi_worker_multiproc_method="${KIMI_WORKER_MULTIPROC_METHOD:-spawn}"
kimi_breakable_cudagraph=0

case "${kimi_load_format}" in
    instanttensor|safetensors) ;;
    *)
        echo "KIMI_LOAD_FORMAT must be instanttensor or safetensors" >&2
        exit 2
        ;;
esac

case "${kimi_execution_mode}" in
    eager)
        kimi_execution_args="--enforce-eager"
        ;;
    full_decode_graph)
        # Only M=1 is captured: this is the latency-critical DCP1 shape and
        # avoids reserving graph pools for unused batch sizes.
        kimi_execution_args="--compilation-config '{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1]}'"
        ;;
    breakable_decode_graph)
        # Capture launch-heavy projections/MoE in graph segments while the
        # attention calls remain eager breakpoints.  This avoids the state
        # corruption observed with K3's monolithic FULL decode graph.
        kimi_breakable_cudagraph=1
        kimi_execution_args="--compilation-config '{\"mode\":0,\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[1]}'"
        ;;
    *)
        echo "KIMI_EXECUTION_MODE must be eager, full_decode_graph, or breakable_decode_graph" >&2
        exit 2
        ;;
esac

case "${kimi_dcp_size}" in
    1) kimi_default_max_model_len=65536 ;;
    2) kimi_default_max_model_len=131072 ;;
    4) kimi_default_max_model_len=262144 ;;
    8) kimi_default_max_model_len=524288 ;;
    16) kimi_default_max_model_len=1048576 ;;
    *)
        echo "KIMI_DCP_SIZE must be one of: 1, 2, 4, 8, 16" >&2
        exit 2
        ;;
esac

kimi_max_model_len="${KIMI_MAX_MODEL_LEN:-${kimi_default_max_model_len}}"
kimi_container="${KIMI_CONTAINER:-kimi-k3-kquant-tp16-dcp${kimi_dcp_size}}"

if docker container inspect "${kimi_container}" >/dev/null 2>&1; then
    if ! docker rm -f "${kimi_container}" >/dev/null; then
        sleep 2
        docker rm -f "${kimi_container}" >/dev/null
    fi
fi

exec docker run -d \
    --init \
    --name "${kimi_container}" \
    --gpus all \
    --ipc=host \
    --network=host \
    --privileged \
    -v /root/vllm/kimi:/root/vllm/kimi:ro \
    -v /root/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro \
    -v /root/vllm/kimi/kimi-k3-jit-cache:/cache/jit \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_LOGGING_LEVEL=INFO \
    -e KDA_DISABLE_AUTOTUNE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e INSTANTTENSOR_BACKEND="${kimi_instanttensor_backend}" \
    -e VLLM_K3_HYBRID_DECODE="${kimi_k3_hybrid_decode}" \
    -e VLLM_WORKER_MULTIPROC_METHOD="${kimi_worker_multiproc_method}" \
    -e VLLM_USE_BREAKABLE_CUDAGRAPH="${kimi_breakable_cudagraph}" \
    --entrypoint bash \
    "${kimi_image}" -lc \
    "unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve ${kimi_model_dir} \
--served-model-name Kimi-K3-MXFP4-NF3-4p05 --host 0.0.0.0 --port ${kimi_port} \
--trust-remote-code --tensor-parallel-size 16 --decode-context-parallel-size ${kimi_dcp_size} \
--distributed-executor-backend mp --max-model-len ${kimi_max_model_len} \
--kv-cache-dtype fp8 --kv-cache-memory-bytes ${kimi_kv_cache_bytes} \
--mamba-cache-dtype bfloat16 --gpu-memory-utilization 0.986 \
--max-num-seqs 1 --max-num-batched-tokens ${kimi_max_batched_tokens} --enable-chunked-prefill \
--no-enable-prefix-caching ${kimi_execution_args} --load-format ${kimi_load_format} \
--chat-template /root/vllm/kimi/kimi-k3-chat-template.jinja \
--generation-config vllm \
--override-generation-config '{\"eos_token_id\":[163586,163588]}'"
