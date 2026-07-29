#!/usr/bin/env bash
set -euo pipefail

# Validated Kimi-K3 DCP1 latency configuration.  The smaller KV allocation
# leaves 0.29 GiB/GPU for the breakable graph pool; 512 MB supports 34,105
# cache tokens and therefore the declared 32,768-token model length.
export KIMI_IMAGE="${KIMI_IMAGE:-local/vllm:kimi-k3-kquant-dcp1-speed-20260729}"
export KIMI_CONTAINER="${KIMI_CONTAINER:-kimi-k3-dcp1-speed}"
export KIMI_PORT="${KIMI_PORT:-5670}"
export KIMI_DCP_SIZE=1
export KIMI_MAX_MODEL_LEN="${KIMI_MAX_MODEL_LEN:-32768}"
export KIMI_KV_CACHE_BYTES="${KIMI_KV_CACHE_BYTES:-512000000}"
export KIMI_LOAD_FORMAT="${KIMI_LOAD_FORMAT:-instanttensor}"
export KIMI_EXECUTION_MODE="${KIMI_EXECUTION_MODE:-breakable_decode_graph}"
export KIMI_WORKER_MULTIPROC_METHOD="${KIMI_WORKER_MULTIPROC_METHOD:-forkserver}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/run-kimi-k3-kquant-tp16-dcp.sh"
