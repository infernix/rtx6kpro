#!/usr/bin/env bash
set -euo pipefail

kimi_port="${KIMI_PORT:-5670}"
kimi_base_url="http://127.0.0.1:${kimi_port}"
kimi_model="${KIMI_MODEL:-Kimi-K3-MXFP4-NF3-4p05}"
kimi_expected_max_model_len="${KIMI_EXPECT_MAX_MODEL_LEN:?set KIMI_EXPECT_MAX_MODEL_LEN}"

health_code="$(curl --max-time 10 -sS -o /dev/null -w '%{http_code}' "${kimi_base_url}/health")"
test "${health_code}" = "200"

models_response="$(curl --fail-with-body --max-time 10 -sS "${kimi_base_url}/v1/models")"
jq -e --arg model "${kimi_model}" --argjson max_len "${kimi_expected_max_model_len}" \
    '.data[] | select(.id == $model and .max_model_len == $max_len)' \
    >/dev/null <<<"${models_response}"

run_chat_check() {
    local prompt="$1"
    local expected="$2"
    local max_tokens="$3"
    local response

    response="$(curl --fail-with-body --max-time 600 -sS \
        "${kimi_base_url}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -cn \
            --arg model "${kimi_model}" \
            --arg prompt "${prompt}" \
            --argjson max_tokens "${max_tokens}" \
            '{model:$model,messages:[{role:"user",content:$prompt}],
              chat_template_kwargs:{thinking:false},temperature:0,
              max_tokens:$max_tokens,stop:["<|close|>"]}')")"

    jq -e --arg expected "${expected}" '
        (.choices[0].message.content | gsub("^\\s+|\\s+$"; "")) == $expected
        and .choices[0].finish_reason == "stop"
        and .choices[0].stop_reason == "<|close|>"
    ' >/dev/null <<<"${response}"
    jq -c '{content:.choices[0].message.content,
            prompt_tokens:.usage.prompt_tokens,
            completion_tokens:.usage.completion_tokens,
            stop_reason:.choices[0].stop_reason}' <<<"${response}"
}

printf 'health=%s max_model_len=%s\n' "${health_code}" "${kimi_expected_max_model_len}"
run_chat_check 'What is 1 + 1? Answer with only the number.' '2' 8
run_chat_check 'What is the capital of France? Answer in one word.' 'Paris' 8
run_chat_check \
    'List the first five prime numbers separated by commas. Output nothing else.' \
    '2, 3, 5, 7, 11' 16
