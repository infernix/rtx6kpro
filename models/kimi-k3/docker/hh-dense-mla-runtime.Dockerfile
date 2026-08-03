FROM voipmonitor/vllm@sha256:2c99435142dd10f85834eaf4c490cb3d4095318152f0cc4fb38c7623d7edb7ac

LABEL org.opencontainers.image.title="Kimi-K3-HH-DenseMLA-DCP8-InstantTensor" \
      org.opencontainers.image.description="Pinned runtime for Kimi-K3 full MXFP4 TP16/DCP8 with HH vLLM, SparkInfer dense MLA, and InstantTensor source overlays" \
      org.opencontainers.image.version="2026-08-03" \
      local-inference-lab.runtime.base-digest="sha256:2c99435142dd10f85834eaf4c490cb3d4095318152f0cc4fb38c7623d7edb7ac" \
      local-inference-lab.vllm.branch="codex/kimi-k3-hh-dense-mla-dcp8-20260803" \
      local-inference-lab.vllm.commit="99506ed20241ad47a269247f691c902c2bf1f6b6" \
      local-inference-lab.sparkinfer.branch="codex/kimi-k3-hh-dense-mla-dcp8-latest-20260803" \
      local-inference-lab.sparkinfer.commit="f39c6bf26be9d92b65d1f031819289c8c1f084a1" \
      local-inference-lab.sparkinfer.production-commit="a84463014bba9933e69c67da0f8a983f9b1e149f" \
      local-inference-lab.model.snapshot="moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970" \
      local-inference-lab.instanttensor.version="0.1.9+consumer1"
