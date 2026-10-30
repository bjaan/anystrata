#!/bin/sh
# tools/vulkan/serve-swift27.sh — serve ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_XS on the
# BC-250 through llama.cpp's Vulkan backend (the tree's own pinned submodule, built with
# -DGGML_VULKAN=1). This is NOT anystrata-gguf-vk: the qwen35-dense architecture has no Strata
# engine family on any backend (docs/VULKAN.md, plan Phase 7); this script is the today-path,
# kept beside the port so the API the fork serves (OpenAI chat) looks the same.
#
# Measured on this card (llama-bench, -ngl 99): prefill 87 tok/s, decode 15.5 tok/s — dense
# 27B touches every weight per token; MoE models are the fast path on 24 CUs.
#
# Do not run while anystrata-gguf-vk holds Qwen3.6-35B in the unified 15.7 GiB: both loads
# together thrash the 14 GiB of system RAM. Stop one before starting the other.
set -e
HERE=$(dirname "$0")
REPO=$(cd "$HERE/../.." && pwd)
MODEL=${SWIFT27_GGUF:-$HOME/models/Swift-1.5-Qwen3.8-27B-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf}
LLAMA=${LLAMA_VK_BIN:-$HOME/llama-vk/bin/llama-server}
[ -f "$MODEL" ] || { echo "no model at $MODEL (set SWIFT27_GGUF)"; exit 1; }
[ -x "$LLAMA" ] || { echo "no llama-server at $LLAMA (set LLAMA_VK_BIN; build: cmake -DGGML_VULKAN=ON on third_party/llama.cpp)"; exit 1; }
exec "$LLAMA" \
    -m "$MODEL" \
    --device Vulkan0 -ngl 99 \
    -c "${CTX:-16384}" --flash-attn on -kv q8_0 \
    --jinja --alias swift-1.5-qwen3.8-27b \
    --host 127.0.0.1 --port "${PORT:-18190}"
