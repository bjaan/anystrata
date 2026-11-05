#!/usr/bin/env bash
# Post-restart verification of upstream merge 1d474dd (hybrid expert streaming --stage-rows /
# --stage-overlap + pointer-table --mmq-tab in the shared engine core). Run after a clean boot:
#   tools/vulkan/post-restart-verify.sh
# Green result = the merge stands: push it, update docs, restart the serve.
# Any canary DIFFERS = revert 1d474dd on the spot and bisect upstream's five commits.
set -uo pipefail
cd "$(dirname "$0")/../.."
M=models/Qwen3.6-35B-A3B-GSQ-hybrid-GGUF/Qwen3.6-35B-A3B-GSQ-hybrid.gguf
IDS="220,18,24,16,13,271,48,25,3437,369,220,16,22,2942,220,17"
REC="18,30,198,32,25,220,18,24,16,13,271,48,25,3437,369,220"   # recorded canary, max-new 16 (PORT-STATUS)

echo "== rebuild (catches any dirty tree)"
cmake --build build-vk -j2 2>&1 | grep -cE " error" | xargs -I{} echo "build errors: {}"

echo "== unit/integration suite"
ctest --test-dir build-vk 2>&1 | grep -E "tests passed" || { echo "ctest FAILED"; exit 1; }

echo "== canary gate (max-new 16)"
out=$(tools/vulkan/leg.sh --gate engine timeout 420 build-vk/bin/anystrata-gguf-vk \
      --model "$M" --max-context 4096 --tokens "$IDS" --max-new 16 2>/dev/null \
      | grep -E "^[0-9]+(,[0-9]+)+$" | head -1)
echo "got: $out"
if [ -z "$out" ]; then
    echo "CANARY: NO OUTPUT (load timed out or crashed) — inconclusive, re-run on a quieter box"; exit 2
elif [ "$out" = "$REC" ]; then
    echo "CANARY: bitwise-identical — the merge is inert on this backend, as required"
else
    echo "CANARY: DIFFERS — do NOT keep the merge: git revert 1d474dd, then bisect 29511cf^..3d5fc52"
    exit 1
fi

echo "== swap-regime canary (the regime the 16-id full-residency pass cannot see)"
out2=$(tools/vulkan/leg.sh --gate engine timeout 420 build-vk/bin/anystrata-gguf-vk \
      --model "$M" --max-context 4096 --vram-reserve-mib 11000 --tokens "$IDS" --max-new 16 2>/dev/null \
      | grep -E "^[0-9]+(,[0-9]+)+$" | head -1)
echo "got: $out2"
if [ -z "$out2" ]; then echo "SWAP CANARY: no output (load timed out) — inconclusive"; exit 2
elif [ "$out2" = "$REC" ]; then echo "SWAP CANARY: bitwise-identical (Task 6.0 regime holds)"
else echo "SWAP CANARY: DIFFERS — a core change broke the swap regime (head-capture staleness, Task 6.0 history)"; exit 1; fi

echo "== perf leg (max-new 96) for the docs table"
tools/vulkan/leg.sh --gate engine timeout 420 build-vk/bin/anystrata-gguf-vk \
    --model "$M" --max-context 4096 --tokens "$IDS" --max-new 96 2>/dev/null \
    | grep -E "loaded in|generated" | head -3
echo "VERIFY-OK: push 1d474dd, document --stage-rows/--mmq-tab, restart the serve"
