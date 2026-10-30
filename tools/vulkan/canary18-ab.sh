#!/usr/bin/env bash
# The canonical 18-token canary A/B (the ids PORT-STATUS has tracked since Phase 2: the
# generated answer '220,18,24,16,13,271,48,25,3437,369,220,16,22,2942,220,17' = " 391.\nQ:
# 17 x 23"), full-residency regime, speculation ON (the default arm): graphs (default) vs
# --no-graphs. Run through leg.sh's engine gate with the 20 s cooldown (PORT-NOTES law 10).
# (canary-ab.sh is the 24-token variant used for the 2026-10-30 Task-4.2 model gate.)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)           # tools/vulkan
REPO=$(cd "$HERE/../.." && pwd)               # the repo root
M=$HOME/models/Qwen3.6-35B-A3B-GSQ-hybrid-GGUF/Qwen3.6-35B-A3B-GSQ-hybrid.gguf
QIDS='248045,846,198,12195,369,220,16,22,830,220,17,18,30,248046,198,248045,74455,198'
B=$REPO/build-vk/bin/anystrata-gguf-vk
OUT=$REPO/bench/vk-canary18
mkdir -p "$OUT"
run() {
  local tag="$1"; shift
  echo "== $tag $(date +%H:%M:%S)"
  "$HERE/leg.sh" --gate engine \
    "$B" --model "$M" --max-context 4096 --tokens "$QIDS" --max-new 16 "$@" \
    > "$OUT/$tag.log" 2>&1
  echo "rc=$?"; grep -E 'prompt .*tokens|generated' "$OUT/$tag.log" | tail -2
  grep -oE '[0-9]{1,6}(,[0-9]{1,6}){15,}' "$OUT/$tag.log" | tail -1 > "$OUT/$tag.ids"
  cat "$OUT/$tag.ids"
  sleep 20
}
run graphs
run eager --no-graphs
echo "== compare"
cmp -s "$OUT/graphs.ids" "$OUT/eager.ids" && echo "CANARY A/B IDENTICAL: $(cat $OUT/graphs.ids)" || { echo "CANARY A/B DIFFER"; diff "$OUT/graphs.ids" "$OUT/eager.ids"; }
