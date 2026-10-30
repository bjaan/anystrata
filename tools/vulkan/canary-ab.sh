#!/usr/bin/env bash
# Task 4.2 model legs (this session): canary eager-vs-graphs A/B, then the graphs-default run
# whose GPU-side timings the new event layer now reports honestly. Canary ids from PORT-STATUS
# (the /tmp/qids.txt gate is gone from /tmp; re-tokenized from the same ChatML shape on the
# hybrid file's tokenizer).
set -u
M=~models/Qwen3.6-35B-A3B-GSQ-hybrid-GGUF/Qwen3.6-35B-A3B-GSQ-hybrid.gguf
QIDS='248045,846,198,12195,369,220,16,22,830,220,17,18,30,21134,440,279,1324,1132,13,248046,198,248045,74455,198'
B=~anystrata/build-vk/bin/anystrata-gguf-vk
OUT=~anystrata/bench/vk-task42
mkdir -p "$OUT"
run() {
  local tag="$1"; shift
  echo "== $tag $(date +%H:%M:%S)"
  ~anystrata/tools/vulkan/leg.sh --gate engine \
    "$B" --model "$M" --max-context 4096 --tokens "$QIDS" --max-new 16 "$@" \
    > "$OUT/$tag.log" 2>&1
  echo "rc=$?"; grep -E 'generated|ids|tok/s|391' "$OUT/$tag.log" | tail -4
  sleep 20   # cooldown between legs (PORT-NOTES law 10)
}
run graphs
run eager --no-graphs
echo "== compare"
grep -o 'generated[^)]*' "$OUT/graphs.log" > "$OUT/graphs.ids" 2>/dev/null
ids() { grep -oE '220,18,24[^ ]*|generated ids[^ ]*|\[([0-9,]+)\]' "$OUT/$1.log" | tail -1; }
ids graphs > "$OUT/g.id"; ids eager > "$OUT/e.id"; cat "$OUT/g.id" "$OUT/e.id"
cmp -s "$OUT/g.id" "$OUT/e.id" && echo "CANARY A/B: IDENTICAL" || echo "CANARY A/B: DIFFERENT (see $OUT)"
