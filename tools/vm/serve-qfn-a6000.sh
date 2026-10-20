#!/bin/bash
# Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS on the A6000 VM (VM 220) through the UnbiasedStrata 2 fork, fresh tree build.
# Engine flags as the step-6 record on this card (bench/unbiased/… a6000 flash-next step6): upstream's 2026-10-17
# config for the model plus the 48 GB choices (--resident-experts, --ple-io mmap, --kv int8, --kv-resident 32768).
# Pictures: the front end takes --mmproj (the model's BF16 projector, encoded on the CPU) and the engine --vision.
# The front end comes from the CUDA-free build directory (make frontend), so a front-end fix needs no engine rebuild.
D=/mnt/ds4/unbiased-strata
T=~/us2-fresh/UnbiasedStrata
B=$T/build/bin
F=$T/build-frontend/bin/unbiased-strata; [ -x "$F" ] || F=$B/unbiased-strata
S1=$D/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
S2=$D/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
MMPROJ=$D/mmproj-Qwen3.8-Flash-Next-BF16.gguf
for f in $S1 $S2 $D/pack-iq3-xxs/dense.bin $D/mtp/experts.bin $MMPROJ; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
export CUDA_MODULE_LOADING=EAGER
mkdir -p ~/us2-qfn; echo "$(date -u +%FT%TZ) start qfn ctx ${QFN_CTX:-65536} port 18090 frontend $F" >> ~/us2-qfn/starts.log
exec $F --engine $B/strata -m $S1 --host 127.0.0.1 --port 18090 --alias qwen3.8-flash-next-iq3_xxs-a6000 --sse-ping-interval 10 \
  --mmproj $MMPROJ --threads ${QFN_THREADS:-6} \
  -- --vision --pack $D/pack-iq3-xxs --native $S1 --ple-gguf $S2 --expert-profile $D/pack-iq3-xxs/expert-profile.bin \
  --resident-experts --prefill auto --spec 4 --spec-min-p 0.5 --mtp $D/mtp \
  --max-context ${QFN_CTX:-65536} --kv int8 --ple-io mmap --kv-resident 32768
