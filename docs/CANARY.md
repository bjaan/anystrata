# The canary

Every build of this engine gets asked a question before it gets asked anything else:

```
17*23?
```

It must answer **391**. That is the whole test. We call it the canary, after the bird carried
into the mine: it does not measure how fast or how smart the engine is — it tells you, in about
forty seconds, whether the air is still breathable.

## Why a multiplication

A wrong kernel almost never announces itself. A quantized dot product with a mis-ordered
accumulation, a KV cache row off by one element, a slot table that went stale mid-replay — all
of these leave the model chatting fluently and confidently. What they do *not* leave intact is
arithmetic. 17 × 23 is a question with exactly one right answer, no round-number pattern the
model can talk its way into (not 20×20, not 15×15), and an answer a human can verify at a
glance. The canary prompt is fed as raw token ids — `220,18,24,16,13,271,48,25,3437,369,220,16,
22,2942,220,17` ("17", "×", "23", "?", twice) — so the tokenizer can never be part of the
variable either.

## What it proves

One canary run crosses the entire stack: GGUF load, the expert cache and its streaming, the
mmvq and gemv kernels, attention and the KV stores (fp16 or q8_0), the sampler, the
detokenizer. If every stage is bit-clean, the 16 generated ids are *identical*, build to build,
forever. That is the actual test: not "does it say 391" but "does the whole id list match the
recorded one, byte for byte". A port, a tuning pass, or a capture change that shifts a single
float is caught here as a diff, with no statistics and no tolerance.

During the Vulkan bring-up the canary did double duty as the work queue: each run died at the
first kernel not yet ported, naming it — *"strata-vk: `native_mmvq` has no Vulkan
implementation yet"* — and the list of halt points marching forward *was* the port's progress
bar. When the halt line disappeared and 391 came out, that was the day the BC-250 (a repurposed
PlayStation 5 chip, no CUDA, no ROCm) ran this engine end to end for the first time.

The canary also keeps catching the subtle stuff. The open bug in the ledger — captured graphs
meeting live expert-swapping under the lookup guesser — was found because the canary's ids
collapsed to zeros from the ninth token. It never crashed; it just stopped being able to
multiply. This is also why `391` is engraved on the mascot's hairpin: it is the first thing she
computed on the die she was born on.

## Silly, and the 18-byte crate

The mascot's companion is **Silly**, a round teal seabird who hauls an 18-byte crate in its
beak. The crate is not decoration: 18 bytes is the size of one Q2_0 quantized block — 32 weights
at 2 bits plus its fp16 scale, the format most of a Mixture-of-Experts model's routed experts
are stored in. Every canary run on this repo is Silly carrying those crates onto the graphics
card, one expert at a time, while the engine decides which of the 10,240 of them fit. The bird
is a skillfish species, which is the BC-250 easter egg (the die codename is *Cyan Skillfish*) —
but the crate is universal: any GPU that runs this engine runs the same 18 bytes.

## Running it

From the repo root, on a box with a card and the GSQ-hybrid model:

```bash
anystrata/tools/vulkan/leg.sh --gate engine \
  build-vk/bin/anystrata-gguf-vk \
  --model /models/Qwen3.6-35B-A3B-GSQ-hybrid.gguf \
  --max-context 4096 \
  --tokens "220,18,24,16,13,271,48,25,3437,369,220,16,22,2942,220,17" \
  --max-new 16
```

The last line of output is the generated ids; compare them against the recorded canary in
`vulkan/PORT-STATUS.md`. Identical: ship it. Differ: the mine air went bad somewhere between
this commit and the last one, and the diff says which token first noticed.

There is a bigger, slower sibling: the teacher-forced correctness gate (`bench/correctness/`,
thousands of steps scored against llama.cpp's own spread). That one judges *quality*. The
canary judges *sameness*. You need both, but only one of them runs in forty seconds, so it
runs after everything.

391. That's it. That's the test.
