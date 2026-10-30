# The activation contract — which quantized image of the activation each weight form reads

> **Reconstructed 2026-10-30**, the same way as `docs/pack-format.md`: cited by `src/core/layer.cpp`,
> `src/kernels/bf16_gemv_parity.cpp`, `src/kernels/cpu/expert_parity.cpp` and `src/kernels/cuda/ple.cu`
> since long before, never carried into this fork's history. Everything below is transcribed from the
> code's own statements, checked against the kernels and the guards that enforce them.

## The rule, per weight type

```
    code_bits == 2           ->  Q8_0,  via `quantize_q8_0` + `s2_gemv_q8`
    anything else quantized  ->  Q8_K,  via `quantize_q8_K` + `s_gemv_q8k_split`
```

(`src/core/layer.cpp`'s gemv_quantized: the comment that opens the dispatch, quoted verbatim.)

## Why it is a CONTRACT and not a detail

The activation is quantized once and can be handed to a GEMV in two shapes; picking wrong changes
the numbers by **0.6-1.4% on the GEMV** — a plausible vector, not a broken one — which is exactly
what Gate C1 exists to find (layer.cpp). Two weight types that look identical on every other
attribute differ ONLY in which activation they consume: Q5_0 and Q5_K are both 8-bit with bias -16
(only `has_offset` differs); IQ4_NL and IQ4_XS are identical in everything `SForm` otherwise holds.
So the choice CANNOT be derived from the form's other fields: it is carried from the manifest's
`source_type` as `act_kind` (0 = Q8_0, 1 = Q8_K) — `include/strata/kernels/s_gemv.hpp`, and
`tools/pack_index.py` writes the column. `s_gemv_q8k`'s own guard (`code_bits 2 has no Q8_K
contract`) is what caught a real violation (LEDGER L54; `src/kernels/cuda/s_gemv.cu`).

## The two shapes

* **Q8_0** — int8 codes, one f32 scale per 32 elements (`ggml_q8_0` block). `quantize_q8_0`
  produces it; `s2_gemv_q8` consumes it.
* **Q8_K** — 292 bytes per 256 elements: `{ f32 d ; int8_t qs[256] ; int16_t bsums[16] }`.
  `bsums` exists for ggml's AVX2 dot product and the engine's kernels do not read it, but the
  STRIDE includes it, so a buffer from `quantize_q8_K` passes straight in
  (`src/kernels/cuda/s_gemv.cu`, Q8K_BLOCK_BYTES = 292).

## For a BF16 (unquantized) weight

A BF16 tensor's activation is the BF16 image itself — the contract for that case is stated at
`src/kernels/cuda/ple.cu:173` (with both sides bf16 the products are exact); a caller that widens
one side and not the other breaks the same way the wrong act_kind does.

## Callers must produce BOTH images

`x80` and `xq8k` are the two quantized images of the SAME activation; a caller produces both once
and `gemv_quantized` picks (layer.cpp). Producing only the one it thinks it needs is how the
assumption gets baked in again.
