# The pack format — the canonical weight form the engine reads

> **Reconstructed 2026-10-30.** This file was cited by the engine's headers and kernels for as long as
> the tree has existed (`src/kernels/cuda/dequant_s2.cu`, `include/strata/kernels/s_gemv.hpp`,
> `include/strata/core/weights.hpp`…) but a copy never made it into this fork's history, and the current
> upstream Strata docs tree does not carry it either. Everything below is transcribed from the code's
> own statements — the plane layouts the parity tests check against, the §-numbers the headers cite, and
> the pack writer's documented behaviour (`tools/iq_pack.py`, `tools/pack_index.py`) — so the citations
> resolve and a reader can check each claim against an executable reference. Where the code does not
> pin a number, this file says so rather than inventing one.

## 1. What a pack is

A pack is the artifact `tools/iq_pack.py` writes once, offline, from a GGUF: quantized experts kept in
their GGUF slices (`experts.bin` + `native_experts.txt` when `--experts-bin`), the dense tensors and
float planes re-laid in `dense.bin` (or read as-is with `--native`), and `index.txt` as the flat table
the engine loads (why an index file and not JSON: `include/strata/core/weights.hpp`). A pack directory
is complete only when `native_experts.txt` — written last — exists.

The ENGINE may transform what the pack holds when it loads (weights.hpp): BF16 sources re-round from
the pack's lossless 32-bit promotion back to 16; fp16 scale planes widen exactly to f32. Pack and
engine are deliberately different documents; this file describes the PACK.

## 2. The canonical form: three code widths, one decode

13 source quant types collapse to three code widths (2/4/8 bits) plus a handful of attributes, carried
per tensor in the manifest as `SForm` (`include/strata/kernels/s_gemv.hpp`):

```
    value = cb[code] * scale + offset
```

* `code_bits`: 2, 4 or 8.
* `code_bias`: added to the code **in the integer domain, before the single multiply** (usually -1;
  Q8_0 uses 0 with an int8 code).
* `group_elems`: elements per scale (and per offset, when present).
* `codebook`: `Affine` (`code + bias`) or `Iq4Nl` (the 16-value non-linear table `kvalues_iq4nl`).
* `has_offset`: only the K-quants with a min (Q4_K, Q5_K).

The kernel takes these as ARGUMENTS, not as 13 specialisations — that uniformity is the point of the
form (s_gemv.hpp).

### 2.4 Scale-plane width

Scale planes are NOT always 4 bytes: 90 of the 303 quantized tensors carry **fp16** scales in the pack
(all 58 Q2_0 expert tensors, plus Q4_0, Q5_0, Q8_0 and IQ4_NL), because that is what the source block
holds (§2.4 is the section the loaders and kernels cite for this). The manifest flag is `scales_fp16`;
assuming 4 bytes per scale instead of reading it is what once made the plane arithmetic disagree with
the pack for a whole round (`include/strata/core/weights.hpp`).

## 3. S2 — Q2_0's canonical form (the hot one)

S2 is what 31.64 GiB of the artifact is, and the hottest decode in the engine
(`src/kernels/cuda/dequant_s2.cu`, `include/strata/kernels/dequant_s2.hpp`).

Plane layout (separate planes, not interleaved blocks):

* `codes`: `n_blocks * 16` bytes — 64 elements a block, **4 codes per byte LSB-first**: element `i`
  at byte `i/4`, bits `(i%4)*2` (Q2_0's own convention, kept for every form).
* `scales`: `n_blocks` floats (f32 in the engine's form; §2.4 says what the PACK stores).
* a tensor `[n_in, n_out]` with `n_in` CONTIGUOUS (GGUF's `ne[0]` varies fastest): row `o` is a
  contiguous run of `n_in/64` blocks (s2_gemv.hpp).

### 3.1 The exactness rule

Decode is `(code - 1) * scale` with the subtraction applied to the CODE, **inside the integer
domain**, then ONE multiply:

```
    y[j] = (float)(code - 1) * d;      // the -1 is applied to the CODE, then ONE multiply
```

This is what makes the decode bit-exact and why the dequant parity tests run at ZERO tolerance
(dequant_s2.cu §3.1; `vulkan/src/tests/dequant_s2_vk_parity.cpp` re-transcribes exactly this rule as
its host reference). Any float-domain subtraction (`float(code) - 1.f`) rounds differently and is the
wrong kernel.

## 4. The S family (S2/S4/S8)

One decode for all three widths — that is the point of §2:

* `codes`: `n_out * (n_in / (8/code_bits))` bytes, the same LSB-first packing at the type's width.
* `scales`: `n_out * (n_in / group_elems)` floats; `offset` the same count, or absent (s_gemv.hpp).

Q8_0's form is the affine 8-bit case with `code_bias = 0` over int8 codes, one fp16 (pack) / f32
(engine) scale per 32 elements — the same plane discipline as §2.4.

## 5. What the pack does NOT contain

* No Q2_0 re-expression of the i-quants: an IQ file's pack keeps every quantized tensor in its GGUF
  form (`tools/iq_pack.py`).
* No geometry or semantics: the loader reads the flat index and knows nothing about what a tensor
  MEANS (weights.hpp); the caller supplies forms and shapes.
* Conversions are recorded, never hidden: every converted tensor goes into `conversions.json` with
  its method, exactness flag, largest absolute error and the source bytes' SHA-256 (iq_pack.py).
