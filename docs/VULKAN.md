# Vulkan: the BC-250 backend (`vulkan/`)

The third GPU backend of this fork, and the one that runs the AMD **BC-250** (Cyan Skillfish, `gfx1013`: an
RDNA-1.x-class APU with 16 GB unified GDDR6). Like `sycl/` it is its own project that takes the engine's sources
from `src/` and carries migrated device code beside it, so the CUDA and HIP trees stay byte-for-byte upstream's.

Why Vulkan and not ROCm/HIP on this card: ROCm's userspace (rocBLAS/Tensile) ships no `gfx1013` kernels and the
HIP runtime does not target it, while the Mesa **RADV** driver is first-class (Vulkan 1.4). llama.cpp's own
Vulkan backend is what people run models on this board with; this page is about running *Strata's* engine there.

## Status (first increment, 2026-10-21 — the ground it was built on)

`vulkan/` builds, runs, and passes its tests on real BC-250 hardware (Mesa RADV, kernel 7.2.11):

```
cmake -S vulkan -B build-vulkan && cmake --build build-vulkan && ctest --test-dir build-vulkan
```

* `vk_device_probe` — **PASS**: the device's two load-bearing capabilities are asserted by execution, not by
  extension presence. `OpUDot` (the `dp4a` every quantized kernel uses; `ggml_cuda_dp4a` appears 109 times in
  `src/`) computes 10 for (1,2,3,4)·(1,1,1,1), and a compute pipeline **forced to a 32-wide subgroup** on this
  wave64 hardware returns 321 for the ballot-popcount + `subgroupShuffleXor` probe. Both modules are assembled
  from `.spvasm` with `spirv-as` at build time: this distro's `glslc` frontend predates
  `GL_EXT_shader_integer_dot_product`, and features present-but-not-enabled crash RADV at pipeline create —
  `vk_device.cpp` enables `shaderIntegerDotProduct`, `subgroupSizeControl` and `computeFullSubgroups` explicitly.
* `s2_gemv_vk_parity` — **PASS, 640/640 rows bit-exact** at the engine's real shapes (n_in 2560, n_out 640). The
  `.comp` is a line-by-line translation of `src/kernels/cuda/s2_gemv.cu` — "the expert matvec is the hottest
  thing in the engine" per its header, 48 layers × 10 experts × 3 roles per token. Same naive shape (dequantize
  on the fly, FP32 accumulation per row, the same integer-domain `(code-1)`, the same term order); only the
  spellings change: `blockIdx*blockDim+threadIdx` -> `gl_GlobalInvocationID.x`, and 16-bit storage becomes
  packed `uint` + `unpackHalf2x16` the way llama.cpp's own Vulkan `*_packed32` block structs do.

* `dequant_s2_vk_parity` — **PASS, 262,144 elements, 0 mismatches at ZERO tolerance** (4096 blocks).
  `dequant_s2.comp` is `cuda/dequant_s2.cu` line by line — the decode S2 experts ride on, the tree's own
  header calls "the single hottest decode in the engine"; bit-exactness is the CUDA side's rule because
  `(code - 1)` in the integer domain then ONE multiply cannot round.

## Context sizes on a unified-memory APU

The engine sizes its device arena at startup, and on the BC-250 (15.7 GiB shared) the model's
weights leave room for a KV cache of about a gigabyte and a half beyond the expert cache:

| `--max-context` | KV (fp16) | resident | status |
| --- | --- | --- | --- |
| 16,384 | 0.65 GiB | ~14.1 GB | proven: every long gate, pi round trips |
| 32,768 | 1.28 GiB | ~13.9 GB | proven stable (32768 gate 2026-10-27) |
| 65,536 | 2.56 GiB fp16 / 1.28 GiB `--kv q8_0` | ~13.8 GB | **proven: `--max-context 65536 --kv q8_0` serves on this card** (law 11's unlock, 2026-10-28: store, prefill-read and fused decode-read all ported and parity-tested; the q8_0 arm costs ~1.82x the llama.cpp KL floor — the storage trade, documented) |
| 131,072 | 5.1 GiB fp16 | — | not this card; the pod machines' territory |

The q8_0 cache stores K and V rows as int8 codes with one fp16 scale per 32 values (544 bytes a
row where fp16 costs 1,024): the engine's `--kv q8_0` switch, the same arm CUDA's correctness
harness scores as `kvq8`. See `vulkan/PORT-STATUS.md`
("q8_0 KV cache") and the driver rules in `vulkan/PORT-NOTES.md` 10-11.

## What the backend provides

`vulkan/src/runtime/vk_device.cpp` / `vk_runtime.cpp` + `vulkan/include/strata/vulkan_runtime.hpp`:

* **Device selection with a named reason**: a card without `VK_KHR_shader_integer_dot_product` or
  `VK_EXT_subgroup_size_control` (or that cannot run a 32-wide subgroup) fails at start instead of producing
  wrong tokens — the rule `src/core/device.cu` applies to CUDA/HIP archs. `STRATA_VK_DEVICE=<substring>` picks
  the card.
* **Arena**: one bump region chain like CUDA's `DeviceArena`, but a region larger than the driver's
  `maxMemoryAllocationSize` (512 MiB on RADV) becomes a chain of buffers; an arena address is an offset, and a
  kernel argument or copy that straddles a piece throws with the offset and size named. The planner's regions
  are sized to pieces, exactly like CUDA's are sized to cudaMallocs.
* **Streams/events/copies**: a single-queue first cut (the BC-250 has one GPU): `cudaStream_t` -> the default
  queue, copies staged in chunks, host-visible coherent memory (it is an APU — `uma=1`, no PCIe tier).
* **Kernel registry**: `kernel_load(name, spv, buffers, pushbytes, workgroup, force_subgroup32)` +
  `kernel_launch_sized(...)`; `.comp` -> `glslc` -> embedded `uint32_t[]` (`vulkan/tools/embed_spv.py`), no run-time
  file access, matching the fork's no-downloads rule.

## The port plan for the rest of the engine

The route is proven by the S2 GEMV parity test; the work is applying it 321 kernels and 99 host files deep:

1. **Device kernels** (`src/kernels/cuda/*.cu` -> `vulkan/src/kernels/*.comp` + `*_vk.cpp`): the naive P2-era
   kernels (gemv/dequant/rope/sampler/elementwise family) translate mechanically. The ones using CUDA-only
   hardware features map as follows:
   * `__shfl_*` -> subgroup ops with the pipeline forced to 32 lanes (proven above);
   * `__dp4a` / `ggml_cuda_dp4a` -> `OpUDot`/`OpSDot` (proven);
   * `__ldg` -> plain loads (RADV's cache policy handles it); `cp.async`/shared-memory staging ->
     `OpGroupNonUniform` + `OpVariable Workgroup` (the workgroup's 64 KiB LDS matches RDNA3's, so the HIP
     kernels' shared-memory budgets fit);
   * WMMA/tensor-core kernels (`wmma`, `mma.`: 34+16 sites, and QSA's matrix instructions) have no RADV
     counterpart on gfx1013 (`VK_KHR_cooperative_matrix` absent) — these fall back to the ordered-FP32
     paths the HIP port already carries for non-tensor-core parts (docs/AMD_HIP.md says exactly this for
     "CUDA-only QSA matrix instructions"), or to hipBLAS-free host GEMM for prefill in the first cut.
2. **Host files** (`src/core/*.cpp`, `src/prefill/...`): compiled against a `vulkan_compat/` header layer that
   maps the CUDA runtime surface onto this backend's arena/queue API (the `hip_compat/` idea: `cudaMalloc` ->
   `Arena::alloc`, `cudaMemcpy*` -> `copy_h2d/d2h`, `cudaStream_t` -> `void*` stream, `cudaEvent_t` -> event).
   `device.cu`, `pinned.cu`, `vmm.cpp` need Vulkan-native replacements: UMA means the pinned/PCIe tier
   collapses (the engine's expert-cache policy keeps working — it plans against `cudaMemGetInfo`, here the
   memory-budget query `vk_device.cpp` prints at start).
3. **cuBLAS**: the engine uses cublas for the dense projections; on gfx1013 there is no BLAS library through
   this driver. The dequant-then-GEMV/GEMM kernels in `src/kernels/cuda/` (the same ones the HIP build compiles
   for prefill, `STRATA_PREFILL_MMQ`) cover prefill without it, or link against ggml's CPU/Vulkan matmul from
   the submodule.
4. **Engine binary**: a `strata-vk` target beside `strata`/`strata-poly`, same `--serve` GENI protocol, so the
   `unbiased-strata` front end and its picture cache are unchanged.

A second measured baseline, 2026-10-22 (same build): **ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_XS** (a `qwen35`-arch DENSE+SSM 27B, 7.8 GiB) on this card: prefill 87/86 tok/s (512/2048), decode 15.5 tok/s. Five times slower than Qwen3.6-35B-A3B because it is dense — every 27B weight touches every token — while the 35B-A3B activates ~3B. Neither `strata-poly` nor any Strata engine runs the `qwen35`-27B architecture at all (the dispatch table at src/poly/engine.cpp:992 knows qwen35moe / glm5-next / gemma4 / deepseek4 / lfm2moe; `strata` knows qwen4exp = Flash-Next), so such a model today means llama.cpp's Vulkan backend. The lesson for this board: MoE models are the play on 24-40 CUs with 16 GiB unified.

Measured baseline while the engine port proceeds — llama.cpp's **Vulkan** backend (the tree's own
`third_party/llama.cpp` submodule, built with `-DGGML_VULKAN=1`) on this board, Qwen3.6-35B-A3B
UD-IQ2_XXS `-ngl 99`: prefill 383-375 tok/s (512/2048), decode 76 tok/s, `fp16: 1, uma: 1, warp 32,
shared 64K`. Strata's engine on top of this backend is what turns that into expert-offload speeds.

## Status now (what has landed since the first increment)

The engine runs models end-to-end on this backend — Qwen3.6-35B-A3B (qwen35moe) at 16K/32K/64K-q8
serve configs, all experts resident — with captured graphs the default (Phase 4.2, bitwise-inert),
deferred-submit batching (Phase 4.1), the q8_0 KV chain ported, and the teacher-forced correctness
gate scored at 1.39x llama.cpp's own floor (Phase 5). The full ledger is `../vulkan/PORT-STATUS.md`
(repo path `vulkan/PORT-STATUS.md`); the driver laws that cost a session each are in
`vulkan/PORT-NOTES.md`.

## Known limits (current)

* One queue still (the BC-250 has one GPU): copies and syncs stay EAGER (the staging-hazard law),
  launches batch into pending CBs and replays; streams/pipelining beyond that are not on the table
  for a single-queue APU.
* Events carry REAL GPU timestamps (Task 4.2, 2026-10-30): a `vkCmdWriteTimestamp` node rides the
  stream, `cudaEventElapsedTime` returns driver ticks scaled by `timestampPeriod` (radv BC-250:
  10 ns/tick, 64 valid bits on the compute family); the host-clock arm stays as the fallback for
  cards without the capability, and `STRATA_VK_NO_GPU_EVENTS=1` forces it for A/B. Both arms are
  pinned by the `vk_event_timing` ctest pair. PORT-NOTES law 15.
* An open deterministic bug: captured graphs + the expert-SWAP regime + the prompt-tail lookup
  guesser collapse generation to id-0s (narrowed and worked around in `vulkan/PORT-STATUS.md`:
  run the swap regime `--no-graphs`; full residency — every config this box serves — is clean).
* `device_count()` returns 1; multi-GPU/VMM are not ported. The pack engine (`anystrata-pack`,
  qwen4exp) and the dsv41 engine are not ported; every unported path traps BY NAME.
