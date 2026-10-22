# Vulkan: the BC-250 backend (`vulkan/`)

The third GPU backend of this fork, and the one that runs the AMD **BC-250** (Cyan Skillfish, `gfx1013`: an
RDNA-1.x-class APU with 16 GB unified GDDR6). Like `sycl/` it is its own project that takes the engine's sources
from `src/` and carries migrated device code beside it, so the CUDA and HIP trees stay byte-for-byte upstream's.

Why Vulkan and not ROCm/HIP on this card: ROCm's userspace (rocBLAS/Tensile) ships no `gfx1013` kernels and the
HIP runtime does not target it, while the Mesa **RADV** driver is first-class (Vulkan 1.4). llama.cpp's own
Vulkan backend is what people run models on this board with; this page is about running *Strata's* engine there.

## Status (first increment)

`vulkan/` builds, runs, and passes its tests on real BC-250 hardware (2026-10-21, Mesa RADV, kernel 7.2.11):

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

## What the backend provides

`src/runtime/vk_device.cpp` / `vk_runtime.cpp` + `include/strata/vulkan_runtime.hpp`:

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
  `kernel_launch_sized(...)`; `.comp` -> `glslc` -> embedded `uint32_t[]` (`tools/embed_spv.py`), no run-time
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

## Known limits of the first cut

* One queue, synchronous submit (launch = record + submit + wait): correct, not fast; streams/pipelining are
  the next increment, and the parity harness tolerates it because it checks results, not throughput.
* Events time 0 (host-clock probes are a TODO); the engine uses them only for PCIe probes, which a UMA APU
  does not need.
* `device_count()` returns 1; multi-GPU/VMM are not ported.
* No model runs end-to-end on this backend yet — the S2 GEMV, probe, and arena/copy layer are increment 1 of
  the plan above.
