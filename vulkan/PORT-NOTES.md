# Vulkan port notes

## Toolchain decision

**Compiler used:** system `glslc` (SPIRV-Tools 1.4.363.0, distro build "2026.4") remains the `.comp` kernel
compiler for `vulkan/`. The `strata_vk_kernel()` custom command now goes through
`vulkan/tools/vk-glslc.sh`, a shim that accepts the glslc-style invocation
(`--target-env=vulkan1.0 -O f.comp -o f.spv`) and dispatches to whichever front-end CMake selected.

**Vendored glslang: not available.** The plan was to build `glslangValidator` out of the llama.cpp submodule at
`third_party/llama.cpp/thirdparty/glslang` (commit `385445fad`, fork `robertkarl/llama.cpp`). That path does not
exist: the fork's `.gitmodules` is **empty** and the tree tracks **zero gitlinks** (modern llama.cpp moved its
third-party code to `vendor/` — httplib, nlohmann, stb, miniaudio, sheredom, hash — and carries no glslang at
all). `git submodule update --init --recursive` therefore has nothing to fetch, and per the task rules no
internet download was used to substitute. The CMakeLists still *prefers* a vendored binary
(`~build-glslang/glslang/glslangValidator` — first entry of `find_program`) so the switch is automatic
if one is ever built there; the shim's `VK_GLSLC_MODE=glslang` path was smoke-tested by pointing it at the
system glslangValidator and compiling `dequant_s2.comp` through it (works: `-V --target-env <env> -O f.comp -o
f.spv`).

**What the front-ends support (tested 2026-10-21, `/tmp/dotcheck.comp` = `dotKHR(uvec4…)` with
`#extension GL_EXT_shader_integer_dot_product : require`, `/tmp/shortbyte.comp` = std430 buffers with
`ushort[]`/`ubyte[]` members):**

| front-end | `dotKHR` / `GL_EXT_shader_integer_dot_product` | `ushort`/`ubyte` in std430 |
|---|---|---|
| system glslc 2026.4 | rejected (`no matching overloaded function`) | rejected (syntax error) |
| system glslangValidator 11:16.6.0 | rejected (`extension not supported`) | rejected (syntax error) |

So even the distro's glslang does not close the gap — only a newer vendored glslang would. The gap is
therefore **not closed**, and:

**Fallback kept:** the dot-product and 16/8-bit-integer probe modules stay hand-assembled `.spvasm` built with
`spirv-as --target-env vulkan1.2` (+ `spirv-val`), exactly as before:
`vulkan/src/probe/kernels/probe_dot.spvasm`, `probe_subgroup.spvasm`. Any future kernel that needs
`dotKHR`/u8-s16 storage types follows the same route until a modern vendored glslang is built.

**Verification after the wrapper change:** `cmake -S vulkan -B build-vk && cmake --build build-vk -j2 &&
ctest --test-dir build-vk` — 3/3 green (`vk_device_probe`, `s2_gemv_vk_parity`, `dequant_s2_vk_parity`).
The CUDA/HIP trees under `src/` and `include/strata/` are untouched.

## .spvasm authoring rules (learned assembling probe kernels on this box)

1. A StorageBuffer array member needs an explicit `OpMemberDecorate ... ArrayStride`
   (no `scalarBlockLayout` on RADV here); a Workgroup array must NOT have one
   (implicit layout).
2. At `--target-env vulkan1.2`, `spirv-val` demands Workgroup `OpVariable`s listed
   in the `OpEntryPoint` interface (contrary to the usual "only descriptor objects"
   rule).
3. SSA ids must be unique — no re-assigning through the same `%id`.
4. `maxComputeSharedMemorySize` = 65536 on gfx1013 (measured): the RDNA3 HIP-port
   kernels' 64 KiB LDS budget fits exactly, with no headroom above it.

## vulkan_compat (Task 1.1): the CUDA runtime + cuBLAS names, on strata::vk

`vulkan/include/strata/vulkan_compat/` mirrors `include/strata/platform/hip_compat/`: headers named exactly
like the CUDA ones (`cuda_runtime.h`, `cuda.h`, `cublas_v2.h`, `cuda_fp16.h`, `cuda_bf16.h`,
`driver_types.h`, `vector_types.h`, `math_constants.h`, `cublasLt.h`), all thin entries onto the one mapping
header `strata_vk.h`, placed BEFORE the include path by the `strata_vk_compat` INTERFACE target.  Unlike
hip_compat (a macro table onto hip's real runtime) this layer implements the semantics itself on
`strata::vk` primitives; inline state where it is cheap (events, last-error), `vulkan/src/runtime/vk_blas.cpp`
where a .cpp is allowed (the only new source file outside the headers).

**Device-pointer contract.** A CUDA "device pointer" is the arena offset plus a fixed 64-byte bias, cast to
`void*` (`strata_vk_compat::off/wrap`; the bias keeps offset-0 allocations distinguishable from `nullptr`,
which host code tests after a fresh cudaMalloc).  Launchers holding a host-side `void*` device pointer
convert with `strata_vk_offset(p)`.  All compat copies/memsets strip the bias before calling
`strata::vk::copy_h2d/copy_d2h/fill_d`, which take the raw arena offset cast to `void*` (the existing
convention in vk_runtime.cpp).

**Mapped (measured roster from the src greps):** cudaMalloc/Async/Free/FreeAsync (bump arena, no-op free —
the DeviceArena rule), cudaMemcpy(Async)/Memcpy2D(Async) incl. D2D (host-scratch walk; correct, unoptimized),
cudaMemset(Async), cudaHostAlloc/cudaMallocHost/cudaFreeHost (aligned host RAM; the staged copies read it
directly on UMA), cudaHostRegister/Unregister (tracked, succeed: nothing to pin against on this box),
cudaMemGetInfo (heap budget minus arena used), streams (Create/CreateWithFlags/CreateWithPriority/Destroy/
Synchronize/Query/WaitEvent/IsCapturing/LaunchHostFunc — one queue, submissions are synchronous, so
Query=done and WaitEvent=no-op are honest), events (pair-based host-clock timing per the plan: record = queue
drain + stamp; ElapsedTime = the real interval; Query/Synchronize = success),
cudaDeviceSynchronize/GetLastError(clears)/PeekAtLastError/GetErrorString/GetErrorName,
cudaGet/SetDevice(Count), cudaSetDeviceFlags, cudaInitDevice, cudaGetDeviceProperties/cudaDeviceGetAttribute
(name, gcnArchName="gfx1013", major/minor=10/1 family code, totalGlobalMem, sharedMemPerBlock=64K,
warpSize=32, maxThreadsPerBlock; multiProcessorCount/l2CacheSize are 0 until the Task-1.3 twin surfaces them),
cudaDeviceGetStreamPriorityRange, cudaDriverGetVersion(12040)/RuntimeGetVersion, cudaFuncSetAttribute
(no-op; no device-function objects exist), cudaFuncGetName (not supported), CUDART_VERSION=12040 chosen so
the engine's CUDA-13 gates (cudaMemcpyBatchAsync, cudaGraphGetEdges_v2) compile OUT, cudaGetDriverEntryPoint
(ByName*) answers "symbol not found" — the documented cuMem* VMM feature-detect in vmm.cpp/expert_cache.cpp
falls back to its plain tier (not a trap: those call sites branch on it), dim3 + vector types + make_*,
__half/__nv_bfloat16 with the tree's own f16_bits.hpp/bf16_bits.hpp conversions transcribed verbatim,
CUBLAS_* on the CPU fallback (below).

**Trapped (print their name and abort — never fake):** the cudaGraph* family
(StreamBeginCapture/EndCapture, GraphInstantiate(+WithFlags), GraphLaunch, GraphUpload, GraphDestroy,
GraphExecDestroy, GraphGetNodes, GraphGetEdges(_v2), GraphNodeGetType, GraphKernelNodeGetParams) —
"strata-vk: cudaGraphX not ported; run with --no-graphs".  graph.cpp's twins land in Task 1.4/Phase 4.
cudaMemcpyToSymbol/FromSymbol and cudaMemcpyPeer(Async)/cudaDeviceEnablePeerAccess: NOT aborts — they set
cudaErrorNotSupported and cudaGetErrorString names the gap (no device-symbol image exists in this backend;
multi-GPU is out of scope); cudaMemcpyDefault errors the same way rather than guess a direction.
cudaHostGetDevicePointer returns an error naming the Task-1.3 twins (every engine call site around it has a
fallback branch); a fake alias could not exist here since arena memory is never host-mapped by vk_runtime.

**cuBLAS:** `cublas_v2.h` declares the used surface (Create/Destroy/SetStream/GetStream/SetWorkspace/
SetMathMode, Sgemm, SgemmStridedBatched, GemmEx, GemmStridedBatchedEx, Hgemm); `vk_blas.cpp` runs a plain
blocked CPU GEMM — operands fetched from the arena through the staged copies, fp32 accumulation for every
CUDA_R_16F/16BF/32F mix, results written back. **speed-todo** (PORT-STATUS "cublas CPU fallback" row):
Phase 3's Task 3.1 replaces it with the dequant+GEMM `.comp` family.  A straddling operand span surfaces as
CUBLAS_STATUS_EXECUTION_FAILED with the runtime's named message, never a silent zero.

**Compile proof:** `vk_compat_smoke` (ctest, green) compiles the REAL host files `src/core/layout.cpp` +
`src/core/weights.cpp` unchanged — cudaMalloc/cudaMemcpy/cudaHostAlloc/cudaFreeHost/cudaDeviceSynchronize all
resolve against the compat headers with no CUDA toolkit on the box — then exercises an arena round-trip and
the cublas fallback (Sgemm and GemmEx fp16-in/fp32-out) against host references.

**Doorbell (Task 1.3, measured 2026-10-22):** RADV on the BC-250 does NOT deliver host stores to a
RUNNING kernel — a 40M-read `atomicAdd` spin on a HOST_VISIBLE|HOST_COHERENT buffer never saw a flag
the host flipped 200 ms in (nor with an uncached/WC allocation + `vkFlushMappedMemoryRanges`): the
Arc A750 landmine from sycl/ reproduces here. What works: a store that predates the LAUNCH is
visible to the next kernel on the same buffer (arm 4), and a flag staged in by a copy node before
the spin is seen on the first read (arm 2, `reads=0`). So `doorbell_ring`/`doorbell_wait`/
`copy_rows_from_mapped_after` must port as ring-then-launch (or copy-flushed flags), never live
spins inside a graph; `cudaHostGetDevicePointer` still refuses rather than hand back a host address
the arena-offset launchers would misread. `vk_doorbell_probe` (ctest) is the witness.

## Task 1.4 compat gaps (found by compiling/linking strata-poly-vk, in the order they appeared)

1. **`cuda.h` driver VMM calls become inline "not supported" answers.** `src/poly/engine.cpp` calls
   `cuDeviceGet`/`cuDeviceGetAttribute`/`cuMem*` DIRECTLY (the CUDA target links `CUDA::cuda_driver`),
   unlike `src/core/vmm.cpp` which resolves them by name through `cudaGetDriverEntryPoint`. The old
   declarations ("never linked against") would not link; they became inline definitions that answer
   `CUDA_ERROR_NOT_SUPPORTED` with `VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED -> 0`, which trips engine.cpp's
   own probe (`vmm_` stays false) onto the plain `cudaMalloc` arena path — the grow-never fallback the
   Task-1.3 twins already carry. Nothing reports success, so no VMM path can half-run.
2. **`cudaGetErrorString` after `cudaGetLastError()`'s clear** — the engine's `ck(e, what)` idiom
   (runtime.hpp) calls `cudaGetLastError()` (which CLEARS, per CUDA) and formats the string after: the
   arena's named reason ("copy span crosses an arena piece", "out of VRAM growing the arena") vanished
   into a bare "error". Fixed in `strata_vk.h` with a `last_reason()` that outlives the clear: the
   code clears exactly as CUDA does, the reason stays readable. (Found when a straddling load copy
   failed with an unnamed "a copy into VRAM: error".)
3. **Copy/fill chunkers clip to the piece they start in** (`vk_runtime.cpp piece_room()`). RADV's max
   allocation is 512 MiB, the engine's `cudaMalloc(512 MiB + tail)` chains pieces, and a 2 MiB chunk
   that straddled the chain boundary threw — the region is contiguous in arena space and the copy just
   has to respect the chain. Straddling copies now round-trip intact (probe case D rewritten: straddling
   copy must round-trip AND a span with no piece under it must still throw by name; `vk_device_probe`
   green on both arms).
   **Corrected (Task 2.11): the 512 MiB cap was FICTIONAL.** Vulkan exposes no max-allocation field
   and the figure was a session-start guess; measured on this box (controller probe): buffers of
   512 MiB, 1, 2, 4 and 8 GiB all SUCCEED, 12 GiB fails on the memory budget. `max_allocation_size`
   is now 6 GiB by default (STRATA_VK_MAX_ALLOC still overrides), so the engine's big single
   `cudaMalloc`s chain at far fewer boundaries. Two rules ride the arena now: (a) **piece-aligned
   regions** — `arena_add()` bumps the cursor past a piece whose tail is shorter than the new region,
   so every alloc that FITS one piece is single-piece by construction and no kernel span computed
   from a region's own numbers can straddle (this was attend's halt: the score scratch landed 5 MiB
   from a fictional 512 MiB edge); (b) genuinely oversized regions still chain and their FULL-span
   kernel binds keep the named rejection — probe case D pins 512 MiB via `setenv` so that class stays
   tested deterministically, and ctest `vk_piece_align` asserts (a) directly through the launcher's
   piece_of binding path.
4. **poly contract traps, not migrated copies.** `src/poly/kernels.cu` + `kernels_dsv4.cu` are host+device
   TUs (77/8 `<<<` launches): with no nvcc there is no "host-only migrated copy" worth extracting — the
   whole `include/strata/poly/kernels.hpp` contract (52 entries) becomes traps, generated mechanically by
   `vulkan/tools/gen_poly_stubs.py` (the gen_stubs.py AST mechanism pointed at that one header). Host-only
   parts carried honestly, NOT trapped: `sum_threads`, `attend_split_floats` (pure arithmetic), and
   `blas_create`/`blas_destroy` (the compat cuBLAS handle's lifecycle — the CPU fallback GEMM is a real
   host library here, no kernel involved).
5. **`strata::kernels` native_mmvq capability/byte-count queries host-defined** (missing.cpp; HOST_DEFINED
   in gen_stubs.py): `native_mmvq_supported`, `native_mmvq_ids_supported`, `native_q8_1_bytes`,
   `native_q8_1_il_bytes` are pure type tables / byte arithmetic in the CUDA build too — the LOAD asks them
   before any launch — so trapping them would re-route the whole load (every expert copied as bf16, the
   expert cache off) instead of reaching the launch traps in the model's demand order. They are transcribed
   from `src/kernels/cuda/native_mmvq.cu` (the `native_mmvq_il_supported` table stays false: the interleaved
   variants are not this backend's planned Phase-2 shape). `native_mmvq`/`_slots`/`_ids` and the quantizers
   STAY trapped — that is where the run stops, as designed.
6. **`strata_poly_mmq` (llama.cpp ggml-cuda MMQ) -> trap twin `vulkan/src/poly/mmq_vk.cpp`.** The
   constructor, `quantize()` and `mul()` name themselves; `fits()` honestly answers false (no MMQ tile
   built for this backend), which is exactly what CUDA's fits() returns when the card has no config —
   the engine keeps its batched prompt path off and the run reaches the decode traps instead.
7. **`src/core/graph.cpp` compiled unchanged**: its `cudaGraph*` calls land on the compat graph traps
   (`missing("cuda-graph: cudaGraphInstantiate")`-style names); `--no-graphs` keeps the engine off them.
   engine.cpp's own record()/replay sites trap the same way.
8. **gen_stubs.py drift guard** extended: the PORT-STATUS `todo` cross-check now tolerates the
   host-defined/ported names added above (the script's own HOST_DEFINED list is the registry).

**Not needed after all** (predicted friction that did not appear): `cudaFuncSetAttribute`,
`cudaOccupancy*`, `cudaMemcpyToSymbol` device-symbol sites, `__constant__` — the poly host files never
reach them (they are in the .cu files, which stay uncompiled); dim3/vector types were already in place.

## RADV arithmetic gotchas (Task 2.1, each measured on the BC-250)

1. **`packHalf2x16` TRUNCATES on RADV** — `(3.937008f, -469.0f)` packs to `0x43DF` (one ULP under
   the round-to-nearest-even `0x43E0`), and it was under on every magnitude-tested block. CUDA's
   `__float2half` is RN. Any kernel storing fp16 through `packHalf2x16` is one-ULP-off from the CUDA
   build at zero-tolerance; `quantize_q8_1.comp` carries an explicit `f16_rn_bits` (bit-twiddled RN,
   finite range) for the two half fields of every Q8_1 block instead.
2. **fp32 `/` is not correctly rounded here** (SPIRV allows the reciprocal path — the same trap the
   SYCL port documented: quantize_act 303k mismatches until `-cl-fp32-correctly-rounded-divide-sqrt`).
   Divides feeding a `round()` (q8_1's `xi / d`, `amax / 127`) go through `double` and back in the
   kernels.
3. `glslc`'s `&&` wants bool operands: `(x & 1u)` alone is a compile error — write `(x & 1u) != 0u`.
4. The parity TEST's reference must model shuffle trees with a per-round copy (`p[j] = l[j]; l[j] =
   l[j] + p[j^o]`); an in-place `l[j] += l[j^o]` clobbers values later lanes still read and sums the
   warp wrong.
5. Streams: `native_mmvq`'s `validate_stream()` REJECTS a null `cudaStream_t`; `stream_create()` now
   hands out distinct non-null tokens (all the same queue underneath) so host code keeps CUDA-shaped
   discipline and the validator passes.
6. **glslc's default contraction IS active on RADV and matches the CUDA --use_fast_math fused dot.**
   Task 2.3 (q5_k): the CUDA native_mmvq.cu builds with contraction ON (CMakeLists.txt:518-538:
   `--use_fast_math`, no `-ffp-contract=off` for this file), so the exact tree dot semantics are
   `sumf = fma(d8, dot*sc, sumf)` per accumulator step and `fma(-dm_y, sumf_m, dm_x*sumf_d)` for the
   final `dm_x*sumf_d - dm_y*sumf_m` pair (the negated product folds in; `fma(dm_x, sumf_d,
   -(dm_y*sumf_m))` differs). glslc does NOT emit OpFma itself (still OpFMul+OpFAdd), but RADV's NIR
   contracts and lands on exactly the CUDA-fused result — the host reference must spell std::fma in
   that order, and the unfused reference differs on ~290/908 outputs. The iq2_xxs dot passed unfused
   only because it has no mul-into-accumulator pair to fuse.
7. `shared` variables must be at global scope in GLSL (not inside main); and a `continue`/`return`
   past a `barrier()` in a loop is divergent-barrier UB — guard the reduce arm with `if` instead.
8. **`half` is a reserved word in GLSL** — a local named `half` (the rope pair count, natural from
   the CUDA source) fails to compile. Task 2.10 renamed it `nh`.
9. **glslc's sin/cos/pow are NOT CUDA libdevice twins on RADV.** Measured on the BC-250
   (Task 2.10, head_norm_rope): `v_sin_f32/v_cos_f32` carry ~2e-6 ABSOLUTE drift at moderate
   angles and wrap badly past ~4e3 rad (0.43 rel at 1e6); hardware `pow` (v_log+v_exp) drifts
   ~4e-6 rel at theta_scale^63. The honest port: pow the mantissa table host-side in the
   launcher (one upload, cached), reduce angles with three-word Cody-Waite 2*pi + pi/2 fold,
   and evaluate Taylor sin/cos in-kernel (coefficients rounded f32: truncation 1.8e-9 sin,
   1.2e-8 cos at pi/4). One Newton step on a divide does NOT fix law 2's frcr error (measured
   no movement at all) — go through double.

10. **On this APU, llama.cpp `-ngl` ANY value can out-of-memory-kill the desktop — never run it
   without containment.** Measured law (Phase-5 gate, 2026-10-26): its Vulkan backend loads the
   model into TWO device tiers even at `--ngl 24` — `Vulkan0` (~6.3 GiB) plus a `Vulkan_Host`
   fallback buffer (~5.3 GiB) — and amdgpu/TTM **pins both into system RAM as non-reclaimable
   pages**. With the 12.2 GiB GGUF resident, kswapd fires and the kernel OOM killer takes its
   victims by score: three crashes in one day, victims were kwin_wayland, plasma-systemmonitor,
   and the Hermes session — never the benchmark itself. Session processes carry oom_score_adj=+200
   by default, so they DIE FIRST; the culprit keeps running. The Strata engine itself is immune —
   it sizes its own device-local arena (9.4 GiB, proven stable all through the long gates) — so
   this is a llama.cpp-on-APU law, and it constrains ONLY the reference legs of the correctness
   harness. Rules that survived: (a) at most ONE llama.cpp model load at a time, kill+sleep >=20 s
   between legs; (b) wrap every leg in `choom -n 400` (raises the leg's OWN adj so a breach kills
   it first) + `nice -n 10`; (c) on this box prefer `--ngl 0` reference arms (pure CPU, weights as
   mmap page cache = reclaimable, pressure evicts pages instead of people); (d) the 32K-context
   leg (d31k) is NOT hostable here at all — record it as machine-limited, as the fork does for
   other machines' missing sets.
11. **ENGINE context sizing law on this APU: 16K fits, 32K fits, 64K does not (yet).** Measured
   (2026-10-27): --max-context 65536 = 2.56 GiB f16 KV INSIDE the pinned arena -> resident 14.75 GB,
   plus desktop ~1 GB: OOM kill, session included (Hermes died with the server; adj tie, law 10b).
   32K (KV 1.28 GiB, resident ~13.9) and 16K (~14.1 incl. warm cache) run stable. The unlock for
   64K is q8_0 KV (halves KV to 1.28 GiB at 64K): the HOST-side code is in-tree and parity-tested
   (kv_q8.hpp; the fork's kvq8 correctness arm), but the Vulkan kernels
   (kv_append_q8_step / kv_gather_q8_step, PORT-STATUS todo) must be ported first — that, plus the
   vmm grow-never arena's 'cost the max context at startup' fallback, is the honest path to 64K.
   Do not raise --max-context past 32768 before those kernels are green.

12. **Graph capture laws (Phase 4.2, measured 2026-10-29).** (a) A captured graph's H2D nodes
   carry the bytes frozen AT CAPTURE — the engine only replays graphs whose inputs are written
   outside the graph (h_at_ eager copy2); never capture a host-dependent per-token value.
   (b) D2H in a graph = node -> per-graph mapped staging, delivered at the fence drain; the
   capture-time spec (d2h_def) must NOT be serviced before a replay ran (staging unwritten).
   (c) Lock law inside the runtime: g_m (pieces) then g_pm (leaf) — a helper re-taking g_m under
   a caller that holds it self-deadlocks (non-recursive mutex, silent hang). (d) The engine's
   copy sites route through strata::vk::copy_h2d/copy_d2h DIRECTLY (poly copy2 launcher), not
   only the compat strata_memcpy: capture-awareness must live at BOTH levels or the capture body
   drains mid-record (flush_wait-in-capture abort, which is the LOUD refusal working).
