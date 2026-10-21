# Vulkan Engine Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete, honest Vulkan backend for PolyStrata's `strata-poly` engine, running Qwen3.6-35B-A3B (and the other `qwen35moe`-family GGUFs) on the AMD BC-250 (gfx1013, Mesa RADV), held to the tree's own correctness bar: every ported kernel green on its in-tree parity contract, and the `bench/correctness/` teacher-forced harness placing `strata-poly-vk`'s per-step distributions within llama.cpp's own CPU-vs-GPU spread.

**Architecture:** Approach C — host shim as the long pole, kernel order driven by the model run, named traps as the work queue. `vulkan/` is its own CMake project in the `sycl/`/`metal/` shape: it compiles the engine's **unmodified** host sources from `../src` against a `vulkan_compat/` header layer that hands them the CUDA runtime names on Vulkan (the `hip_compat/` pattern), over a device runtime (`vk_runtime`) that owns the arena, queues, events, and a kernel registry. Every not-yet-ported kernel entry point is a trap that prints its name (`missing.cpp`'s pattern). Model runs enumerate the queue; parity tests close it.

**Tech Stack:** C++20, Vulkan 1.4 loader + RADV (Mesa 26.2), GLSL→SPIR-V via the vendored glslang in `third_party/llama.cpp` (built once; falls back to `spirv-as` on `.spvasm` for dot-product intrinsics on distros whose glslc predates `GL_EXT_shader_integer_dot_product`), CMake ≥3.24, the tree's existing parity-test harness (`ctest`) and correctness harness (`bench/correctness/`).

**Target hardware facts (measured 2026-10-21 on this box):** gfx1013 = RDNA1 family, **native subgroup 32** (no wave32 emulation needed — the forced-32 probe proved the *mechanism* for wave64 parts), 64 KiB workgroup memory (matches the RDNA3 HIP kernels' budgets), 16 GiB **unified** GDDR6 (uma=1: no PCIe tier, pinned/host-mirror paths collapse), max single VkDeviceMemory allocation 512 MiB (arena is a piece chain), Vulkan 1.4.354 with `VK_KHR_shader_integer_dot_product` (OpUDot executes correctly) and `VK_EXT_subgroup_size_control`; **no** `VK_KHR_cooperative_matrix` (WMMA/tensor-core kernels take the ordered-FP32 fallbacks the HIP port already carries); no ROCm userspace.

**Build budget warning (this box):** 6c/12t, 14 GiB RAM. ggml-vulkan alone took ~40 min. Build kernel libs separately (`ninja strata_vk_kernel_<name>`), never `make -j12` on the full engine; expect full-engine links ~1-2 h. Use `-DCMAKE_BUILD_TYPE=Release` and `ccache` if present.

---

## Continuation protocol (read this first, every session)

1. The live state ledger is **`vulkan/PORT-STATUS.md`** (created in Task 0.3). It lists every `src/kernels/cuda/*.cu` + `src/poly/kernels*.cu` entry point with state `todo | ported+parity-green | ported+model-verified | fallback`. Update it in the same commit as the work.
2. Resume point = first non-green row in PORT-STATUS.md that a model run reached, or the next unchecked task below — whichever is later. `git log --oneline -- vulkan/` tells the rest.
3. Gate commands, always green before a commit:
   - `cmake --build build-vk -j2 && ctest --test-dir build-vk --output-on-failure` (unit gate)
   - `build-vk/bin/strata-poly-vk --model ~models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --no-graphs --prompt "hello"` (trap gate — must fail with a *named* missing kernel until Phase 4)
4. Environment: repo `~UnbiasedStrata`, branch **`vulkan-backend`**, build dir `build-vk/` (`make vulkan`), model at the path above. Never edit files under `src/`, `include/strata/` (CUDA/HIP trees) — the fork's rule, upheld by `git diff main -- src include` staying empty.

**Proven so far (main→vulkan-backend, committed):** `vulkan/` scaffold builds; `vk_device_probe` PASS (dp4a computes 10; forced-32 subgroup returns 321); `s2_gemv_vk_parity` PASS 640/640 bit-exact; `dequant_s2_vk_parity` PASS 0 mismatches at zero tolerance. Runtime pieces exist: arena (piece-chained, straddle throws named), staged copies, single queue, kernel registry with build-time-embedded SPIR-V (`tools/embed_spv.py`), `Makefile` targets `vulkan`/`vulkan-test`, top-level `STRATA_ENABLE_VULKAN` guard.

**Driver landmines (each cost a debugging session; don't re-trip):**
- RADV **segfaults** at `vkCreateComputePipelines` if `pLayout` is unset — set it before creating.
- SPIR-V declaring a capability whose *device feature* wasn't enabled at `vkCreateDevice` → segfault, not an error code. `vk_device.cpp` enables `shaderIntegerDotProduct`, `subgroupSizeControl`, `computeFullSubgroups` explicitly.
- This distro's glslc/glslang (2026.4) lack `GL_EXT_shader_integer_dot_product` in the frontend and reject `ushort`/`ubyte` std430 members → pack to `uint` + `unpackHalf2x16`/`unpackSnorm4x8`-style shifts (llama.cpp's `*_packed32` structs do the same), or hand-assemble `.spvasm` (worked incantation in `vulkan/src/probe/kernels/probe_dot.spvasm`: `OpCapability DotProduct` + `DotProductInputAll` + `OpExtension "SPV_KHR_integer_dot_product"`).
- `vulkaninfo --json` produces empty output on this box; parse `vulkaninfo` text instead.
- llama.cpp's `llama-bench -p` wants comma-separated sizes (`-p 512,2048`), not space.

---

## Phase 0 — Toolchain, status ledger, and the trap pattern (half a day)

### Task 0.1: Vendor glslang from the submodule; make it the kernel compiler

**Files:**
- Create: `vulkan/tools/build_glslang.sh`
- Modify: `vulkan/CMakeLists.txt` (GLSLC discovery order)

- [ ] **Step 1:** Confirm the submodule carries glslang: `ls third_party/llama.cpp/thirdparty/glslang` (currently empty — submodules not recursive).
- [ ] **Step 2:** `git submodule update --init --recursive third_party/llama.cpp/thirdparty/glslang` (skip if the checkout is absent from this fork's .gitmodules — then keep spirv-as route and mark this task done "fallback only").
- [ ] **Step 3:** Build it once: `cmake -S third_party/llama.cpp/thirdparty/glslang -B build-glslang -DENABLE_OPT=OFF -DENABLE_HLSL=OFF -DBUILD_TESTING=OFF && cmake --build build-glslang -j2 --target glslangValidator`. Expected: `build-glslang/glslang/glslangValidator`.
- [ ] **Step 4:** In `vulkan/CMakeLists.txt`, prefer it: `find_program(GLSLC NAMES glslc ... PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../build-glslang/glslang /usr/bin)`; also accept `glslangValidator -V` via a thin wrapper `vulkan/tools/build_glslang.sh` that maps `--target-env=vulkan1.0` → `--target-env vulkan1.0`.
- [ ] **Step 5:** Prove the frontend gap closes: compile a test kernel containing `#extension GL_EXT_shader_integer_dot_product : require` + `dotKHR(uvec4(1,2,3,4),uvec4(1))` writing to a `uint` std430. Expected: compiles. If the vendored glslang is also too old, keep the `.spvasm` route for dot kernels and record the decision in PORT-STATUS.md.
- [ ] **Step 6:** Commit: `vulkan: vendor glslang for the dot-product GLSL frontend` (or the fallback note).

### Task 0.2: Workgroup-memory and address sanity probe (cheap insurance before 320 kernels)

**Files:**
- Modify: `vulkan/src/probe/kernels/` (add `probe_uma.spvasm` or `.comp`), `vulkan/src/probe/device_probe.cpp`

- [ ] **Step 1:** Add probe case C: a kernel declaring `OpVariable Workgroup` 64 KiB (the RDNA3 LDS budget the hip-compat kernels assume) doing one shared-array round-trip; assert the value. This proves the biggest HIP-port kernel shapes fit before translating them.
- [ ] **Step 2:** Add probe case D: two arena allocations far apart (>1 GiB apart if free memory allows) holding a known pattern; a kernel with both buffers bound reads each correctly — proves the piece-chained arena's descriptor binding under real sizes, and the straddle-assert's sanity.
- [ ] **Step 3:** Run: `ctest --test-dir build-vk -R vk_device_probe --output-on-failure`. Expected: PASS.
- [ ] **Step 4:** Commit: `vulkan/probe: 64K workgroup arrays and far-apart arena spans execute on RADV`.

### Task 0.3: PORT-STATUS.md — the ledger

**Files:**
- Create: `vulkan/PORT-STATUS.md`

- [ ] **Step 1:** Generate the roster: `for f in src/kernels/cuda/*.cu src/poly/kernels.cu src/poly/kernels_dsv4.cu; do grep -oE '^[a-z_0-9]+ [a-z_0-9]+\(|void [a-z_0-9_]+\(|__global__ void [a-z_0-9_]+' $f; done` — plus read each `include/strata/kernels/*.hpp` for the host-callable names. A kernel *entry point* = one host-callable function in `strata::kernels` (e.g. `s2_gemv`, `rope_neox_apply`), not one `__global__` (files hold several variants; note variants in the row).
- [ ] **Step 2:** Write the table: `entry point | header | .cu file | parity test in-tree | CUDA-only constructs (shfl/shared/ldg/wmma/mma/cp.async/graph) | state`. Mark `dequant_s2`, `s2_gemv` `ported+parity-green`; `s2_gemv_fast/quads`, `iq_kernels`, `native_*`, `qsa_*` etc. `todo`. Add a "decision" column values: `translate | fp32-fallback | host(free)` where the HIP port already defines an ordered-FP32 fallback (grep `docs/AMD_HIP.md` "QSA matrix instructions" and `STRATA_ATTN_PRE75`).
- [ ] **Step 3:** Commit: `vulkan: port status ledger — the full entry-point roster`.

---

## Phase 1 — The compat layer, host-file first compile (1-2 days)

The goal of this phase is **not** running: it is `strata-poly`'s eleven host files + `strata_core`/`strata_kernels`-dependent sources *compiling and linking* against Vulkan, with unported kernels as traps. This is the long pole; do it before more kernels.

### Task 1.1: `vulkan_compat/` — the CUDA runtime surface on Vulkan

**Files:**
- Create: `vulkan/include/strata/vulkan_compat/cuda_runtime.h`, `cuda.h`, `cuda_fp16.h`, `cuda_bf16.h`, `cublas_v2.h`, `cublasLt.h`, `driver_types.h`, `vector_types.h` (mirror `include/strata/platform/hip_compat/`'s file list)
- Modify: `vulkan/CMakeLists.txt` (new `strata_vk_compat` gets these on the BEFORE include path — hip_backend does exactly this: `target_include_directories(... BEFORE INTERFACE .../hip_compat)`)

- [ ] **Step 1:** `cuda_runtime.h` → `#include "strata/vulkan_runtime.hpp"` + inline mappings, following hip_compat's shape but *not* re-declaring CUDA types the same way (read hip_compat/cuda_runtime.h first and diff the design decisions):
  - `cudaError_t` enum with `cudaSuccess=0`, `cudaErrorMemoryAllocation`; `cudaGetErrorString` returns the arena-throw message when set.
  - device pointers: **Vulkan keeps CUDA's `void*` device pointers** — the arena hands out offset-as-pointer `DevicePtr`s cast to `void*` (already the convention in the two ported launchers). All compat copy/malloc calls pass through `vk::copy_*`/`Arena::alloc`. This is what makes the 99 host files compile *unchanged*.
  - `cudaMalloc/cudaFree/cudaMallocAsync` → arena alloc (Free is a no-op like CUDA's `--no-free` arena rule; document that).
  - `cudaHostAlloc/cudaFreeHost/cudaMallocHost` → `VK_MEMORY_PROPERTY_HOST_VISIBLE|COHERENT` buffer (UMA: host pointer + arena DevicePtr both valid; keep both, hip_compat's mapped-memory model).
  - `cudaMemcpy/cudaMemcpyAsync` + kind enum → the staged chunked copies; **straddling a piece boundary throws named** (planner regions must be sized to pieces — record region sizes vs 512 MiB in PORT-STATUS when the first big tensor load trips it).
  - `cudaMemset/cudaMemsetAsync` → `fill_d`.
  - `cudaStream_t`=opaque `void*` (null = default queue, matching the existing convention); `cudaStreamSynchronize/Destroy/CreateWithFlags` map onto queue wait.
  - `cudaEvent_t` → **timeline semaphores with `vkGetCalibratedTimestampsEXT`-free host-clock timing**: `event_record` submits a timestamp write via a tiny kernel OR (first cut) host clock after `vkQueueWaitIdle` — correctness-of-timing only matters to PCIe probes; on UMA they should mostly be skipped: add a
`STRATA_VK_SKIP_PCIE_PROBE=1` default-on behavior honored by the pinned twin (Task 1.3).
  - `cudaGetLastError/cudaPeekAtLastError` → thread-local last-error string; `cudaDeviceSynchronize` → queue+device wait; `cudaDeviceGetAttribute`/`cudaGetDeviceProperties` → filled from `vk::DeviceInfo` (`multiProcessorCount` ← `maxComputeWorkGroupCount[0]` sanity value; `major/minor` ← report RDNA1 family; `totalGlobalMem` ← heap budget).
- [ ] **Step 2:** `cuda_fp16.h`/`cuda_bf16.h`: `__half`/`__nv_bfloat16` as 16-bit wrapper types with `__half2float`/`__float2half` via software IEEE conversion (RADV-side conversions happen in kernels through `unpackHalf2x16`; host conversions only touch metadata paths). hip_compat already carries these transcriptions — copy their tested implementations verbatim where they exist.
- [ ] **Step 3:** `cublas_v2.h`: implement the used surface (measured roster: `cublasCreate/Destroy/SetStream/SetWorkspace/SetMathMode`, `cublasSgemm` ×14, `cublasGemmEx` ×14, `cublasGemmStridedBatchedEx` ×5, `cublasSgemmStridedBatched` ×3, `cublasStatus_t`) as a **CPU fallback + Vulkan kernel backend**:
  - first cut: call ggml's CPU path — link `ggml-base`+`ggml-cpu` (already in the CUDA build for native experts) and route GEMMs through `ggml_backend_cpu_buffer` + `ggml_mul_mat`? No: too heavy. Do the simple thing: a hand-written blocked fp32/fp16 GEMM in `vulkan/src/runtime/vk_blas.cpp` (AVX2, ~150 lines) for correctness; PORT-STATUS marks it `host(free)` speed-todo. cublasGemmEx's fp16-in/fp32-acc semantics spelled out in its header comment.
  - kernel GEMM comes in Phase 3 (prefill) — do NOT stall Phase 1 on it.
- [ ] **Step 4:** Compile check: `strata-plan`-equivalent host TU: compile `src/core/layout.cpp` + `src/core/weights.cpp` with `-I vulkan/include/strata/vulkan_compat -I include` and zero CUDA installed. Expected: compiles+links against `strata_vk_runtime`. Commit: `vulkan_compat: the CUDA runtime and cuBLAS names, mapped`.

### Task 1.2: The trap library — `missing()` and the stub roster

**Files:**
- Create: `vulkan/src/engine/missing.cpp` (copy `metal/src/engine/missing.cpp` shape: default prints `strata-vk: <name> has no Vulkan implementation yet` + `_Exit(86)`; `STRATA_VK_STUBS=log` lists per-entry once)
- Create: `vulkan/src/engine/stubs_phase0.cpp` — trap definitions for every `strata::kernels` entry point from PORT-STATUS marked `todo`, each one-liner `{ missing("s2_gemv_fast"); /* return-neutral defaults where the contract allows */ }`
- Modify: `vulkan/CMakeLists.txt`

- [ ] **Step 1:** Generate the stub file mechanically from PORT-STATUS (script it: `vulkan/tools/gen_stubs.py` reading the ledger → emitting traps with correct signatures from the headers). Traps for functions with return values must still satisfy the signature (return {} where legal).
- [ ] **Step 2:** Link order: real launchers (e.g. `strata_vk_kernel_s2_gemv`) win over the stub object because stubs live in a separate static lib at link tail (mirror how metal's `stubs.cpp` sits behind real kernels). Verify with `nm` that `s2_gemv` resolves from the kernel lib.
- [ ] **Step 3:** Commit: `vulkan: trap roster — unported kernels name themselves and stop the run`.

### Task 1.3: Vulkan-native twins for the three CUDA runtime files

**Files:**
- Create: `vulkan/src/core/vk_device.cu`-equivalent → plain `vk_device_impl.cpp` implementing `include/strata/core/device.hpp` (DeviceInfo fields, `gpu_arch_problem` = "" for gfx1013 when probe passed, `device_summary`, `set_primary_device` no-op single-device, `DeviceArena` over `vk::Arena`)
- Create: `vulkan/src/core/pinned.cpp` → implements `pinned.hpp` contract on UMA: pinned host memory is just host-visible coherent buffers; `PinnedRing` keeps its semantics (doorbell flags work: RADV host coherence was implicit in the copies — **verify with a dedicated test below**).
- Modify: `vulkan/CMakeLists.txt` (`strata_vk_core` static lib)

- [ ] **Step 1:** Doorbell test first (this box's `sycl/` docs warn some drivers hide host stores from running kernels — must be proven, not assumed on RADV): kernel spins on a host-visible buffer word ≤2M reads; host flips it; kernel must see it. If it fails, add the uncached-allocation route the SYCL port used, and record it.
- [ ] **Step 2:** Implement `device.hpp` + `pinned.hpp` contracts; `src/core/vmm.cpp` gets a **pre-allocated fallback** (grow-never: allocate max-context attention caches upfront from the arena — 35B-A3B at 32K ctx int8 KV ≈ 2-3 GiB, fits; sparse `cuMemMap` equivalent = `VK_KHR_maintenance5`/no sparse; PORT-STATUS row `host-free fallback`).
- [ ] **Step 3:** Commit: `vulkan: device + pinned + vmm contracts, doorbell handshake proven on RADV`.

### Task 1.4: Host sources compile UNCHANGED — first big link

**Files:**
- Modify: `vulkan/CMakeLists.txt` — add `strata_poly_vk` executable mirroring the CUDA target's source list (copy the `add_executable(strata-poly ...)` block's file list from `CMakeLists.txt:2059`, plus the `strata_core`/`strata_kernels`/`strata_spec`/`strata_plan`/`strata_artifact`/`strata_ngram`/`strata_qwen35`/`strata_prefill`/`strata_program` libs as their sources list them — read `CMakeLists.txt` sections and mirror, with `device.cu`/`pinned.cu` swapped for the Task-1.3 twins and `graph.cpp` compiling with cudaGraph calls mapped to a **capture-disabled stub** (host graph ops throw a trap naming the site; the `--no-graphs` switch keeps the engine from reaching them): add to compat: `cudaStreamBeginCapture/EndCapture/GraphLaunch/GraphExecDestroy/GraphCreate/GraphAdd` → trap `missing("cuda-graph")`.
- Test: `build-vk/bin/strata-poly-vk --help` runs at all.

- [ ] **Step 1:** Compile. Expected: a fistful of errors — each one a compat gap (a CUDA identifier used that Task 1.1 missed: e.g. `cudaFuncSetAttribute` → no-op inline, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` → computed from limits, `__constant__`/device-symbol query calls → arena pre-allocation, `cudaDriverGetVersion` → `VK_API_VERSION_1_4`). Fix in compat, NOT in host sources. Iterate until link succeeds.
- [ ] **Step 2:** Run `build-vk/bin/strata-poly-vk --model <IQ2_XXS.gguf> --no-graphs --max-context 4096 --prompt "hi"`. Expected: startup log, model load, then a **named trap** (first unported kernel the qwen35moe decode path hits). That trap name = the port order for Phase 2. Paste it into PORT-STATUS under "model-demanded order".
- [ ] **Step 3:** Commit: `vulkan: strata-poly links against the compat layer; model run demands kernels by name`.

---

## Phase 2 — Decode-path kernels, model-driven (the long grind; each task identical shape)

**The recipe, verbatim per kernel** (this is the DRY instance; every task below applies it to one entry point):

1. Read the `.cu` (all variants it holds) and its `include/strata/kernels/*.hpp` contract.
2. Write `vulkan/src/kernels/<name>.comp` — line-by-line, same term order, same integer-domain rules; 16-bit data packed to `uint`; `__ldg`→plain loads; `__syncthreads`→`barrier()`; `__shfl_xor_sync(FULL,…)@32`→`subgroupShuffleXor` (native 32 here); `__shared__` arrays→`shared` (64 KiB budget proven by probe C).
3. Write `vulkan/src/kernels/<name>_vk.cpp` launcher implementing the header contract over `kernel_load`/`kernel_launch_sized` (the `s2_gemv_vk.cpp` pattern; sizes table for piece asserts).
4. Port the **in-tree parity test** (`src/kernels/<name>_parity.cpp`) to `vulkan/src/tests/<name>_vk_parity.cpp` — host reference identical, device call swapped to arena (the `s2_gemv_vk_parity.cpp` pattern; keep SKIP-return-77 and `--selftest`).
5. `strata_vk_kernel(<name>)` in `vulkan/CMakeLists.txt`; delete the trap line from `stubs_phase0.cpp` (the generator re-run does this).
6. `ctest -R <name>_vk_parity` green → PORT-STATUS flips → commit: `vulkan: <name> on the backend (parity: <the number the CUDA test asserts>)`.

**Tolerance law:** copy each CUDA parity test's bound exactly — dequant family zero tolerance; gemv family rel 1e-5; fp16 paths 1e-3; if a GLSL-side number won't meet the in-tree bound, STOP and document the divergence in PORT-STATUS before committing (never widen a tolerance silently; the fork's whole correctness culture is the printed measured number).

- [ ] **Task 2.1:** First three by *model demand order* (from Task 1.4 Step 2's trap log — expected candidates: `router_top10`, `quantize_act`, `rope` or an `iq_*` expert GEMV; let the trap log decide, don't guess).
- [ ] **Task 2.2-2.5:** Next four demanded. One task per trap cleared, same recipe.
- [ ] **Task 2.6:** `native_moe` / `iq_kernels` expert decode family — the hot path. Note: `iq_kernels.cu` is large; split its entry points across 2-3 commits.
- [ ] **Task 2.7:** KV decode family (`kv_q4`, `kv_q8`, `kv_stream`) — includes the mrope device-table path (`mrope_pos`/`mtab`: `include/strata/kernels/mrope.hpp` is host-buildable; the table rides an arena buffer).
- [ ] **Task 2.8:** sampler + elementwise + cvec family (`elementwise.cu`, 461 lines, several entry points).
- [ ] **Task 2.9:** gate: **first token out.** `strata-poly-vk --no-graphs` completes one prompt + one greedy token on IQ2_XXS, log shows zero traps. Then run `poly-run.py` (bench/correctness) against the llama.cpp reference at short length as a smoke test — not the formal gate, just shape-checking.
- [ ] **Task 2.10:** spec/verify + MTP decode kernels (`verify_kernels.cu`, `src/spec/*` device parts) — speculation is how the NVIDIA numbers hit; the draft path must run before the teacher-forced harness uses realistic prompts.

**Deferred-by-design (do NOT port yet):** graph capture (`--no-graphs` is the interim), the `prefill/` MMQ/pointer-table paths, `remote_experts`/`peer_experts` (multi-GPU), `nvfp4` (NVIDIA-only format, mark `n/a` in PORT-STATUS).

---

## Phase 3 — Prefill, dense GEMM, attention (the speed-and-fallback phase)

- [ ] **Task 3.1:** dense projections: replace the Task-1.1 cublas-compat CPU GEMM with a `dequant+GEMV/GEMM` `.comp` family — port the tree's own portable HIP path (grep `hip_backend`'s `STRATA_PREFILL_MMQ` and `STRATA_DENSE_MMQ` sites; on CUDA these come from `strata_poly_mmq`/`libstrata_kq.so` — read `src/prefill/` and mirror the **portable** kernels only).
- [ ] **Task 3.2:** attention decode kernels (`qsa_decode_attn`, gr/gdn recurrent kernels as `strata-poly` reaches them on qwen35moe) — **ordered-FP32 variants only** (the docs' exact wording: "CUDA-only QSA matrix instructions have an ordered FP32 fallback" — the HIP build's fallback is the reference; parity tests must compare against the same oracle the HIP parity runs use, e.g. `qsa_*_parity.cpp` on their non-TC branch).
- [ ] **Task 3.3:** prompt attention (`qsa_prompt_attn`): HIP compiles the tensor-core kernel out and runs the FP32 kernel — same route; `STRATA_ATTN_PRE75`'s 8-cells-per-step kernel is the RDNA2-proven shape (gfx1030 report), closest existing evidence for gfx1013.
- [ ] **Task 3.4:** prefill parity: `bench/micro` prompts probes where present, else the model-run check: teacher-forced prompt passes on the 16 harness sequences (that's Phase 5's harness; early taste via `make-inputs.py`).
- [ ] **Task 3.5:** measure and record honestly in PORT-STATUS: pp/tg tok/s for IQ2_XXS on strata-poly-vk `--no-graphs` vs the llama.cpp-Vulkan baseline (383 pp / 76 tg). If below: tune later; do not claim above.

## Phase 4 — Graphs and engine polish

- [ ] **Task 4.1:** Replace the per-launch record+submit+wait with per-layer recorded command buffers — the honest port of `CapturedGraph`: read `include/strata/core/graph.hpp` for the contract (capture-once/replay-many, `reset()`'s destroy-order comment applies: free the command buffer only after queue idle), map `begin()/end()/launch()` onto `vkBeginCommandBuffer` (secondary) / `vkEndCommandBuffer` / `vkCmdExecuteCommands` from one primary buffer per layer. `src/core/graph.cpp`'s host calls stay untouched behind compat traps removed one at a time.
- [ ] **Task 4.2:** events with real GPU timestamps (`VK_KHR_timestamp_counter` via `timestampComputeAndGraphics=true` — proven on this box) so `bench` timings and the engine's per-stage tables become meaningful.
- [ ] **Task 4.3:** drop `--no-graphs` from the model run; verify identical tokens with and without (this is `poly_draft_race_test`'s job, port it too).

## Phase 5 — The correctness gate (the definition of done)

- [ ] **Task 5.1:** run `bench/correctness/README.md`'s procedure verbatim with `US2_MODELS=~models/Qwen3.6-35B-A3B-MTP-GGUF` (`check-batch.sh`, `make-inputs.py`, `parity-run.sh`, `strata-run.py`): teacher-forced 16 sequences × 2 models (IQ2_XXS, IQ3_XXS), both arms, compared against the pinned llama.cpp build in `~llama-vk`.
- [ ] **Task 5.2:** assert the fork's bar: mean KL within 0.6-1.2× llama.cpp's own CPU-vs-GPU spread; greedy token agrees at every position with a 0.5-nat margin except the near-tie handful documented on CUDA; record findings as USTRAT-style tickets in `vulkan/PORT-STATUS.md` (deviations get tickets, not silence).
- [ ] **Task 5.3:** VRAM-residency note: USTRAT-9/11-style finding — free-VRAM-dependent expert placement applies on unified 15.7 GiB; record `--expert-cache` sizing at which outputs are bitwise stable for this backend.
- [ ] **Task 5.4:** write `docs/VULKAN.md`'s "Measured" section from the harness run; final commit of the phase.

## Phase 6 — Finish the roster (never blocks the gate)

- [ ] Sweep PORT-STATUS to zero `todo` for everything `strata-poly`'s qwen35moe + gemma4 + glm5next paths can reach; `native_*`/`qsa` TC variants stay `fp32-fallback (TC n/a on gfx1013)` honestly forever.
- [ ] Port `strata` (the Qwen3.8 engine)? **Out of scope unless asked** — `strata-poly` carries Qwen3.6; note in docs.
- [ ] PR-readiness pass: `git diff main --stat` review, UPSTREAM.md line, commit messages self-contained.

---

## Test plan summary
Unit gate every commit: `ctest --test-dir build-vk`. Model gate from Phase 2 on: `strata-poly-vk --no-graphs` must advance its trap position (strictly fewer named traps each session) and never produce output from an unimplemented path. Formal gate: Phase 5's teacher-forced harness at the fork's KL/spread numbers. Nothing merges with a silently widened tolerance or a trap that returns zeros — `missing()` exits 86 on purpose.
