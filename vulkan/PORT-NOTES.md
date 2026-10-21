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
