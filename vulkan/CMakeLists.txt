# Strata on Vulkan GPUs - the engine's Vulkan backend (vulkan/, the BC-250 port).
#
# The shape is sycl/'s, not hip_compat's: this is its own project that takes the engine's host sources from ../src
# and carries Vulkan-migrated device code beside them, so upstream's CUDA and HIP trees stay byte-for-byte.
# A `.cu` becomes `vulkan/src/kernels/<name>.comp` (GLSL, compiled to SPIR-V at build time by glslc) plus a host
# launcher `vulkan/src/kernels/<name>.vk.cpp` implementing the SAME include/strata/kernels/<name>.hpp contract.
# The CUDA runtime calls in ../src's host files (cudaMalloc/cudaMemcpy/streams/events) resolve against
# vulkan/include/strata/vulkan_compat/ on the include path, the way include/strata/platform/hip_compat/ maps them
# to HIP.
#
#   cmake -S vulkan -B build-vulkan && cmake --build build-vulkan && ctest --build-dir build-vulkan
#
# It needs: a Vulkan loader + headers (libvulkan, vulkan-headers), glslc (SPIRV-Tools), a C++20 compiler, and a
# device whose driver exposes VK_KHR_shader_integer_dot_product and VK_EXT_subgroup_size_control (the BC-250's
# RADV GFX1013 does; both are executed and asserted by vk_device_probe).
cmake_minimum_required(VERSION 3.24)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt _strata_proj REGEX "^project\\(strata VERSION [0-9.]+")
string(REGEX MATCH "VERSION ([0-9.]+)" _m "${_strata_proj}")
if(CMAKE_MATCH_1)
  set(STRATA_VK_VERSION ${CMAKE_MATCH_1})
else()
  set(STRATA_VK_VERSION 0.1.45)
endif()
project(strata_vulkan VERSION ${STRATA_VK_VERSION} LANGUAGES C CXX)
set(STRATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")

find_package(Vulkan REQUIRED)

# Front-end selection: the kernels' .comp files compile to .spv next to the build tree and are embedded into
# strata_vk_kernels as a byte array by vulkan/tools/embed_spv.py (no downloads). We prefer a VENDORED glslang
# build if one exists (the llama.cpp fork at third_party/llama.cpp was expected to carry thirdparty/glslang; it
# does not — its .gitmodules is empty and no gitlinks are tracked, so it cannot be `submodule update`d; see
# vulkan/PORT-NOTES.md). Failing that, the system glslc. Both are driven through tools/vk-glslc.sh, a shim that
# accepts the glslc-style invocation and translates to glslangValidator's CLI when that front-end is selected.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/spv)
find_program(GLSLANG_VENDORED NAMES glslangValidator PATHS ~build-glslang/glslang NO_DEFAULT_PATH)
find_program(GLSLC NAMES glslc glslc.bat PATHS /usr/bin /usr/local/bin ENV PATH)
if(GLSLANG_VENDORED)
  set(GLSLC_TOOL ${GLSLANG_VENDORED})
  set(GLSLC_MODE glslang)
  message(STATUS "vk kernel compiler: vendored glslangValidator ${GLSLANG_VENDORED}")
elseif(GLSLC)
  set(GLSLC_TOOL ${GLSLC})
  set(GLSLC_MODE glslc)
  message(STATUS "vk kernel compiler: system glslc ${GLSLC}")
else()
  message(FATAL_ERROR "neither a vendored glslangValidator nor glslc (spirv-tools package) found; the .comp kernels cannot be built")
endif()
set(VK_GLSLC ${CMAKE_CURRENT_SOURCE_DIR}/tools/vk-glslc.sh)

# ---- the runtime: device, arena, queue, buffers, the compat headers the ../src host files compile against
add_library(strata_vk_runtime STATIC
  src/runtime/vk_device.cpp
  src/runtime/vk_runtime.cpp)
target_include_directories(strata_vk_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STRATA_ROOT}/include)
target_link_libraries(strata_vk_runtime PUBLIC Vulkan::Vulkan)

# ---- vk_compat: the CUDA-shaped surface (cudaMalloc -> arena, cudaMemcpyAsync -> queue copies,
# cudaStream_t -> stream sync, cudaEvent_t -> host-clock timing, cublas -> the CPU fallback in vk_blas.cpp)
# for ../src's host translation units.  BEFORE so these win over any CUDA toolkit headers on the path, the
# way hip_compat does in the HIP build (CMakeLists.txt:166).
add_library(strata_vk_blas STATIC src/runtime/vk_blas.cpp)
target_include_directories(strata_vk_blas PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/vulkan_compat
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STRATA_ROOT}/include)
target_link_libraries(strata_vk_blas PUBLIC strata_vk_runtime)
add_library(strata_vk_compat INTERFACE)
target_include_directories(strata_vk_compat BEFORE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/vulkan_compat)
target_link_libraries(strata_vk_compat INTERFACE strata_vk_runtime strata_vk_blas)

# ---- the kernels: each .comp -> .spv -> an embedded array + a host launcher .vk.cpp
function(strata_vk_kernel name)
  cmake_parse_arguments(K "" "SRC_DIR;COMP;LAUNCHER" "" ${ARGN})
  set(_srcdir ${K_SRC_DIR})
  if(NOT _srcdir)
    set(_srcdir ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels)
  endif()
  set(_comp ${K_COMP})
  if(NOT _comp)
    set(_comp ${name}.comp)
  endif()
  set(_launcher ${K_LAUNCHER})
  if(NOT _launcher)
    set(_launcher src/kernels/${name}_vk.cpp)
  endif()
  set(comp ${_srcdir}/${_comp})
  set(spv  ${CMAKE_CURRENT_BINARY_DIR}/spv/${name}.spv)
  add_custom_command(OUTPUT ${spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O ${comp} -o ${spv}
    DEPENDS ${comp} ${VK_GLSLC}
    COMMENT "vk-glslc(${GLSLC_MODE}) ${name}.comp")
  set(inc ${CMAKE_CURRENT_BINARY_DIR}/spv/${name}.spv.h)
  add_custom_command(OUTPUT ${inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${spv} ${inc} ${name}
    DEPENDS ${spv}
    COMMENT "embed ${name}.spv")
  add_library(strata_vk_kernel_${name} STATIC ${_launcher} ${inc})
  target_include_directories(strata_vk_kernel_${name} PUBLIC ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
  target_link_libraries(strata_vk_kernel_${name} PUBLIC strata_vk_runtime)
endfunction()

strata_vk_kernel(s2_gemv)
strata_vk_kernel(dequant_s2)

# Task 2.1 (Phase 2): the model-demanded decode trio.  copy2 lives in src/poly/kernels/
# (the poly namespace contract); rms_norm builds ONE launcher lib against four workgroup-size
# variants (-DRMS_THREADS=N, matching the CUDA launcher's sum_threads(n) choice), compiled at
# SPIR-V 1.3 for the subgroup shuffle chain.  native_quantize_q8_1 is the strata::kernels contract.
set(_rms_spvs "")
foreach(_n 128 256 512 1024)
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/rms_norm_${_n}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DRMS_THREADS=${_n}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/rms_norm.comp -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/rms_norm.comp ${VK_GLSLC}
    COMMENT "vk-glslc rms_norm.comp RMS_THREADS=${_n}")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/rms_norm_${_n}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} rms_norm_${_n}
    DEPENDS ${_spv})
  list(APPEND _rms_spvs ${_inc})
endforeach()
add_library(strata_vk_kernel_rms_norm STATIC src/poly/kernels/rms_norm_vk.cpp ${_rms_spvs})
target_include_directories(strata_vk_kernel_rms_norm PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(strata_vk_kernel_rms_norm PUBLIC strata_vk_runtime strata_vk_compat)

strata_vk_kernel(native_quantize_q8_1 COMP quantize_q8_1.comp LAUNCHER src/kernels/native_quantize_q8_1_vk.cpp)
# Task 2.2: native_mmvq for IQ2_XXS (type 16) — the model's routed experts. The .spv embeds under
# the entry point's name (native_mmvq.spv.h) so the launcher includes it like the others.
strata_vk_kernel(native_mmvq COMP mmvq_iq2_xxs.comp LAUNCHER src/kernels/native_mmvq_vk.cpp)
strata_vk_kernel(copy2 SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels LAUNCHER src/poly/kernels/copy2_vk.cpp)

# ---- the trap roster (Task 1.2): every strata::kernels entry point the PORT-STATUS ledger still marks
# `todo` gets a definition that names itself and stops the run (missing.cpp; the definitions are
# generated from the headers' exact signatures by vulkan/tools/gen_stubs.py -> src/engine/stubs_gen.cpp).
# LINK ORDER MATTERS: the real launcher libs (strata_vk_kernel_*) must come BEFORE strata_vk_stubs on
# any link line — the stubs exist only so the link COMPLETES and the run names what is missing; a symbol
# defined in a kernel lib resolves from that lib, never from the trap. (Static libs: first definition
# wins; strata_vk_stubs is the tail.)
add_library(strata_vk_stubs STATIC src/engine/missing.cpp src/engine/stubs_gen.cpp)
target_include_directories(strata_vk_stubs PUBLIC
  ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/vulkan_compat)
target_link_libraries(strata_vk_stubs PUBLIC strata_vk_runtime)

# ---- probe: asserts the device's Vulkan surface the port needs (dp4a executes, subgroups force to 32),
# and prints the device line the engine logs at start. Its two modules come from spirv-as: this distro's
# glslc frontend predates GL_EXT_shader_integer_dot_product, and hand-assembled SPIR-V is what ran on the
# BC-250 during the port's design (vk_probe: dp4a 10/10, forced-32 subgroup 321/321).
find_program(SPIRV_AS NAMES spirv-as)
if(NOT SPIRV_AS)
  message(FATAL_ERROR "spirv-as not found (spirv-tools package)")
endif()
find_program(SPIRV_VAL NAMES spirv-val)
set(_probe_dot_spv  ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_dot.spv)
set(_probe_sg_spv   ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_subgroup.spv)
add_custom_command(OUTPUT ${_probe_dot_spv}
  COMMAND ${SPIRV_AS} --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_dot.spvasm -o ${_probe_dot_spv}
  COMMAND ${SPIRV_VAL} --target-env vulkan1.2 ${_probe_dot_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_dot.spvasm)
add_custom_command(OUTPUT ${_probe_sg_spv}
  COMMAND ${SPIRV_AS} --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_subgroup.spvasm -o ${_probe_sg_spv}
  COMMAND ${SPIRV_VAL} --target-env vulkan1.2 ${_probe_sg_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_subgroup.spvasm)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_dot.spv.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_probe_dot_spv} ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_dot.spv.h probe_dot
  DEPENDS ${_probe_dot_spv})
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_subgroup.spv.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_probe_sg_spv} ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_subgroup.spv.h probe_subgroup
  DEPENDS ${_probe_sg_spv})
set(_probe_wg_spv   ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_wg64k.spv)
set(_probe_span_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_spans.spv)
add_custom_command(OUTPUT ${_probe_wg_spv}
  COMMAND ${SPIRV_AS} --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_wg64k.spvasm -o ${_probe_wg_spv}
  COMMAND ${SPIRV_VAL} --target-env vulkan1.2 ${_probe_wg_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_wg64k.spvasm)
add_custom_command(OUTPUT ${_probe_span_spv}
  COMMAND ${SPIRV_AS} --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_spans.spvasm -o ${_probe_span_spv}
  COMMAND ${SPIRV_VAL} --target-env vulkan1.2 ${_probe_span_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_spans.spvasm)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_wg64k.spv.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_probe_wg_spv} ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_wg64k.spv.h probe_wg64k
  DEPENDS ${_probe_wg_spv})
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_spans.spv.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_probe_span_spv} ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_spans.spv.h probe_spans
  DEPENDS ${_probe_span_spv})
add_executable(vk_device_probe src/probe/device_probe.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_dot.spv.h ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_subgroup.spv.h
  ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_wg64k.spv.h ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_spans.spv.h)
target_include_directories(vk_device_probe PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(vk_device_probe PRIVATE strata_vk_runtime)
enable_testing()
add_test(NAME vk_device_probe COMMAND vk_device_probe)

# ---- probe case E (Task 1.3 step 1, the doorbell): can a RUNNING RADV kernel see host stores to
# HOST_VISIBLE|HOST_COHERENT memory? The sycl/ port's Arc A750 landmine (sycl_queue.hpp:43) says some
# drivers never deliver them; the pinned twins' design is decided by THIS box's answer, measured before
# anything is built on it. Self-contained raw Vulkan (the runtime's submit-always-waits path would make
# "while it runs" unmeasurable), so the .comp goes through glslc + embed_spv like everything else.
set(_probe_db_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_doorbell.spv)
add_custom_command(OUTPUT ${_probe_db_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.0 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_doorbell.comp -o ${_probe_db_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/probe/kernels/probe_doorbell.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) probe_doorbell.comp")
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_doorbell.spv.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_probe_db_spv} ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_doorbell.spv.h probe_doorbell
  DEPENDS ${_probe_db_spv})
add_executable(vk_doorbell_probe src/probe/doorbell_probe.cpp ${CMAKE_CURRENT_BINARY_DIR}/spv/probe_doorbell.spv.h)
target_include_directories(vk_doorbell_probe PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(vk_doorbell_probe PRIVATE Vulkan::Vulkan)
add_test(NAME vk_doorbell_probe COMMAND vk_doorbell_probe)

# ---- parity: the tree's own s2_gemv parity contract, on Vulkan. The reference transcription (validated
# against ggml upstream) stays in ../src; only the device call is this backend's.
add_executable(s2_gemv_vk_parity src/tests/s2_gemv_vk_parity.cpp)
target_include_directories(s2_gemv_vk_parity PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(s2_gemv_vk_parity PRIVATE strata_vk_kernel_s2_gemv strata_vk_runtime strata_vk_compat)
add_test(NAME s2_gemv_vk_parity COMMAND s2_gemv_vk_parity --selftest)

add_executable(dequant_s2_vk_parity src/tests/dequant_s2_vk_parity.cpp)
target_include_directories(dequant_s2_vk_parity PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dequant_s2_vk_parity PRIVATE strata_vk_kernel_dequant_s2 strata_vk_runtime strata_vk_compat)
add_test(NAME dequant_s2_vk_parity COMMAND dequant_s2_vk_parity --selftest)

# Task 2.2: bitwise parity for native_mmvq's IQ2_XXS path (host transcription of mmvq_kernel<16>).
add_executable(native_mmvq_vk_parity src/tests/native_mmvq_vk_parity.cpp)
target_include_directories(native_mmvq_vk_parity PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(native_mmvq_vk_parity PRIVATE strata_vk_kernel_native_mmvq strata_vk_kernel_native_quantize_q8_1 strata_vk_runtime strata_vk_compat strata_vk_stubs)   # stubs LAST: the launcher's missing() trap for unported types; the real native_mmvq resolves from its kernel lib first
add_test(NAME native_mmvq_vk_parity COMMAND native_mmvq_vk_parity --selftest)

# Task 2.1: unit parity for the model-demanded trio (poly's copy2/rms_norm + the q8_1 quantizer).
add_executable(poly_kernels_vk_parity src/tests/poly_kernels_vk_parity.cpp)
target_include_directories(poly_kernels_vk_parity PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(poly_kernels_vk_parity PRIVATE strata_vk_kernel_copy2 strata_vk_kernel_rms_norm strata_vk_kernel_native_quantize_q8_1 strata_vk_poly_kernels strata_vk_runtime strata_vk_compat strata_vk_stubs)   # stubs LAST: missing() for the poly trap lib; real launchers already resolved
# strata_vk_poly_kernels supplies the host-defined sum_threads() the rms_norm launcher asks for; its
# traps stay dormant here (the test calls only ported entry points).
add_test(NAME poly_kernels_vk_parity COMMAND poly_kernels_vk_parity)

# scratch diagnostic (not in ctest): one known Q8_1 block, raw bytes.
add_executable(quant_dump src/tests/quant_dump.cpp)
target_include_directories(quant_dump PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(quant_dump PRIVATE strata_vk_kernel_native_quantize_q8_1 strata_vk_runtime strata_vk_compat)

# ---- Task 1.2's proof: the trap roster. Arm 1 (default): calling an unported entry point names it
# and stops the run (exit 86) — checked by a shell wrapper so ctest sees both the name and the code.
# Arm 2 (STRATA_VK_STUBS=log): one STUB line per entry, the call returns.
add_executable(vk_stubs_trap src/tests/vk_stubs_trap.cpp)
target_include_directories(vk_stubs_trap PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vk_stubs_trap PRIVATE strata_vk_stubs)
# PASS_REGULAR_EXPRESSION makes ctest ignore the exit code (the trap's _Exit(86) is what the message
# below witnesses; the exit code itself is asserted in the test program's own output on the log arm).
add_test(NAME vk_stubs_trap COMMAND vk_stubs_trap)
add_test(NAME vk_stubs_trap_log COMMAND vk_stubs_trap log)
set_tests_properties(vk_stubs_trap PROPERTIES PASS_REGULAR_EXPRESSION "strata::kernels::rope_neox_apply has no Vulkan implementation yet")
set_tests_properties(vk_stubs_trap_log PROPERTIES ENVIRONMENT "STRATA_VK_STUBS=log"
  PASS_REGULAR_EXPRESSION "strata-vk: STUB strata::kernels::rope_neox_apply")

# ---- the core twins (Task 1.3): device.hpp / pinned.hpp / vmm.hpp contracts on Vulkan.
# vk_device_impl.cpp implements include/strata/core/device.hpp over strata::vk (src/core/device.cu is
# NOT compiled by this backend); vk_pinned.cpp is pinned.cu's body carried unchanged against the compat
# layer with the PCIe probe routed to a named skip; vk_vmm.cpp gives the elastic-K/V interface a
# grow-never arena fallback. PinnedRing (pinned_ring.hpp) is header-inline over the compat calls.
add_library(strata_vk_core STATIC
  src/core/vk_device_impl.cpp
  src/core/vk_pinned.cpp
  src/core/vk_vmm.cpp)
target_include_directories(strata_vk_core PUBLIC
  ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_vk_core PUBLIC strata_vk_compat)

add_executable(vk_core_smoke src/tests/vk_core_smoke.cpp ${STRATA_ROOT}/src/platform/memory.cpp)
target_include_directories(vk_core_smoke PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vk_core_smoke PRIVATE strata_vk_core)
add_test(NAME vk_core_smoke COMMAND vk_core_smoke)

# ---- Task 1.1's compile proof: TWO real engine host files (src/core/layout.cpp, src/core/weights.cpp -
# they call cudaMalloc/cudaMemcpy/cudaHostAlloc straight through <cuda_runtime.h>) compiled UNCHANGED
# against vulkan_compat, plus a smoke main that exercises the arena round-trip and the cublas CPU fallback.
# No CUDA toolkit is on this box: if this links, the compat surface is real.  device.cu/pinned.cu/graph.cpp
# are NOT in this target - their Vulkan twins arrive in Task 1.3.
add_executable(vk_compat_smoke
  src/tests/vk_compat_smoke.cpp
  ${STRATA_ROOT}/src/core/layout.cpp
  ${STRATA_ROOT}/src/core/weights.cpp
  ${STRATA_ROOT}/src/platform/memory.cpp)
target_include_directories(vk_compat_smoke PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vk_compat_smoke PRIVATE strata_vk_compat)
add_test(NAME vk_compat_smoke COMMAND vk_compat_smoke)

# ---- Task 1.4: strata-poly-vk - the engine's strata-poly target (root CMakeLists.txt:2059) mirrored
# onto Vulkan with the host sources UNCHANGED.
#
# The .cu libraries of the CUDA target are NOT compiled here. src/poly/kernels.cu and kernels_dsv4.cu hold
# the host wrappers AND the __global__ kernels in one TU - with no nvcc, the sycl port's "migrated copy"
# rule lands here as the trap roster: include/strata/poly/kernels.hpp's whole contract is defined by name
# (src/poly/stubs_gen.cpp, generated by tools/gen_poly_stubs.py), plus host-only parts carried honestly:
# sum_threads/attend_split_floats are pure arithmetic and blas_create/blas_destroy is the compat cuBLAS
# handle's lifecycle (a real host library here - the CPU fallback GEMM). strata_poly_mmq (llama.cpp's
# ggml-cuda MMQ) is trapped by its twin src/poly/mmq_vk.cpp: fits() answers the honest false (no MMQ tile
# built for this backend yet; PORT-STATUS: mmq prompt kernels trapped, Phase 3 covers prefill), the
# constructor and kernel sites name themselves. strata_kernels' 49 CUDA .cu files are already replaced by
# strata_vk_stubs (Task 1.2); graph.cpp compiles with its cudaGraph calls landing on the compat graph
# traps, and --no-graphs keeps the engine from reaching them; the driver-API VMM sites in engine.cpp
# resolve to the compat cuda.h's "not supported" answers, which trip engine.cpp's own vmm_ off-switch to
# the arena path (vulkan/include/strata/vulkan_compat/cuda.h). device.cu/pinned.cu -> the Task-1.3 twins.
# ggml-base + ggml-cpu come from the same pinned llama.cpp checkout the CUDA build uses (the CPU experts).
find_package(Threads REQUIRED)
set(STRATA_VK_GGML_DIR ${STRATA_ROOT}/third_party/llama.cpp/ggml CACHE PATH "ggml checkout for strata-poly-vk")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_VK_GGML_DIR} ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

add_library(strata_vk_poly_kernels STATIC src/poly/stubs_gen.cpp src/poly/mmq_vk.cpp)
target_include_directories(strata_vk_poly_kernels PUBLIC
  ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/vulkan_compat)
target_link_libraries(strata_vk_poly_kernels PUBLIC strata_vk_runtime)

add_executable(strata-poly-vk
  ${STRATA_ROOT}/src/poly/main.cpp
  ${STRATA_ROOT}/src/poly/engine.cpp
  ${STRATA_ROOT}/src/poly/glm5next.cpp
  ${STRATA_ROOT}/src/poly/qwen35.cpp
  ${STRATA_ROOT}/src/poly/gemma4.cpp
  ${STRATA_ROOT}/src/poly/deepseek4.cpp
  ${STRATA_ROOT}/src/poly/lfm2moe.cpp
  ${STRATA_ROOT}/src/poly/cpu_experts.cpp
  ${STRATA_ROOT}/src/core/conversation_memory.cpp
  ${STRATA_ROOT}/src/core/graph.cpp          # cudaGraph* -> compat graph traps (--no-graphs stays off them)
  ${STRATA_ROOT}/src/core/weights.cpp
  ${STRATA_ROOT}/src/core/layout.cpp
  ${STRATA_ROOT}/src/platform/memory.cpp
  ${STRATA_ROOT}/src/spec/suffix_drafter.cpp
  ${STRATA_ROOT}/src/spec/controller.cpp
  ${STRATA_ROOT}/src/spec/draft_policy.cpp
  ${STRATA_ROOT}/src/spec/draft_source.cpp)
target_include_directories(strata-poly-vk PRIVATE ${STRATA_ROOT}/include ${STRATA_VK_GGML_DIR}/include)
target_link_libraries(strata-poly-vk PRIVATE
  strata_vk_kernel_s2_gemv strata_vk_kernel_dequant_s2   # real launchers FIRST (link-order rule above)
  strata_vk_kernel_copy2 strata_vk_kernel_rms_norm strata_vk_kernel_native_quantize_q8_1
  strata_vk_kernel_native_mmvq
  strata_vk_poly_kernels strata_vk_core
  ggml-cpu ggml-base
  strata_vk_compat strata_vk_stubs                      # stubs last: they only let the link complete
  Threads::Threads)
