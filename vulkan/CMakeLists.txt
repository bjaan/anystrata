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
  set(comp ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/${name}.comp)
  set(spv  ${CMAKE_CURRENT_BINARY_DIR}/spv/${name}.spv)
  add_custom_command(OUTPUT ${spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.0 -O ${comp} -o ${spv}
    DEPENDS ${comp} ${VK_GLSLC}
    COMMENT "vk-glslc(${GLSLC_MODE}) ${name}.comp")
  set(inc ${CMAKE_CURRENT_BINARY_DIR}/spv/${name}.spv.h)
  add_custom_command(OUTPUT ${inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${spv} ${inc} ${name}
    DEPENDS ${spv}
    COMMENT "embed ${name}.spv")
  add_library(strata_vk_kernel_${name} STATIC src/kernels/${name}_vk.cpp ${inc})
  target_include_directories(strata_vk_kernel_${name} PUBLIC ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
  target_link_libraries(strata_vk_kernel_${name} PUBLIC strata_vk_runtime)
endfunction()

strata_vk_kernel(s2_gemv)
strata_vk_kernel(dequant_s2)

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
