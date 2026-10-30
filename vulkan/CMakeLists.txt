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
find_program(GLSLANG_VENDORED NAMES glslangValidator PATHS "$ENV{HOME}/build-glslang/glslang" NO_DEFAULT_PATH)
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
  message(FATAL_ERROR "no kernel compiler found: install glslc (Debian/Ubuntu/Mint: apt install glslc, from the shaderc source; universe on Ubuntu/Mint) or build glslang, and the .comp kernels can be built")
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
# Phase 3 (attempt 2): the GPU tiled SGEMM under the cublas shim. One .comp, four compiled
# variants (TRANS_A x F16AB); embedded .spv.h ride the same vk-glslc + embed_spv pattern as
# the kernel libs, but the launcher lives IN vk_blas.cpp (no separate _vk.cpp), so the blas
# target carries the four generated headers itself.
set(_mm_spvs "")
foreach(_ta 0 1)
  foreach(_f16 0 1)
    set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mm_gemm_t${_ta}f${_f16}.spv)
    add_custom_command(OUTPUT ${_spv}
      COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
              ${VK_GLSLC} --target-env=vulkan1.3 -O -DTRANS_A=${_ta} -DF16AB=${_f16}
              ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/mm_gemm.comp -o ${_spv}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/mm_gemm.comp ${VK_GLSLC}
      COMMENT "vk-glslc mm_gemm.comp TRANS_A=${_ta} F16AB=${_f16}")
    set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mm_gemm_t${_ta}f${_f16}.spv.h)
    add_custom_command(OUTPUT ${_inc}
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} mm_gemm_t${_ta}f${_f16}
      DEPENDS ${_spv})
    list(APPEND _mm_spvs ${_inc})
  endforeach()
endforeach()
add_library(strata_vk_blas STATIC src/runtime/vk_blas.cpp ${_mm_spvs})
target_include_directories(strata_vk_blas PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/vulkan_compat
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/spv
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

# Task 2.4: poly gemv_f32 — five workgroup-size variants (-DGEMV_THREADS=N: 32 for the CUDA
# wrapper's narrow-input arm + the four sum_threads values), same shape as rms_norm above.
set(_gemv_spvs "")
foreach(_n 32 128 256 512 1024)
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/gemv_${_n}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DGEMV_THREADS=${_n}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/gemv.comp -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/gemv.comp ${VK_GLSLC}
    COMMENT "vk-glslc gemv.comp GEMV_THREADS=${_n}")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/gemv_${_n}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} gemv_${_n}
    DEPENDS ${_spv})
  list(APPEND _gemv_spvs ${_inc})
endforeach()
add_library(strata_vk_kernel_gemv STATIC src/poly/kernels/gemv_vk.cpp ${_gemv_spvs})
target_include_directories(strata_vk_kernel_gemv PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(strata_vk_kernel_gemv PUBLIC strata_vk_runtime strata_vk_compat)

# Task 2.16: poly gemv_bf16 — the bf16-weight twin of gemv.comp (ldw's exact 16-bit shift; the
# partition contract is the CUDA gemv<> wrapper's, shared by both), same five -DGEMV_THREADS variants.
set(_gemv_bf16_spvs "")
foreach(_n 32 128 256 512 1024)
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/gemv_bf16_${_n}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DGEMV_THREADS=${_n}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/gemv_bf16.comp -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/gemv_bf16.comp ${VK_GLSLC}
    COMMENT "vk-glslc gemv_bf16.comp GEMV_THREADS=${_n}")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/gemv_bf16_${_n}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} gemv_bf16_${_n}
    DEPENDS ${_spv})
  list(APPEND _gemv_bf16_spvs ${_inc})
endforeach()
add_library(strata_vk_kernel_gemv_bf16 STATIC src/poly/kernels/gemv_bf16_vk.cpp ${_gemv_bf16_spvs})
target_include_directories(strata_vk_kernel_gemv_bf16 PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(strata_vk_kernel_gemv_bf16 PUBLIC strata_vk_runtime strata_vk_compat)

# Task 2.4: poly conv4_silu — three compiled variants of one .comp (CONV_VARIANT=0 the token-walk
# kernel, =1 the rows kernel) plus the separate conv4_state.comp the rows arm pairs with.
set(_conv_spvs "")
foreach(_v walk rows)
  if(_v STREQUAL walk)
    set(_cv 0)
  else()
    set(_cv 1)
  endif()
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/conv4_${_v}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DCONV_VARIANT=${_cv}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/conv4_silu.comp -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/conv4_silu.comp ${VK_GLSLC}
    COMMENT "vk-glslc conv4_silu.comp CONV_VARIANT=${_cv}")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/conv4_${_v}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} conv4_${_v}
    DEPENDS ${_spv})
  list(APPEND _conv_spvs ${_inc})
endforeach()
set(_conv_state_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/conv4_state.spv)
add_custom_command(OUTPUT ${_conv_state_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O
          ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/conv4_state.comp -o ${_conv_state_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/conv4_state.comp ${VK_GLSLC}
  COMMENT "vk-glslc conv4_state.comp")
set(_conv_state_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/conv4_state.spv.h)
add_custom_command(OUTPUT ${_conv_state_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_conv_state_spv} ${_conv_state_inc} conv4_state
  DEPENDS ${_conv_state_spv})
add_library(strata_vk_kernel_conv4_silu STATIC src/poly/kernels/conv4_silu_vk.cpp ${_conv_spvs} ${_conv_state_inc})
target_include_directories(strata_vk_kernel_conv4_silu PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(strata_vk_kernel_conv4_silu PUBLIC strata_vk_runtime strata_vk_compat)

# Task 2.7: poly elementwise batch (axpy, swiglu_clamp, mul_sigmoid, add_scaled) — one .comp,
# three -DEW_VARIANT arms (0 axpy/add_scaled pass, 1 swiglu_clamp, 2 mul_sigmoid), one launcher lib.
set(_ew_spvs "")
foreach(_v axpy swiglu swiglu_rows mul_sigmoid)   # Task 2.8: swiglu_rows = EW_VARIANT 3
  if(_v STREQUAL axpy)
    set(_ev 0)
  elseif(_v STREQUAL swiglu)
    set(_ev 1)
  elseif(_v STREQUAL swiglu_rows)
    set(_ev 3)
  else()
    set(_ev 2)
  endif()
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/elementwise_${_v}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DEW_VARIANT=${_ev}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/elementwise.comp -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/elementwise.comp ${VK_GLSLC}
    COMMENT "vk-glslc elementwise.comp EW_VARIANT=${_ev}")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/elementwise_${_v}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} elementwise_${_v}
    DEPENDS ${_spv})
  list(APPEND _ew_spvs ${_inc})
endforeach()
add_library(strata_vk_kernel_elementwise STATIC src/poly/kernels/elementwise_vk.cpp ${_ew_spvs})
target_include_directories(strata_vk_kernel_elementwise PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(strata_vk_kernel_elementwise PUBLIC strata_vk_runtime strata_vk_compat)

# Task 2.10: poly head_norm_rope - two workgroup-size variants (-DHNR_THREADS=N matching the
# CUDA launcher's `dim >= 256 ? 256 : 128`, kernels.cu:2445), same shape as rms_norm above.
set(_hnr_spvs "")
foreach(_n 128 256)
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/head_norm_rope_${_n}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DHNR_THREADS=${_n}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/head_norm_rope.comp -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/head_norm_rope.comp ${VK_GLSLC}
    COMMENT "vk-glslc head_norm_rope.comp HNR_THREADS=${_n}")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/head_norm_rope_${_n}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} head_norm_rope_${_n}
    DEPENDS ${_spv})
  list(APPEND _hnr_spvs ${_inc})
endforeach()
add_library(strata_vk_kernel_head_norm_rope STATIC src/poly/kernels/head_norm_rope_vk.cpp ${_hnr_spvs})
target_include_directories(strata_vk_kernel_head_norm_rope PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(strata_vk_kernel_head_norm_rope PUBLIC strata_vk_runtime strata_vk_compat)

# Task 2.10: poly store_f16 (the f16 KV write; store_f16_at stays a trap - no card-time row
# read wired for it yet). f16 packing is RN via f16_rn_bits, NOT packHalf2x16 (law 1).
strata_vk_kernel(store_f16 SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels COMP store_f16.comp LAUNCHER src/poly/kernels/store_f16_vk.cpp)

# q8_0 KV cache (the 64K-context unlock, law 11): poly store_q8 (the q8 KV write). store_q8_at
# stays a trap (no card-time row read wired under --no-graphs). rint ties-to-even (q8_rint), NOT
# GLSL round(); scales are f16_rn_bits; codes/scales word layouts per store_f16's law.
strata_vk_kernel(store_q8 SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels COMP store_q8.comp LAUNCHER src/poly/kernels/store_q8_vk.cpp)

# Phase 4.2: the _at twins (card-time rows from the engine's at_ cell; the graph replay path).
strata_vk_kernel(store_f16_at SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels COMP store_f16_at.comp LAUNCHER src/poly/kernels/store_f16_at_vk.cpp)
strata_vk_kernel(store_q8_at SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels COMP store_q8_at.comp LAUNCHER src/poly/kernels/store_q8_at_vk.cpp)

# Task 2.19: poly likeliest (the sampler argmax; one workgroup of 1024 like the CUDA <<<1,1024>>>).
strata_vk_kernel(likeliest SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels COMP likeliest.comp LAUNCHER src/poly/kernels/likeliest_vk.cpp)

# Hybrid bring-up: poly bf16_to_f32 (the exact bf16 -> f32 widening; demanded by the long-context
# prefill when a BF16 dense matrix crosses kStageRows into the compat cuBLAS gemm_f32 path).
strata_vk_kernel(bf16_to_f32 SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels COMP bf16_to_f32.comp LAUNCHER src/poly/kernels/bf16_to_f32_vk.cpp)

# Hybrid bring-up: poly gemm_f32 - a HOST launcher (no .comp): the CUDA body is pure cuBLAS
# (cublasSgemm / StridedBatched), real in the compat surface (vk_blas.cpp, ordered fp32).
add_library(strata_vk_kernel_gemm_f32 STATIC src/poly/kernels/gemm_f32_vk.cpp)
target_include_directories(strata_vk_kernel_gemm_f32 PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_vk_kernel_gemm_f32 PUBLIC strata_vk_runtime strata_vk_compat strata_vk_blas)

# Task 2.11: poly attend - the general fp16-KV chain: one score kernel (reg<4,1,false>, the
# model's dim-128 per=8 arm), one softmax kernel, and THREE value arms compiled from one .comp
# (-DATT_VARIANT 0 = attn_value_t<1>, 1 = attn_value_t<2>, 2 = the plain kernel) so the launcher
# mirrors the CUDA wrapper's value-arm pick exactly. Every other attend arm (split/join dim-64,
# reg dim 256/512, plain score kernel, kq8/vq8) traps by name inside the launcher.
set(_att_spvs "")
set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_softmax.spv)
add_custom_command(OUTPUT ${_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O
          ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_softmax.comp -o ${_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_softmax.comp ${VK_GLSLC}
  COMMENT "vk-glslc attend_softmax.comp")
set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_softmax.spv.h)
add_custom_command(OUTPUT ${_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} attend_softmax
  DEPENDS ${_spv})
list(APPEND _att_spvs ${_inc})
# the four reg<> score variants the CUDA chain instantiates (kernels.cu:2488-2499)
foreach(_ns 4_1 8_2 4_2 4_4)
  string(REPLACE "_" ";" _nv ${_ns})
  list(GET _nv 0 _n)
  list(GET _nv 1 _st)
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_score_${_ns}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DATT_N=${_n} -DATT_STEPS=${_st}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_score.comp -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_score.comp ${VK_GLSLC}
    COMMENT "vk-glslc attend_score.comp ATT_N=${_n} ATT_STEPS=${_st}")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_score_${_ns}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} attend_score_${_ns}
    DEPENDS ${_spv})
  list(APPEND _att_spvs ${_inc})
  # the fused q8 score arm of the SAME variant (the kvq8 decode read; law 11's unlock)
  set(_spvq ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_score_${_ns}_q8.spv)
  add_custom_command(OUTPUT ${_spvq}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DATT_N=${_n} -DATT_STEPS=${_st} -DATT_Q8=1
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_score.comp -o ${_spvq}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_score.comp ${VK_GLSLC}
    COMMENT "vk-glslc attend_score.comp ATT_N=${_n} ATT_STEPS=${_st} Q8")
  set(_incq ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_score_${_ns}_q8.spv.h)
  add_custom_command(OUTPUT ${_incq}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spvq} ${_incq} attend_score_${_ns}_q8
    DEPENDS ${_spvq})
  list(APPEND _att_spvs ${_incq})
endforeach()
foreach(_n t1 t2 plain)
  if(_n STREQUAL "t1")
    set(_var 0)
  elseif(_n STREQUAL "t2")
    set(_var 1)
  else()
    set(_var 2)
  endif()
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_value_${_n}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DATT_VARIANT=${_var}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_value.comp -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_value.comp ${VK_GLSLC}
    COMMENT "vk-glslc attend_value.comp ATT_VARIANT=${_var}")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_value_${_n}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} attend_value_${_n}
    DEPENDS ${_spv})
  list(APPEND _att_spvs ${_inc})
  set(_spvq ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_value_${_n}_q8.spv)
  add_custom_command(OUTPUT ${_spvq}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O -DATT_VARIANT=${_var} -DATT_Q8=1
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_value.comp -o ${_spvq}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/attend_value.comp ${VK_GLSLC}
    COMMENT "vk-glslc attend_value.comp ATT_VARIANT=${_var} Q8")
  set(_incq ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_value_${_n}_q8.spv.h)
  add_custom_command(OUTPUT ${_incq}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spvq} ${_incq} attend_value_${_n}_q8
    DEPENDS ${_spvq})
  list(APPEND _att_spvs ${_incq})
endforeach()
add_library(strata_vk_kernel_attend STATIC src/poly/kernels/attend_vk.cpp ${_att_spvs})
target_include_directories(strata_vk_kernel_attend PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(strata_vk_kernel_attend PUBLIC strata_vk_runtime strata_vk_compat)

# Task: poly attend_batch - the prefill/multi-token chain: the wrapper's chunk/key-block loop
# over the compat CPU cublas's two strided-batched products (vk_blas.cpp, transcribed arg for
# arg), plus three .comp arms (group/ungroup one file with -DUNGROUP, softmax-block, scale).
# The kq8/vq8 and dim-64 flash arms are covered the way attend_vk.cpp traps: by name, here.
set(_ab_spvs "")
foreach(_n group group_ungroup softmax scale dequant)
  if(_n STREQUAL "group")
    set(_comp attend_batch_group.comp)
    set(_def "-DUNGROUP=0")
  elseif(_n STREQUAL "group_ungroup")
    set(_comp attend_batch_group.comp)
    set(_def "-DUNGROUP=1")
  elseif(_n STREQUAL "softmax")
    set(_comp attend_batch_softmax.comp)
    set(_def "")
  else()
    if(_n STREQUAL "dequant")
      set(_comp dequant_q8.comp)          # the q8 arm: rows of the q8 cache -> the f16 workspace
    else()
      set(_comp attend_batch_scale.comp)
    endif()
    set(_def "")
  endif()
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_batch_${_n}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O ${_def}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/${_comp} -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/${_comp} ${VK_GLSLC}
    COMMENT "vk-glslc ${_comp} -> attend_batch_${_n}")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/attend_batch_${_n}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} attend_batch_${_n}
    DEPENDS ${_spv})
  list(APPEND _ab_spvs ${_inc})
endforeach()
add_library(strata_vk_kernel_attend_batch STATIC src/poly/kernels/attend_batch_vk.cpp ${_ab_spvs})
target_include_directories(strata_vk_kernel_attend_batch PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(strata_vk_kernel_attend_batch PUBLIC strata_vk_runtime strata_vk_compat strata_vk_blas)

strata_vk_kernel(native_quantize_q8_1 COMP quantize_q8_1.comp LAUNCHER src/kernels/native_quantize_q8_1_vk.cpp)
# Task 2.2: native_mmvq for IQ2_XXS (type 16) — the model's routed experts. The .spv embeds under
# the entry point's name (native_mmvq.spv.h) so the launcher includes it like the others.
strata_vk_kernel(native_mmvq COMP mmvq_iq2_xxs.comp LAUNCHER src/kernels/native_mmvq_vk.cpp)
# Task 2.3: the Q5_K (type 13) variant rides in the SAME launcher lib: compile mmvq_q5_k.comp to
# its own embedded .spv.h and add it as a source (the launcher includes both headers).
set(_q5k_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q5_k.spv)
add_custom_command(OUTPUT ${_q5k_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q5_k.comp -o ${_q5k_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q5_k.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_q5_k.comp")
set(_q5k_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q5_k.spv.h)
add_custom_command(OUTPUT ${_q5k_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_q5k_spv} ${_q5k_inc} mmvq_q5_k
  DEPENDS ${_q5k_spv})
add_custom_target(mmvq_q5_k.spv_h DEPENDS ${_q5k_inc})
# Task 2.6: the Q6_K (type 14) variant, same pattern (mmvq_q6_k.comp -> its own embedded .spv.h).
set(_q6k_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q6_k.spv)
add_custom_command(OUTPUT ${_q6k_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q6_k.comp -o ${_q6k_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q6_k.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_q6_k.comp")
set(_q6k_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q6_k.spv.h)
add_custom_command(OUTPUT ${_q6k_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_q6k_spv} ${_q6k_inc} mmvq_q6_k
  DEPENDS ${_q6k_spv})
add_custom_target(mmvq_q6_k.spv_h DEPENDS ${_q6k_inc})
# Task 2.8: native_mmvq_slots for types 16/13/14 — one slots .comp per format (24-25 storage
# bindings: w/x/y per slot, + the iq2xxs grid table for type 16), same embed pattern.
set(_slots_names mmvq_slots_iq2_xxs mmvq_slots_q5_k mmvq_slots_q6_k mmvq_slots_q2_0)
foreach(_s ${_slots_names})
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/${_s}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/${_s}.comp -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/${_s}.comp ${VK_GLSLC}
    COMMENT "vk-glslc(${GLSLC_MODE}) ${_s}.comp")
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/${_s}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} ${_s}
    DEPENDS ${_spv})
  add_custom_target(${_s}.spv_h DEPENDS ${_inc})
endforeach()
# Task 2.9: the IQ3_XXS (type 18) variant, same pattern (mmvq_iq3_xxs.comp -> its own embedded
# .spv.h; the iq3xxs grid table rides the same arena-upload route as iq2xxs).
set(_iq3xxs_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_iq3_xxs.spv)
add_custom_command(OUTPUT ${_iq3xxs_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_iq3_xxs.comp -o ${_iq3xxs_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_iq3_xxs.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_iq3_xxs.comp")
set(_iq3xxs_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_iq3_xxs.spv.h)
add_custom_command(OUTPUT ${_iq3xxs_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_iq3xxs_spv} ${_iq3xxs_inc} mmvq_iq3_xxs
  DEPENDS ${_iq3xxs_spv})
add_custom_target(mmvq_iq3_xxs.spv_h DEPENDS ${_iq3xxs_inc})
# Task 2.12: the IQ4_XS (type 23) variant, same pattern (mmvq_iq4_xs.comp -> its own embedded
# .spv.h; no grid table - IQ4_XS decodes nibbles through the inlined 16-entry kvalues_iq4nl).
set(_iq4xs_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_iq4_xs.spv)
add_custom_command(OUTPUT ${_iq4xs_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_iq4_xs.comp -o ${_iq4xs_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_iq4_xs.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_iq4_xs.comp")
set(_iq4xs_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_iq4_xs.spv.h)
add_custom_command(OUTPUT ${_iq4xs_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_iq4xs_spv} ${_iq4xs_inc} mmvq_iq4_xs
  DEPENDS ${_iq4xs_spv})
add_custom_target(mmvq_iq4_xs.spv_h DEPENDS ${_iq4xs_inc})
# Task 2.13: the Q8_0 (type 8) variant, same pattern (mmvq_q8_0.comp -> its own embedded .spv.h;
# 34-byte blocks, no grid table - pure dp4a over int8 qs with fp16 scales).
set(_q80_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q8_0.spv)
add_custom_command(OUTPUT ${_q80_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q8_0.comp -o ${_q80_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q8_0.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_q8_0.comp")
set(_q80_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q8_0.spv.h)
add_custom_command(OUTPUT ${_q80_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_q80_spv} ${_q80_inc} mmvq_q8_0
  DEPENDS ${_q80_spv})
add_custom_target(mmvq_q8_0.spv_h DEPENDS ${_q80_inc})
# Task 2.14: the IQ2_S (type 22) variant, same pattern (mmvq_iq2_s.comp -> its own embedded
# .spv.h; the 1024-entry uint64 iq2s grid table rides the same arena-upload route as iq2xxs).
set(_iq2s_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_iq2_s.spv)
add_custom_command(OUTPUT ${_iq2s_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_iq2_s.comp -o ${_iq2s_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_iq2_s.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_iq2_s.comp")
set(_iq2s_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_iq2_s.spv.h)
add_custom_command(OUTPUT ${_iq2s_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_iq2s_spv} ${_iq2s_inc} mmvq_iq2_s
  DEPENDS ${_iq2s_spv})
add_custom_target(mmvq_iq2_s.spv_h DEPENDS ${_iq2s_inc})
# Task 2.15: the Q4_K (type 12) variant, same pattern (mmvq_q4_k.comp -> its own embedded .spv.h).
set(_q4k_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q4_k.spv)
add_custom_command(OUTPUT ${_q4k_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q4_k.comp -o ${_q4k_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q4_k.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_q4_k.comp")
set(_q4k_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q4_k.spv.h)
add_custom_command(OUTPUT ${_q4k_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_q4k_spv} ${_q4k_inc} mmvq_q4_k
  DEPENDS ${_q4k_spv})
add_custom_target(mmvq_q4_k.spv_h DEPENDS ${_q4k_inc})
# Task 2.17: the Q2_K (type 10) variant, same pattern (mmvq_q2_k.comp -> its own embedded .spv.h).
set(_q2k_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q2_k.spv)
add_custom_command(OUTPUT ${_q2k_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q2_k.comp -o ${_q2k_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q2_k.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_q2_k.comp")
set(_q2k_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q2_k.spv.h)
add_custom_command(OUTPUT ${_q2k_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_q2k_spv} ${_q2k_inc} mmvq_q2_k
  DEPENDS ${_q2k_spv} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py)
add_custom_target(mmvq_q2_k.spv_h DEPENDS ${_q2k_inc})
# Task 2.18: the Q3_K (type 11) variant, same pattern (mmvq_q3_k.comp -> its own embedded .spv.h).
set(_q3k_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q3_k.spv)
add_custom_command(OUTPUT ${_q3k_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q3_k.comp -o ${_q3k_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q3_k.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_q3_k.comp")
set(_q3k_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q3_k.spv.h)
add_custom_command(OUTPUT ${_q3k_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_q3k_spv} ${_q3k_inc} mmvq_q3_k
  DEPENDS ${_q3k_spv} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py)
add_custom_target(mmvq_q3_k.spv_h DEPENDS ${_q3k_inc})
# Hybrid bring-up: the Q2_0 (type 42) variant, same pattern (mmvq_q2_0.comp -> its own embedded
# .spv.h; 18-byte blocks, no grid table - the 2-bit codes unpack through the byte_perm chain).
set(_q20_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q2_0.spv)
add_custom_command(OUTPUT ${_q20_spv}
  COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
          ${VK_GLSLC} --target-env=vulkan1.3 -O ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q2_0.comp -o ${_q20_spv}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_q2_0.comp ${VK_GLSLC}
  COMMENT "vk-glslc(${GLSLC_MODE}) mmvq_q2_0.comp")
set(_q20_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/mmvq_q2_0.spv.h)
add_custom_command(OUTPUT ${_q20_inc}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_q20_spv} ${_q20_inc} mmvq_q2_0
  DEPENDS ${_q20_spv} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py)
add_custom_target(mmvq_q2_0.spv_h DEPENDS ${_q20_inc})
add_dependencies(strata_vk_kernel_native_mmvq mmvq_q5_k.spv_h mmvq_q6_k.spv_h mmvq_iq3_xxs.spv_h mmvq_iq4_xs.spv_h mmvq_q8_0.spv_h mmvq_iq2_s.spv_h mmvq_q4_k.spv_h mmvq_q2_k.spv_h mmvq_q3_k.spv_h mmvq_q2_0.spv_h mmvq_slots_iq2_xxs.spv_h mmvq_slots_q5_k.spv_h mmvq_slots_q6_k.spv_h mmvq_slots_q2_0.spv_h)
strata_vk_kernel(copy2 SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels LAUNCHER src/poly/kernels/copy2_vk.cpp)

# Task 2.5: poly gdn_step — five CUDA kernels, one launcher lib. gdn_norm/gdn_out compile per dim
# (32..256 in 32s: block_sum's part[] walk counts warps off blockDim.x, so the workgroup size IS
# the fp contract, like the gemv sizes); gdn_state as plain + reg<1,2,4,8> variants (the CUDA
# wrapper's switch (dim/WARP)), threads = columns*WARP = 1024 for every legal dim.
set(_gdn_spvs "")
macro(_gdn_variant comp out_name)   # ARGN = extra glslc -D flags
  set(_spv ${CMAKE_CURRENT_BINARY_DIR}/spv/${out_name}.spv)
  add_custom_command(OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E env VK_GLSLC_MODE=${GLSLC_MODE} VK_GLSLC_TOOL=${GLSLC_TOOL}
            ${VK_GLSLC} --target-env=vulkan1.3 -O ${ARGN}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/${comp} -o ${_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/kernels/${comp} ${VK_GLSLC})
  set(_inc ${CMAKE_CURRENT_BINARY_DIR}/spv/${out_name}.spv.h)
  add_custom_command(OUTPUT ${_inc}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py ${_spv} ${_inc} ${out_name}
    DEPENDS ${_spv})
  list(APPEND _gdn_spvs ${_inc})
endmacro()
foreach(_d 32 64 96 128 160 192 224 256)
  _gdn_variant(gdn_norm.comp gdn_norm_${_d} -DGDNDIM=${_d})
  _gdn_variant(gdn_out.comp gdn_out_${_d} -DGDNDIM=${_d})
endforeach()
_gdn_variant(gdn_gates.comp gdn_gates)
_gdn_variant(gdn_state.comp gdn_state_plain -DGDN_VARIANT=0 -DGDNS_THREADS=1024)
foreach(_p 1 2 4 8)
  _gdn_variant(gdn_state.comp gdn_state_reg_${_p} -DGDN_VARIANT=1 -DGDNS_REG_PER=${_p} -DGDNS_THREADS=1024)
endforeach()
add_library(strata_vk_kernel_gdn STATIC src/poly/kernels/gdn_vk.cpp ${_gdn_spvs})
target_include_directories(strata_vk_kernel_gdn PUBLIC ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(strata_vk_kernel_gdn PUBLIC strata_vk_runtime strata_vk_compat)

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
target_link_libraries(poly_kernels_vk_parity PRIVATE strata_vk_kernel_copy2 strata_vk_kernel_rms_norm strata_vk_kernel_native_quantize_q8_1 strata_vk_kernel_gemv strata_vk_kernel_gemv_bf16 strata_vk_kernel_conv4_silu strata_vk_kernel_elementwise strata_vk_kernel_head_norm_rope strata_vk_kernel_store_f16 strata_vk_kernel_store_q8 strata_vk_kernel_likeliest strata_vk_poly_kernels strata_vk_runtime strata_vk_compat strata_vk_stubs)   # stubs LAST: missing() for the poly trap lib; real launchers already resolved
# strata_vk_poly_kernels supplies the host-defined sum_threads() the rms_norm launcher asks for; its
# traps stay dormant here (the test calls only ported entry points).
add_test(NAME poly_kernels_vk_parity COMMAND poly_kernels_vk_parity)

# Task 2.5: gdn_step parity (host-reference transcription of the five-kernel chain, both arms).
add_executable(poly_gdn_vk_parity src/tests/poly_gdn_vk_parity.cpp)
target_include_directories(poly_gdn_vk_parity PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(poly_gdn_vk_parity PRIVATE strata_vk_kernel_gdn strata_vk_poly_kernels strata_vk_runtime strata_vk_compat strata_vk_stubs)
add_test(NAME poly_gdn_vk_parity COMMAND poly_gdn_vk_parity)

# Task 2.11: poly attend parity - the general fp16-KV chain (score reg<4,1> / softmax / value).
add_executable(poly_attend_vk_parity src/tests/poly_attend_vk_parity.cpp)
target_include_directories(poly_attend_vk_parity PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(poly_attend_vk_parity PRIVATE strata_vk_kernel_attend strata_vk_kernel_attend_batch strata_vk_kernel_store_q8 strata_vk_kernel_store_q8 strata_vk_blas strata_vk_poly_kernels strata_vk_runtime strata_vk_compat strata_vk_stubs)
add_test(NAME poly_attend_vk_parity COMMAND poly_attend_vk_parity)

# Phase 3 (attempt 2): the GPU tiled GEMM's micro-parity gate - GPU arm vs the ordered-fp32 CPU
# reference through the same public cublas surface (arm-switch via strata_vk_blas_use_gpu), all
# compiled trans/f16 variants + strided batch + alpha/beta {0,1}, rel 1e-5, plus the 2048^3 bench.
add_executable(vk_blas_gpu_parity src/tests/vk_blas_gpu_parity.cpp)
target_include_directories(vk_blas_gpu_parity PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/vulkan_compat)
target_link_libraries(vk_blas_gpu_parity PRIVATE strata_vk_blas strata_vk_runtime strata_vk_compat)
add_test(NAME vk_blas_gpu_parity COMMAND vk_blas_gpu_parity)

# Task 2.11: the arena's piece-align law - every region <= max allocation is single-piece by
# construction (arena_add bumps the cursor past a short tail); oversize regions still chain and
# keep the named full-span rejection. The bind check rides kernel_launch_sized's piece_of loop.
add_executable(vk_piece_align src/tests/vk_piece_align.cpp)
target_include_directories(vk_piece_align PRIVATE ${STRATA_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(vk_piece_align PRIVATE strata_vk_kernel_store_f16 strata_vk_runtime)
add_test(NAME vk_piece_align COMMAND vk_piece_align)

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

# Phase 4.2: the capture layer pinned alone (record/replay bits vs eager; destroy; flags)
add_executable(vk_graph_capture src/tests/vk_graph_capture.cpp)
target_include_directories(vk_graph_capture PRIVATE ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(vk_graph_capture PRIVATE strata_vk_kernel_store_f16 strata_vk_runtime strata_vk_compat)
add_test(NAME vk_graph_capture COMMAND vk_graph_capture)

# Task 4.2: events with real GPU timestamps, pinned by execution on BOTH arms (gpu-timestamp
# node vs the host-clock fallback) — elapsed() must be >0, plausible and never fake-negative.
add_executable(vk_event_timing src/tests/vk_event_timing.cpp)
target_include_directories(vk_event_timing PRIVATE ${STRATA_ROOT}/include ${STRATA_ROOT}/src/poly ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/spv)
target_link_libraries(vk_event_timing PRIVATE strata_vk_kernel_store_f16 strata_vk_runtime strata_vk_compat)
add_test(NAME vk_event_timing COMMAND vk_event_timing)
add_test(NAME vk_event_timing_host_arm COMMAND vk_event_timing)
set_tests_properties(vk_event_timing_host_arm PROPERTIES ENVIRONMENT "STRATA_VK_NO_GPU_EVENTS=1")

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

# ---- Task 1.4: anystrata-gguf-vk - the engine's strata-poly target (root CMakeLists.txt:2059) mirrored
# onto Vulkan with the host sources UNCHANGED.
#
# The .cu libraries of the CUDA target are NOT compiled here. src/poly/kernels.cu and kernels_dsv4.cu hold
# the host wrappers AND the __global__ kernels in one TU - with no nvcc, the sycl port's "migrated copy"
# rule lands here as the trap roster: include/strata/poly/kernels.hpp's whole contract is defined by name
# (src/poly/stubs_gen.cpp, generated by tools/gen_anystrata_stubs.py), plus host-only parts carried honestly:
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
set(STRATA_VK_GGML_DIR ${STRATA_ROOT}/third_party/llama.cpp/ggml CACHE PATH "ggml checkout for anystrata-gguf-vk")
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

# The poly engine's source set and link order (real launchers FIRST, stubs last: the link-order
# rule above), shared by the two engines AND the Task-4.3 port of poly_draft_race_test (same
# sources, same libs, run by hand with a model file — the CUDA build's convention).
set(VK_POLY_ENGINE_SRC
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
set(VK_POLY_ENGINE_LIBS
  strata_vk_kernel_s2_gemv strata_vk_kernel_dequant_s2   # real launchers FIRST (link-order rule above)
  strata_vk_kernel_copy2 strata_vk_kernel_rms_norm strata_vk_kernel_native_quantize_q8_1
  strata_vk_kernel_native_mmvq
  strata_vk_kernel_gemv strata_vk_kernel_gemv_bf16 strata_vk_kernel_conv4_silu strata_vk_kernel_gdn strata_vk_kernel_elementwise
  strata_vk_kernel_head_norm_rope strata_vk_kernel_store_f16 strata_vk_kernel_store_q8 strata_vk_kernel_store_f16_at strata_vk_kernel_store_q8_at strata_vk_kernel_attend strata_vk_kernel_attend_batch strata_vk_kernel_likeliest
  strata_vk_kernel_bf16_to_f32 strata_vk_kernel_gemm_f32
  strata_vk_poly_kernels strata_vk_core
  ggml-cpu ggml-base)

add_executable(anystrata-gguf-vk
  ${STRATA_ROOT}/src/poly/main.cpp
  ${VK_POLY_ENGINE_SRC})
target_include_directories(anystrata-gguf-vk PRIVATE ${STRATA_ROOT}/include ${STRATA_VK_GGML_DIR}/include)
target_link_libraries(anystrata-gguf-vk PRIVATE
  ${VK_POLY_ENGINE_LIBS}
  strata_vk_compat strata_vk_stubs                      # stubs last: they only let the link complete
  Threads::Threads)

# Task 4.3's other half: poly_draft_race_test ported (CUDA's own src/poly/draft_race_test.cpp,
# UNCHANGED, linked against the Vulkan engine set). "Does a read with the draft block repeat bit
# for bit?" — the graphs-default law this task asked for, run by hand with a model file like the
# CUDA build's: poly_draft_race_test VK-ENGINE-MODEL.gguf [RUNS]. Exit 0 every run matched the
# first; 3 SKIP when the read gave the draft block too few calls to show the race.
add_executable(poly_draft_race_test_vk
  ${STRATA_ROOT}/src/poly/draft_race_test.cpp
  ${VK_POLY_ENGINE_SRC})
target_include_directories(poly_draft_race_test_vk PRIVATE ${STRATA_ROOT}/include ${STRATA_VK_GGML_DIR}/include)
target_link_libraries(poly_draft_race_test_vk PRIVATE
  ${VK_POLY_ENGINE_LIBS}
  strata_vk_compat strata_vk_stubs
  Threads::Threads)
