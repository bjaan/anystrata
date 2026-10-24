// vulkan/src/poly/kernels/gemm_f32_vk.cpp - the Vulkan launcher for strata::poly::gemm_f32
// (include/strata/poly/kernels.hpp; the CUDA body is src/poly/kernels.cu:2190-2211). The CUDA
// wrapper is pure host cuBLAS: y = w x per token group, expressed as Y = W^T X through
// cublasSgemm / cublasSgemmStridedBatched. Those two entry points are REAL in the compat
// surface (vk_blas.cpp: gemm_one ordered-fp32 CPU fallback; the handle's math mode is pinned
// CUBLAS_DEFAULT_MATH, "always ordered fp32"), so this launcher transcribes the wrapper whole.
//
// One honest deviation, recorded here and in PORT-STATUS: the small-out arm (:2194-2200,
// groups==1 && n_out <= 8*SMALL_OUT = 256) is a CUDA perf kernel (gemm_small_kernel, a warp
// per token with SMALL_K-chunked accumulation). This backend routes that shape through the
// same cublasSgemm call: the fp32 accumulation ORDER of the CUDA perf kernel is not a parity
// contract anywhere in this port (the CUDA build's own cuBLAS path is heuristic-order for the
// big shapes), so the compat ordered GEMM is the reference the parity suite pins. The engine
// reaches gemm_f32 only through lin()'s prompt-batch arm (n >= kStageRows) for F32/BF16 dense
// matrices - decode stays on gemv (bitwise-ported) and mmvq.
//
// Demanded by the GSQ-hybrid long-context gate (485 tokens, max-context 16384): engine.cpp
// lin() -> bf16_to_f32 (port 2) then this.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_compat/cublas_v2.h"
#include "strata/vulkan_compat/strata_vk.h"

#include <stdexcept>

namespace strata::poly {

void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out,
              int rows_per_x, int rows) {
    const cublasHandle_t h = (cublasHandle_t) blas;
    if (!h) throw std::invalid_argument("gemm_f32: null cuBLAS handle");
    if (n_in <= 0 || n_out <= 0 || rows <= 0) throw std::invalid_argument("gemm_f32: non-positive shape");
    if (rows_per_x <= 0 || rows_per_x > n_out || n_out % rows_per_x)
        throw std::invalid_argument("gemm_f32: rows_per_x must divide n_out");
    const float one = 1.0f, zero = 0.0f;
    const int groups = n_out / rows_per_x;
    if (groups == 1) {
        const cublasStatus_t s = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_out, rows, n_in,
                                             &one, w, n_in, x, n_in, &zero, y, n_out);
        if (s != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("gemm_f32: a batch's projection failed in the compat cuBLAS");
        return;
    }
    const cublasStatus_t s = cublasSgemmStridedBatched(
        h, CUBLAS_OP_T, CUBLAS_OP_N, rows_per_x, rows, n_in, &one, w, n_in,
        (long long) rows_per_x * n_in, x, groups * n_in, n_in, &zero, y, n_out,
        rows_per_x, groups);
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("gemm_f32: a batch's projections failed in the compat cuBLAS");
}

}  // namespace strata::poly
