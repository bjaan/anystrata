// cublas_v2.h for a Vulkan build: the cuBLAS names the engine's GEMM sites use, on a CPU fallback GEMM
// (vulkan/src/runtime/vk_blas.cpp).  hip_compat maps these onto hipBLAS by macro; there is no BLAS vendor
// runtime here, so the surface is real declarations resolved by vk_blas.cpp.
//
// SCOPE: the measured roster src/{poly,prefill,qwen35}/*.cu use - handle create/destroy, stream/workspace/
// math-mode knobs, Sgemm, SgemmStridedBatched, GemmEx, GemmStridedBatchedEx, Hgemm.  The kernel GEMM
// replaces this in Phase 3 (the plan's Task 3.1); correctness first, speed is a PORT-STATUS todo.
//
// OPERAND ADDRESSES: cuBLAS takes DEVICE pointers in this tree; the compat device-pointer contract (arena
// offset + bias cast to void*, see strata_vk.h) applies here too.  vk_blas.cpp resolves them through the
// same copies the runtime uses: arena -> host scratch, GEMM on the CPU, host scratch -> arena.  That is
// correct on any span the copy layer accepts (a straddling span surfaces as CUBLAS_STATUS_EXECUTION_FAILED
// with the runtime's message).
//
// GEMM SEMANTICS spelled out (cuBLAS's column-major contract, exactly what the .cu call sites assume):
//   C = alpha * op(A) * op(B) + beta * C, op sizes m x k, k x n, ld strides; fp16-in / fp32-accumulate /
//   out-type per the Atype/Btype/Ctype arguments of GemmEx.
#pragma once

#include "cuda_runtime.h"
#include "cuda_fp16.h"     // cublasHgemm's __half operands

#include <cstddef>

enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
};
typedef void* cublasHandle_t;
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };
// cudaDataType values as CUDA spells them (the subset the tree passes).
enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
    CUDA_R_16BF = 14,
};
typedef cudaDataType_t cudaDataType;
enum cublasComputeType_t {
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
};
enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = -2,   // ignored on this backend (no tensor-core path); ordered fp32
};
constexpr unsigned int CUBLAS_DEFAULT_MATH = 0, CUBLAS_TF32_TENSOR_OP_MATH = 1;

extern "C" {
cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* stream);
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace, size_t bytes);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, unsigned mode);
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, unsigned* mode);

cublasStatus_t cublasSgemm_v2(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                              int m, int n, int k, const float* alpha, const float* A, int lda,
                              const float* B, int ldb, const float* beta, float* C, int ldc);
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                         int m, int n, int k, const float* alpha, const float* A, int lda,
                                         long long strideA, const float* B, int ldb, long long strideB,
                                         const float* beta, float* C, int ldc, long long strideC, int batchCount);
cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                            int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda,
                            const void* B, cudaDataType Btype, int ldb, const void* beta, void* C,
                            cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                          int m, int n, int k, const void* alpha, const void* A,
                                          cudaDataType Atype, int lda, long long strideA, const void* B,
                                          cudaDataType Btype, int ldb, long long strideB, const void* beta,
                                          void* C, cudaDataType Ctype, int ldc, long long strideC,
                                          int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
cublasStatus_t cublasHgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                           int m, int n, int k, const __half* alpha, const __half* A, int lda,
                           const __half* B, int ldb, const __half* beta, __half* C, int ldc);
}

// The v2 names are what CUDA's headers alias; the engine calls both spellings.
#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasSetStream cublasSetStream_v2
#define cublasGetStream cublasGetStream_v2
#define cublasSetWorkspace cublasSetWorkspace_v2
#define cublasSgemm cublasSgemm_v2
