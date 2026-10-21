// vulkan/src/tests/vk_compat_smoke.cpp - the Task 1.1 compile proof: ../src's real host files
// (src/core/layout.cpp, src/core/weights.cpp) compiled UNCHANGED against the vulkan_compat headers, plus a
// few live calls through the compat surface so the link is exercised, not just the parse.
//
// What it asserts (cheap, local, no GGUF needed):
//   * the Vulkan runtime initializes (the device line prints);
//   * cudaMalloc/cudaMemcpy/cudaMemset/cudaFree round-trip through the arena and staged copies;
//   * a cublas CPU-fallback GEMM (cublasSgemm) matches a hand reference on tiny matrices;
//   * the graph traps are compiled in (this file never calls them; --no-graphs keeps the engine off them).
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    // ---- the runtime and the properties the host files read
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "vk_compat_smoke: cudaGetDeviceProperties: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    std::printf("vk_compat_smoke: device \"%s\" (arch %s), %.1f GiB\n", prop.name, prop.gcnArchName,
                (double) prop.totalGlobalMem / (1024.0 * 1024 * 1024));

    // ---- arena round-trip: alloc, H2D, memset, D2H, compare
    constexpr size_t N = 4096;
    void* dev = nullptr;
    if (cudaMalloc(&dev, N * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "vk_compat_smoke: cudaMalloc: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    if (!dev) { std::fprintf(stderr, "vk_compat_smoke: the arena handed out null\n"); return 1; }
    std::vector<float> host(N), back(N);
    for (size_t i = 0; i < N; ++i) host[i] = (float) i * 0.5f - 100.f;
    if (cudaMemcpy(dev, host.data(), N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemsetAsync((char*) dev + N * sizeof(float) - 16, 0, 16, nullptr) != cudaSuccess ||
        cudaMemcpy(back.data(), dev, N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "vk_compat_smoke: copy round-trip: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    for (size_t i = 0; i + 4 < N; ++i)
        if (back[i] != host[i]) { std::fprintf(stderr, "vk_compat_smoke: arena round-trip differs at %zu\n", i); return 1; }
    cudaFree(dev);   // bump arena: a no-op by contract

    // ---- cublas CPU fallback: C = A^T * B (m=3, n=2, k=4), column-major like the engine's sites
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "vk_compat_smoke: cublasCreate failed\n"); return 1; }
    cublasSetStream(h, nullptr);
    cublasSetWorkspace(h, nullptr, 0);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    // cuBLAS OP_T: A is stored k x m column-major (lda >= k).  A here is 4x3 with lda=4; B is 4x2 OP_N.
    float A[4 * 3], B[4 * 2], C[3 * 2], Cref[3 * 2];
    for (int i = 0; i < 12; ++i) A[i] = (float) (i % 5) - 2.f;
    for (int i = 0; i < 8; ++i) B[i] = (float) (i % 3) + 1.f;
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 3; ++i) {
        float acc = 0.f;
        for (int kk = 0; kk < 4; ++kk) acc += A[kk + i * 4] * B[kk + j * 4];   // A^T(i,kk)=A(kk,i) * B(kk,j)
        Cref[j * 3 + i] = acc;
    }
    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, sizeof A); cudaMalloc(&dB, sizeof B); cudaMalloc(&dC, sizeof C);
    cudaMemcpy(dA, A, sizeof A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof B, cudaMemcpyHostToDevice);
    const float one = 1.f, zero = 0.f;
    if (cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, 3, 2, 4, &one, (const float*) dA, 4,
                    (const float*) dB, 4, &zero, (float*) dC, 3) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "vk_compat_smoke: cublasSgemm failed\n"); return 1;
    }
    cudaMemcpy(C, dC, sizeof C, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 6; ++i)
        if (C[i] != Cref[i]) { std::fprintf(stderr, "vk_compat_smoke: sgemm mismatch at %d (%g vs %g)\n", i, C[i], Cref[i]); return 1; }

    // ---- cublasGemmEx with fp16 operands / fp32 accumulate / fp32 out (the prefill gemm.cu shape)
    __half hA[12], hB[8];
    float C32[6];
    for (int i = 0; i < 12; ++i) hA[i] = __float2half(A[i]);
    for (int i = 0; i < 8; ++i) hB[i] = __float2half(B[i]);
    void *dhA = nullptr, *dhB = nullptr, *dC32 = nullptr;
    cudaMalloc(&dhA, sizeof hA); cudaMalloc(&dhB, sizeof hB); cudaMalloc(&dC32, sizeof C32);
    cudaMemcpy(dhA, hA, sizeof hA, cudaMemcpyHostToDevice);
    cudaMemcpy(dhB, hB, sizeof hB, cudaMemcpyHostToDevice);
    if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 3, 2, 4, &one, dhA, CUDA_R_16F, 4,
                     dhB, CUDA_R_16F, 4, &zero, dC32, CUDA_R_32F, 3,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "vk_compat_smoke: cublasGemmEx failed\n"); return 1;
    }
    cudaMemcpy(C32, dC32, sizeof C32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 6; ++i)
        if (C32[i] != Cref[i]) { std::fprintf(stderr, "vk_compat_smoke: GemmEx f16 mismatch at %d (%g vs %g)\n", i, C32[i], Cref[i]); return 1; }
    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dhA); cudaFree(dhB); cudaFree(dC32);

    // ---- the two engine host files: their symbols linked (a minimal call each, no pack needed)
    strata::core::ModelGeometry g{};
    std::string err;
    if (strata::core::check_all(strata::core::WeightTable{}, g, err)) {
        std::fprintf(stderr, "vk_compat_smoke: check_all on an empty table should fail\n"); return 1;
    }
    std::printf("vk_compat_smoke: layout.cpp+weights.cpp link and run; layout said: %s\n", err.c_str());
    std::printf("vk_compat_smoke: OK\n");
    return 0;
}
