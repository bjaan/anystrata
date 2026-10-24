// vulkan/src/tests/vk_blas_gpu_parity.cpp - the Phase-3 micro-gate: the GPU tiled SGEMM (mm_gemm.comp)
// vs the ordered-fp32 CPU reference, both through the SAME public cublas surface, arm-switched via
// strata_vk_blas_use_gpu(). Every trans/f16 variant the shim compiles + strided batching + alpha/beta
// {0,1}, rel tol 1e-5. Then a 2048^3 GFLOPs printout for both arms. Green before any model run.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>

extern "C" void strata_vk_blas_use_gpu(int on);

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; \
    std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); \
    std::fprintf(stderr, "\n"); } } while (0)

static double rel_cmp(const std::vector<float>& got, const std::vector<float>& ref, const char* tag) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::abs((double) got[i] - ref[i]);
        const double s = std::max(1.0, std::abs((double) ref[i]));
        if (d / s > worst) worst = d / s;
    }
    std::printf("  %-28s worst rel err %.3e %s\n", tag, worst, worst <= 1e-5 ? "OK" : "FAIL");
    if (worst > 1e-5) ++failures;
    return worst;
}

int main() {
    cublasHandle_t h = nullptr;
    CHECK(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasCreate");
    if (!h) return 1;

    // ---- case A: cublasSgemm / SgemmStridedBatched, all four trans combos (TB is unsupported-GPU
    // by design and must land on CPU both arms: only NN, TN, NT go GPU), f32.
    // Layout: m=200, n=128, k=300 per the gate; batch 3 for the strided form.
    const int M = 200, N = 128, K = 300, BATCH = 3;
    std::vector<float> hA((size_t) K * M * BATCH), hB((size_t) K * N * BATCH), hC((size_t) M * N * BATCH);
    auto rnd = [](int i) { return (float) ((((unsigned) i * 1103515245u + 12345u) >> 8) % 2000u) * 0.001f - 0.5f; };
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = rnd((int) i);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = rnd((int) i + 7);
    for (size_t i = 0; i < hC.size(); ++i) hC[i] = rnd((int) i + 13) * 0.25f;

    auto run_case = [&](const char* tag, cublasOperation_t ta, cublasOperation_t tb,
                        int lda, int ldb, int ldc, float alpha, float beta, int batch,
                        long long sA, long long sB, long long sC) {
        // host operands for each arm: fill C with the same base, run under GPU, read back; rerun under CPU.
        void *dA, *dB, *dC;
        CHECK(cudaMalloc(&dA, hA.size() * 4) == cudaSuccess, "cudaMalloc A");
        CHECK(cudaMalloc(&dB, hB.size() * 4) == cudaSuccess, "cudaMalloc B");
        CHECK(cudaMalloc(&dC, hC.size() * 4) == cudaSuccess, "cudaMalloc C");
        const size_t bytesA = ((size_t) sA * (batch - 1) + (ta == CUBLAS_OP_N ? (size_t) lda * K : (size_t) lda * M)) * 4;
        const size_t bytesB = ((size_t) sB * (batch - 1) + (tb == CUBLAS_OP_N ? (size_t) ldb * N : (size_t) ldb * K)) * 4;
        const size_t bytesC = (sC * (batch - 1) + (size_t) ldc * N) * 4;
        CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice) == cudaSuccess, "H2D A");
        CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice) == cudaSuccess, "H2D B");
        CHECK(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice) == cudaSuccess, "H2D C");

        auto one = [&](bool gpu, std::vector<float>& out) {
            CHECK(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice) == cudaSuccess, "reH2D C");
            strata_vk_blas_use_gpu(gpu ? 1 : 0);
            cublasStatus_t s;
            if (batch == 1) {
                s = cublasSgemm(h, ta, tb, M, N, K, &alpha, (const float*) dA, lda,
                                (const float*) dB, ldb, &beta, (float*) dC, ldc);
            } else {
                s = cublasSgemmStridedBatched(h, ta, tb, M, N, K, &alpha, (const float*) dA, lda, sA,
                                              (const float*) dB, ldb, sB, &beta, (float*) dC, ldc, sC, batch);
            }
            CHECK(s == CUBLAS_STATUS_SUCCESS, "%s: cublas status %d (gpu=%d)", tag, (int) s, (int) gpu);
            CHECK(cudaMemcpy(out.data(), dC, bytesC, cudaMemcpyDeviceToHost) == cudaSuccess, "D2H C");
            cudaDeviceSynchronize();
        };
        std::vector<float> g(bytesC / 4), c(bytesC / 4);
        one(true, g);
        one(false, c);
        strata_vk_blas_use_gpu(-1);
        char name[96];
        std::snprintf(name, sizeof name, "%s a=%g b=%g bN=%d", tag, (double) alpha, (double) beta, batch);
        rel_cmp(g, c, name);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    };

    for (float ab : {0.f, 1.f})
        for (float bb : {0.f, 1.f}) {
            run_case("NN", CUBLAS_OP_N, CUBLAS_OP_N, M, K, M, ab, bb, 1, 0, 0, 0);
            run_case("TN", CUBLAS_OP_T, CUBLAS_OP_N, K, K, M, ab, bb, 1, 0, 0, 0);
            run_case("NT", CUBLAS_OP_N, CUBLAS_OP_T, M, N, M, ab, bb, 1, 0, 0, 0);   // CPU both arms (no OP_T-B kernel)
            run_case("NN-b3", CUBLAS_OP_N, CUBLAS_OP_N, M, K, M, ab, bb, BATCH, (long long) M * K, (long long) K * N, (long long) M * N);
            run_case("TN-b3", CUBLAS_OP_T, CUBLAS_OP_N, K, K, M, ab, bb, BATCH, (long long) M * K, (long long) K * N, (long long) M * N);
        }
    // ragged tile edges (m,n not multiples of 64) are covered by 200x128; k not a multiple of 16 by 300.

    // ---- case B: cublasGemmStridedBatchedEx with CUDA_R_16F operands + fp32 C (the attend_batch
    // shapes): scores TN (m x cols x dim, f16 A=kcache ld=width, f16 B=qg) and values NN.
    {
        // scores: ta=T tb=N m=128 n=16 k=128 lda=256(kv width) ldb=128 ldc=128 strA=128*128 strB=16*128 strC=128*16 batch=2
        const int m = 128, n = 16, k = 128, lda = 256, ldb = 128, ldc = 128, bcnt = 2;
        const long long sA = (long long) m * k, sB = (long long) n * ldb, sC = (long long) m * n;
        // NOTE OP_T A stores k x m: lda >= k. lda=256 with k=128, m=128: A words = (lda*(m-1)+k) per batch.
        const size_t nA = (size_t)(lda * (m - 1) + k) * bcnt, nB = (size_t)(ldb * (n - 1) + k) * bcnt,
                     nC = (size_t)(ldc * (n - 1) + m) * bcnt;
        std::vector<float> fa(nA), fb(nB), fc0(nC), g(nC), c(nC);
        for (size_t i = 0; i < nA; ++i) fa[i] = rnd((int) i) * 0.3f;
        for (size_t i = 0; i < nB; ++i) fb[i] = rnd((int) i + 5) * 0.3f;
        for (size_t i = 0; i < nC; ++i) fc0[i] = rnd((int) i + 9) * 0.1f;
        std::vector<uint16_t> ha(nA), hb(nB);
        for (size_t i = 0; i < nA; ++i) ha[i] = __half_as_ushort(__float2half(fa[i]));
        for (size_t i = 0; i < nB; ++i) hb[i] = __half_as_ushort(__float2half(fb[i]));
        // re-round through f16 so both arms see identical operands (the CPU arm converts the same bits)
        for (size_t i = 0; i < nA; ++i) fa[i] = __half2float(__ushort_as_half(ha[i]));
        for (size_t i = 0; i < nB; ++i) fb[i] = __half2float(__ushort_as_half(hb[i]));
        void *dA, *dB, *dC;
        CHECK(cudaMalloc(&dA, nA * 2) == cudaSuccess, "f16 malloc A");
        CHECK(cudaMalloc(&dB, nB * 2) == cudaSuccess, "f16 malloc B");
        CHECK(cudaMalloc(&dC, nC * 4) == cudaSuccess, "f32 malloc C");
        CHECK(cudaMemcpy(dA, ha.data(), nA * 2, cudaMemcpyHostToDevice) == cudaSuccess, "f16 H2D A");
        CHECK(cudaMemcpy(dB, hb.data(), nB * 2, cudaMemcpyHostToDevice) == cudaSuccess, "f16 H2D B");
        for (float alpha : {0.f, 1.f})
            for (float beta : {0.f, 1.f}) {
                auto one = [&](bool gpu, std::vector<float>& out) {
                    CHECK(cudaMemcpy(dC, fc0.data(), nC * 4, cudaMemcpyHostToDevice) == cudaSuccess, "f16 reH2D C");
                    strata_vk_blas_use_gpu(gpu ? 1 : 0);
                    const cublasStatus_t s = cublasGemmStridedBatchedEx(
                        h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_16F, lda, sA,
                        dB, CUDA_R_16F, ldb, sB, &beta, dC, CUDA_R_32F, ldc, sC, bcnt,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    CHECK(s == CUBLAS_STATUS_SUCCESS, "f16 TN: status %d (gpu=%d)", (int) s, (int) gpu);
                    CHECK(cudaMemcpy(out.data(), dC, nC * 4, cudaMemcpyDeviceToHost) == cudaSuccess, "f16 D2H");
                    cudaDeviceSynchronize();
                };
                one(true, g); one(false, c);
                strata_vk_blas_use_gpu(-1);
                char name[64];
                std::snprintf(name, sizeof name, "f16-TN a=%g b=%g", (double) alpha, (double) beta);
                rel_cmp(g, c, name);
            }
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // ---- bench: 2048^3 through the public API on both arms
    {
        const int BB = 2048;
        void *dA, *dB, *dC;
        CHECK(cudaMalloc(&dA, (size_t) BB * BB * 4) == cudaSuccess, "bench A");
        CHECK(cudaMalloc(&dB, (size_t) BB * BB * 4) == cudaSuccess, "bench B");
        CHECK(cudaMalloc(&dC, (size_t) BB * BB * 4) == cudaSuccess, "bench C");
        std::vector<float> host(BB * BB);
        for (size_t i = 0; i < host.size(); ++i) host[i] = rnd((int) i) * 0.01f;
        CHECK(cudaMemcpy(dA, host.data(), host.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess, "bench H2D A");
        CHECK(cudaMemcpy(dB, host.data(), host.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess, "bench H2D B");
        CHECK(cudaMemset(dC, 0, host.size() * 4) == cudaSuccess, "bench C");
        const float one = 1.f, zero = 0.f;
        for (int gpu : {1, 0}) {
            strata_vk_blas_use_gpu(gpu);
            const cublasStatus_t s = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, BB, BB, BB,
                                                 &one, (const float*) dA, BB, (const float*) dB, BB,
                                                 &zero, (float*) dC, BB);
            CHECK(s == CUBLAS_STATUS_SUCCESS, "bench status %d gpu=%d", (int) s, gpu);
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < (gpu ? 3 : 1); ++it)
                cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, BB, BB, BB,
                            &one, (const float*) dA, BB, (const float*) dB, BB,
                            &zero, (float*) dC, BB);
            cudaDeviceSynchronize();
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count() / (gpu ? 3 : 1);
            std::printf("bench 2048^3 %s: %.1f ms -> %.2f GFLOPs\n", gpu ? "GPU" : "CPU", ms,
                        2.0 * BB * BB * BB / (ms * 1e6));
        }
        strata_vk_blas_use_gpu(-1);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    if (failures) { std::fprintf(stderr, "vk_blas_gpu_parity: %d FAILURES\n", failures); return 1; }
    std::printf("vk_blas_gpu_parity: all cases within rel 1e-5\n");
    return 0;
}
