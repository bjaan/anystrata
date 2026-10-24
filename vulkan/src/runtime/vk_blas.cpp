// vulkan/src/runtime/vk_blas.cpp - the cublas_v2.h surface on a plain blocked CPU GEMM.
//
// CORRECTNESS FIRST (the plan, Task 1.1 Step 3): Phase 3's dequant+GEMM kernels replace this; speed is the
// PORT-STATUS `host(free)` todo.  What is implemented here: the exact column-major contract cuBLAS states
// (C = alpha*op(A)*op(B) + beta*C), fp32 accumulation for every operand mix the tree passes
// (CUDA_R_32F / CUDA_R_16F / CUDA_R_16BF in, fp32 or the operand type out), and strided batching.
//
// Operand pointers are arena device pointers under the compat contract (offset + kPtrBias, see strata_vk.h):
// each matrix is fetched through the runtime's staged copies into a host block, multiplied, and written back
// the same way.  A span the copy layer refuses (crossing an arena piece) surfaces as
// CUBLAS_STATUS_EXECUTION_FAILED carrying the runtime's named message - never a silent zero.
//
// The 16-bit conversions are the tree's own tested transcriptions (cuda_fp16.h carries f16_bits.hpp's
// converter verbatim; bf16 mirrors bf16_bits.hpp), so a host reference GEMM and this one round identically.
#include "strata/vulkan_compat/cublas_v2.h"
#include "strata/vulkan_compat/cuda_bf16.h"
#include "strata/vulkan_runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

// ---- Phase 3 (attempt 2): the GPU tiled SGEMM (mm_gemm.comp, four compiled variants:
// TRANS_A x F16AB). The common call-site combos (attend_batch's two strided-batched f16/f32-C
// products with OP_N B, gemm_f32's OP_T/OP_N f32 products) dispatch GPU-side with the exact
// strided-batch semantics; anything exotic falls back to the ordered-fp32 CPU reference below
// with a counted one-line notice (never silent). GPU is the default; STRATA_VK_BLAS_CPU=1
// forces the CPU arm (A/B), and the parity test re-toggles through strata_vk_blas_use_gpu().
#include "mm_gemm_t0f0.spv.h"
#include "mm_gemm_t0f1.spv.h"
#include "mm_gemm_t1f0.spv.h"
#include "mm_gemm_t1f1.spv.h"

int g_gpu_override = -1;   // -1: use env; 0/1: the parity test's explicit arm
constexpr std::uintptr_t kGemmPtrBias = 64;   // same value as kPtrBias below (strata_vk.h's contract)
bool gpu_wanted() {
    if (g_gpu_override >= 0) return g_gpu_override == 1;
    static const bool cpu = std::getenv("STRATA_VK_BLAS_CPU") != nullptr;
    return !cpu;
}
long long& fallback_notes() { static long long n = 0; return n; }
void note_fallback(const char* why) {
    long long& n = fallback_notes();
    if (++n <= 5) std::fprintf(stderr, "strata-vk cublas: GPU GEMM not used (%s); CPU fallback (note %lld)\n", why, n);
}

const strata::vk::Kernel& mm_kernel(int transA, int f16) {
    static const strata::vk::Kernel k00 = strata::vk::kernel_load("mm_gemm_t0f0", mm_gemm_t0f0_spv, mm_gemm_t0f0_spv_len, 3, 64, 256, false);
    static const strata::vk::Kernel k01 = strata::vk::kernel_load("mm_gemm_t0f1", mm_gemm_t0f1_spv, mm_gemm_t0f1_spv_len, 3, 64, 256, false);
    static const strata::vk::Kernel k10 = strata::vk::kernel_load("mm_gemm_t1f0", mm_gemm_t1f0_spv, mm_gemm_t1f0_spv_len, 3, 64, 256, false);
    static const strata::vk::Kernel k11 = strata::vk::kernel_load("mm_gemm_t1f1", mm_gemm_t1f1_spv, mm_gemm_t1f1_spv_len, 3, 64, 256, false);
    if (!transA && !f16) return k00;
    if (!transA && f16)  return k01;
    if (transA && !f16)  return k10;
    return k11;
}

struct MMParams {   // must match mm_gemm.comp's push_constant block (56B of fields), padded to 64B
    int32_t m, n, k, lda, ldb, ldc, tilesX, tilesY, batch;
    float alpha, beta;
    int32_t strideA, strideB, strideC;
    int32_t pad[2];
};
static_assert(sizeof(MMParams) == 64, "push-constant struct must be exactly 64B");

// GPU-wall counters (STRATA_VK_GEMM_WALL=1 measurement arm; see dump()).
long long g_wall_calls = 0, g_wall_f16_calls = 0, g_wall_f32_calls = 0;
double g_wall_ms = 0.0, g_wall_f16_ms = 0.0, g_wall_f32_ms = 0.0;

// The strided-batched GPU arm. Returns true = launched (queue-ordered against the runtime's
// other work); false = the caller runs the CPU reference. Element extents stay under 2^31 and
// every binding offset 4-byte aligned (the kernel addresses f16 through uint words).
bool gemm_gpu(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
              float alpha, const void* A, cudaDataType At, int lda, long long strideA,
              const void* B, cudaDataType Bt, int ldb, long long strideB,
              float beta, void* C, cudaDataType Ct, int ldc, long long strideC, int batch) {
    if (!gpu_wanted()) return false;
    if (m <= 0 || n <= 0 || k <= 0 || batch <= 0 || !A || !B || !C) return false;
    const bool f16 = (At == CUDA_R_16F && Bt == CUDA_R_16F);
    if (!f16 && !(At == CUDA_R_32F && Bt == CUDA_R_32F)) { note_fallback("operand type mix"); return false; }
    if (Ct != CUDA_R_32F) { note_fallback("non-f32 C"); return false; }
    if (tb != CUBLAS_OP_N) { note_fallback("OP_T B"); return false; }
    const int transA = ta == CUBLAS_OP_T;
    if (!transA && ta != CUBLAS_OP_N) return false;
    if (lda < (transA ? k : m) || ldb < k || ldc < m) return false;   // cuBLAS contract; bad input -> CPU
    const long long ea = f16 ? 2 : 4;
    if (strideA < 0 || strideB < 0 || strideC < 0 ||
        strideA > 0x7FFFFFFFll || strideB > 0x7FFFFFFFll || strideC > 0x7FFFFFFFll) {
        note_fallback("negative/oversize stride"); return false;
    }
    const long long extA = transA ? (long long)(m - 1) * lda + k : (long long)(k - 1) * lda + m;
    const long long extB = (long long)(n - 1) * ldb + k;
    const long long extC = (long long)(n - 1) * ldc + m;
    const long long topA = extA + (long long)(batch - 1) * strideA;
    const long long topB = extB + (long long)(batch - 1) * strideB;
    const long long topC = extC + (long long)(batch - 1) * strideC;
    if (topA >= (1ll << 31) || topB >= (1ll << 31) || topC >= (1ll << 31)) { note_fallback("span beyond 2^31 elems"); return false; }
    const uint64_t ao = (uint64_t)(std::uintptr_t) A - kGemmPtrBias,
                   bo = (uint64_t)(std::uintptr_t) B - kGemmPtrBias,
                   co = (uint64_t)(std::uintptr_t) C - kGemmPtrBias;   // same strip as fetch() (kPtrBias)
    if ((ao | bo | co) & 3u) { note_fallback("misaligned operand"); return false; }
    MMParams pc;
    std::memset(&pc, 0, sizeof pc);
    pc.m = m; pc.n = n; pc.k = k; pc.lda = lda; pc.ldb = ldb; pc.ldc = ldc;
    pc.tilesX = (n + 63) / 64; pc.tilesY = (m + 63) / 64; pc.batch = batch;
    pc.alpha = alpha; pc.beta = beta;
    pc.strideA = (int32_t) strideA; pc.strideB = (int32_t) strideB; pc.strideC = (int32_t) strideC;
    const uint64_t bufs[3] = {ao, bo, co};
    const uint64_t sizes[3] = {(uint64_t) topA * (uint64_t) ea, (uint64_t) topB * (uint64_t) ea,
                               (uint64_t) topC * 4ull};
    // STRATA_VK_GEMM_WALL=1: drain before/after the launch so the enqueue-only BlasClock gets a
    // companion measuring real GPU time per GEMM (measurement arm; serialization is queue-order
    // the per-dispatch barrier already imposes, so results stay bitwise-identical).
    static const bool gemm_wall = std::getenv("STRATA_VK_GEMM_WALL") != nullptr;
    std::chrono::steady_clock::time_point wall_t0;
    if (gemm_wall) { strata::vk::device_sync(); wall_t0 = std::chrono::steady_clock::now(); }
    try {
        // buffers are {A, B, C}: C (bit 2) is written; A/B are read (beta!=0 also reads C,
        // which the runtime counts as a span-touch either way).
        strata::vk::kernel_launch_sized_rw(mm_kernel(transA, f16 ? 1 : 0), pc.tilesX * pc.tilesY * pc.batch,
                                bufs, sizes, 4u, 3, &pc);
    } catch (const std::exception& e) {   // e.g. a span crossing an arena piece: the reference still runs
        if (gemm_wall) strata::vk::device_sync();
        note_fallback(e.what());
        return false;
    }
    if (gemm_wall) {
        strata::vk::device_sync();
        const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_t0).count();
        g_wall_ms += dt;
        g_wall_calls++;
        // f16 GEMMs happen only in attend_batch (scores/values); f32 in gemm_f32 dense proj.
        if (f16) { g_wall_f16_ms += dt; g_wall_f16_calls++; }
        else     { g_wall_f32_ms += dt; g_wall_f32_calls++; }
        if (std::getenv("STRATA_VK_GEMM_SHAPES"))
            std::fprintf(stderr, "gemm_wall: %s m=%d n=%d k=%d b=%d tiles=%dx%d groups=%d %.2fms\n",
                         f16 ? "f16" : "f32", m, n, k, batch, pc.tilesX, pc.tilesY,
                         pc.tilesX * pc.tilesY * batch, dt);
    }
    return true;
}

// ---- Phase 3 step 0: env-gated measurement (STRATA_VK_BLAS_TIME=1). One record per entry
// point: calls, total ms, worst single call, and the biggest (m,n,k,batch) seen.
struct BlasStat {
    long long calls = 0;
    double total_ms = 0.0, max_ms = 0.0;
    long long bm = 0, bn = 0, bk = 0, bb = 0;   // biggest-flops shape seen
    double flops(const BlasStat& s) { return double(s.bm) * s.bn * s.bk * s.bb; }
};
std::map<const char*, BlasStat>& blas_stats() { static std::map<const char*, BlasStat> m; return m; }
bool blas_time_on() { static bool v = std::getenv("STRATA_VK_BLAS_TIME") != nullptr; return v; }
void dump();
int& blas_clock_depth() { static thread_local int d = 0; return d; }
struct BlasClock {
    const char* kind; BlasStat* st = nullptr; double flops;
    std::chrono::steady_clock::time_point t0;
    BlasClock(const char* k, int m, int n, int kk, int b) : kind(k) {
        if (blas_clock_depth()++) { st = nullptr; return; }   // don't double-count nested entries
        st = &blas_stats()[k];
        flops = 2.0 * m * n * kk * (b > 0 ? b : 1);
        t0 = std::chrono::steady_clock::now();
    }
    ~BlasClock() {
        if (!st) { blas_clock_depth()--; return; }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        st->calls++; st->total_ms += ms;
        if (ms > st->max_ms) st->max_ms = ms;
        blas_clock_depth()--;
        if (blas_time_on() && st->calls % 250 == 0) dump();   // survive a timeout kill
    }
};
void blas_stat_shape(const char* kind, int m, int n, int k, int b) {
    BlasStat& s = blas_stats()[kind];
    if (2.0 * m * n * k * (b > 0 ? b : 1) > 2.0 * s.bm * s.bn * s.bk * s.bb) {
        s.bm = m; s.bn = n; s.bk = k; s.bb = b;
    }
}
void dump() {
    std::fprintf(stderr, "=== STRATA_VK_BLAS_TIME summary ===\n");
    double tot = 0;
    for (auto& kv : blas_stats()) {
        std::fprintf(stderr, "%-24s calls=%lld total=%.1fms max=%.1fms biggest m,n,k,b=%lld,%lld,%lld,%lld\n",
                     kv.first, kv.second.calls, kv.second.total_ms, kv.second.max_ms,
                     kv.second.bm, kv.second.bn, kv.second.bk, kv.second.bb);
        tot += kv.second.total_ms;
    }
    std::fprintf(stderr, "%-24s total=%.1fms\n", "ALL cublas (enqueue)", tot);
    if (g_wall_calls)
        std::fprintf(stderr, "%-24s calls=%lld gpu-drain=%.1fms (f16 %lld/%.1fms, f32 %lld/%.1fms)\n",
                     "GPU GEMM wall", g_wall_calls, g_wall_ms,
                     g_wall_f16_calls, g_wall_f16_ms, g_wall_f32_calls, g_wall_f32_ms);
    std::fflush(stderr);
}
struct BlasStatDump {
    ~BlasStatDump() { if (blas_time_on()) dump(); }
};
BlasStatDump g_blas_dump;

struct Handle {
    cudaStream_t stream = nullptr;
    void* workspace = nullptr;
    size_t ws_bytes = 0;
};

constexpr std::uintptr_t kPtrBias = 64;   // MUST match strata_vk.h (offset+bias device pointers)
constexpr size_t kTile = 256;             // host-block tile (K-panel); correctness path, not a tuned GEMM

size_t elem_bytes(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_32F:  return 4;
        case CUDA_R_16F: case CUDA_R_16BF: return 2;
        default: return 0;
    }
}

// device span -> host buffer (staged copies; throws with a named reason if the span crosses a piece)
void fetch(const void* p, float* dst, size_t count, cudaDataType_t t, cudaStream_t s) {
    const size_t bytes = count * elem_bytes(t);
    std::vector<uint8_t> raw(bytes);
    const std::uintptr_t arena = (std::uintptr_t) p - kPtrBias;   // compat handle -> raw arena offset
    if (t == CUDA_R_32F) {
        strata::vk::copy_d2h(dst, (void*) arena, bytes, s);
        return;
    }
    strata::vk::copy_d2h(raw.data(), (void*) arena, bytes, s);
    for (size_t i = 0; i < count; ++i) {
        if (t == CUDA_R_16F) {
            uint16_t bits; std::memcpy(&bits, raw.data() + 2 * i, 2);
            dst[i] = __half2float(__ushort_as_half(bits));
        } else {   // CUDA_R_16BF
            uint16_t bits; std::memcpy(&bits, raw.data() + 2 * i, 2);
            dst[i] = __bfloat162float(__ushort_as_bfloat16(bits));
        }
    }
}
void store(void* p, const float* src, size_t count, cudaDataType_t t, cudaStream_t s) {
    const size_t bytes = count * elem_bytes(t);
    std::vector<uint8_t> raw;
    const float* out = src;
    if (t != CUDA_R_32F) {
        raw.resize(bytes);
        for (size_t i = 0; i < count; ++i) {
            uint16_t bits;
            if (t == CUDA_R_16F) bits = __half_as_ushort(__float2half(src[i]));
            else bits = __bfloat16_as_ushort(__float2bfloat16(src[i]));
            std::memcpy(raw.data() + 2 * i, &bits, 2);
        }
        out = nullptr;
    }
    strata::vk::copy_h2d((void*)((std::uintptr_t) p - kPtrBias),   // compat handle -> raw arena offset
                 out ? (const void*) out : (const void*) raw.data(), bytes, s);
}
// Read a value at a host float-block position back into the operand type C wants (accumulate-then-store path).
float load_scalar(const void* p, cudaDataType_t t, size_t i) {
    if (t == CUDA_R_32F) { float f; std::memcpy(&f, (const char*)p + 4 * i, 4); return f; }
    uint16_t bits; std::memcpy(&bits, (const char*)p + 2 * i, 2);
    return t == CUDA_R_16F ? __half2float(__ushort_as_half(bits))
                           : __bfloat162float(__ushort_as_bfloat16(bits));
}
void store_scalar(void* p, cudaDataType_t t, size_t i, float v) {
    if (t == CUDA_R_32F) { std::memcpy((char*)p + 4 * i, &v, 4); return; }
    uint16_t bits = t == CUDA_R_16F ? __half_as_ushort(__float2half(v))
                                    : __bfloat16_as_ushort(__float2bfloat16(v));
    std::memcpy((char*)p + 2 * i, &bits, 2);
}

// op(X) element (i, j) of a (k?)-x-n matrix held column-major with leading dim ld:
//   OP_N:   X[i + j*ld]      (k x n)
//   OP_T:   X[j + i*ld]      (n x k stored, read transposed)
inline size_t at(cublasOperation_t op, int i, int j, int ld) {
    return op == CUBLAS_OP_N ? (size_t) i + (size_t) j * ld : (size_t) j + (size_t) i * ld;
}

// One strided-batch member.  A/B come from device, C read/updated on device.  m x n output, K the inner.
cublasStatus_t gemm_one(Handle& h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                        float alpha, const void* A, cudaDataType At, int lda,
                        const void* B, cudaDataType Bt, int ldb,
                        float beta, void* C, cudaDataType Ct, int ldc) {
    if (m <= 0 || n <= 0) return CUBLAS_STATUS_SUCCESS;
    if (!A || !B || !C) return CUBLAS_STATUS_INVALID_VALUE;
    if (ta != CUBLAS_OP_N && ta != CUBLAS_OP_T) return CUBLAS_STATUS_INVALID_VALUE;
    if (tb != CUBLAS_OP_N && tb != CUBLAS_OP_T) return CUBLAS_STATUS_INVALID_VALUE;
    if (!elem_bytes(At) || !elem_bytes(Bt) || !elem_bytes(Ct)) return CUBLAS_STATUS_NOT_SUPPORTED;
    if (At != CUDA_R_32F && At != CUDA_R_16F && At != CUDA_R_16BF) return CUBLAS_STATUS_NOT_SUPPORTED;
    // cuBLAS's contract: OP_N stores m x k (lda >= m); OP_T stores k x m (lda >= k).
    const long long ar = ta == CUBLAS_OP_N ? m : k, br = tb == CUBLAS_OP_N ? k : n;
    if ((long long) lda < ar || (long long) ldb < br || (long long) ldc < m) return CUBLAS_STATUS_INVALID_VALUE;

    // Blocked: one column-panel of C at a time (bounded host buffers); A and B fetched whole.  Correctness-
    // shaped, not a tuned GEMM - the Phase-3 kernels replace it.
    try {
        const size_t acount = (size_t)(ta == CUBLAS_OP_N ? (long long)lda * k : (long long)lda * m);
        const size_t bcount = (size_t)(tb == CUBLAS_OP_N ? (long long)ldb * n : (long long)ldb * k);
        std::vector<float> fa(acount), fb(bcount);
        fetch(A, fa.data(), acount, At, h.stream);
        fetch(B, fb.data(), bcount, Bt, h.stream);

        for (int j0 = 0; j0 < n; j0 += (int) kTile) {
            const int jn = n - j0 < (int) kTile ? n - j0 : (int) kTile;
            // read the C panel (device, column-major rows [0,m) of cols [j0, j0+jn))
            std::vector<float> crows((size_t) m * jn);
            for (int jj = 0; jj < jn; ++jj) {
                const size_t count = (size_t) m;
                std::vector<float> col(count);
                fetch((const char*) C + (size_t)(j0 + jj) * (size_t) ldc * elem_bytes(Ct),
                      col.data(), count, Ct, h.stream);
                for (int i = 0; i < m; ++i) crows[(size_t) i * jn + jj] = beta * col[i];
            }
            for (int i = 0; i < m; ++i)
                for (int jj = 0; jj < jn; ++jj) {
                    float acc = 0.f;
                    for (int kk = 0; kk < k; ++kk)
                        acc += fa[at(ta, i, kk, lda)] * fb[at(tb, kk, j0 + jj, ldb)];
                    crows[(size_t) i * jn + jj] += alpha * acc;
                }
            // write the panel back (column-major: one column at a time)
            for (int jj = 0; jj < jn; ++jj) {
                std::vector<float> col(m);
                for (int i = 0; i < m; ++i) col[i] = crows[(size_t) i * jn + jj];
                store((char*) C + (size_t)(j0 + jj) * (size_t) ldc * elem_bytes(Ct),
                      col.data(), (size_t) m, Ct, h.stream);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-vk cublas: %s\n", e.what());
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    *handle = new Handle{};
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) { delete (Handle*) handle; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    ((Handle*) handle)->stream = stream;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* stream) {
    if (!handle || !stream) return CUBLAS_STATUS_INVALID_VALUE;
    *stream = ((Handle*) handle)->stream;
    return CUBLAS_STATUS_SUCCESS;
}
// The workspace is a device pointer in the CUDA build; the CPU GEMM allocates its own host tiles, so the
// handle keeps the value (rebind() resets it) but never uses it.
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace, size_t bytes) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    Handle* h = (Handle*) handle;
    h->workspace = workspace; h->ws_bytes = bytes;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSetMathMode(cublasHandle_t, unsigned) { return CUBLAS_STATUS_SUCCESS; }  // always ordered fp32
// The parity test's arm selector (the same toggle the STRATA_VK_BLAS_CPU env makes for binaries):
// on = the GPU tiled kernel path, off = the ordered-fp32 CPU reference. -1 restores env control.
void strata_vk_blas_use_gpu(int on) { g_gpu_override = on < 0 ? -1 : (on ? 1 : 0); }
cublasStatus_t cublasGetMathMode(cublasHandle_t, unsigned* m) { if (m) *m = CUBLAS_DEFAULT_MATH; return CUBLAS_STATUS_SUCCESS; }

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
                              int m, int n, int k, const float* alpha, const float* A, int lda,
                              const float* B, int ldb, const float* beta, float* C, int ldc) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (blas_time_on()) blas_stat_shape("Sgemm", m, n, k, 1);
    BlasClock clk("Sgemm", m, n, k, 1);
    if (gemm_gpu(ta, tb, m, n, k, *alpha, A, CUDA_R_32F, lda, 0, B, CUDA_R_32F, ldb, 0,
                 *beta, C, CUDA_R_32F, ldc, 0, 1)) return CUBLAS_STATUS_SUCCESS;
    return gemm_one(*(Handle*) handle, ta, tb, m, n, k, *alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb,
                    *beta, C, CUDA_R_32F, ldc);
}
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
                                         int m, int n, int k, const float* alpha, const float* A, int lda,
                                         long long strideA, const float* B, int ldb, long long strideB,
                                         const float* beta, float* C, int ldc, long long strideC, int batchCount) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (blas_time_on()) blas_stat_shape("SgemmStridedBatched", m, n, k, batchCount);
    BlasClock clk("SgemmStridedBatched", m, n, k, batchCount);
    if (gemm_gpu(ta, tb, m, n, k, *alpha, A, CUDA_R_32F, lda, strideA, B, CUDA_R_32F, ldb, strideB,
                 *beta, C, CUDA_R_32F, ldc, strideC, batchCount)) return CUBLAS_STATUS_SUCCESS;
    for (int b = 0; b < batchCount; ++b) {
        cublasStatus_t s = gemm_one(*(Handle*) handle, ta, tb, m, n, k, *alpha,
                                    A + strideA * b, CUDA_R_32F, lda, B + strideB * b, CUDA_R_32F, ldb,
                                    *beta, C + strideC * b, CUDA_R_32F, ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
                            int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda,
                            const void* B, cudaDataType Btype, int ldb, const void* beta, void* C,
                            cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (blas_time_on()) blas_stat_shape("GemmEx", m, n, k, 1);
    BlasClock clk("GemmEx", m, n, k, 1);
    if (computeType != CUBLAS_COMPUTE_32F && computeType != CUBLAS_COMPUTE_32F_FAST_TF32)
        return CUBLAS_STATUS_NOT_SUPPORTED;      // fp64/integer compute types: the tree never asks for them
    float a = load_scalar(alpha, CUDA_R_32F, 0), b = load_scalar(beta, CUDA_R_32F, 0);
    if (gemm_gpu(ta, tb, m, n, k, a, A, Atype, lda, 0, B, Btype, ldb, 0, b, C, Ctype, ldc, 0, 1))
        return CUBLAS_STATUS_SUCCESS;
    return gemm_one(*(Handle*) handle, ta, tb, m, n, k, a, A, Atype, lda, B, Btype, ldb, b, C, Ctype, ldc);
}
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
                                          int m, int n, int k, const void* alpha, const void* A,
                                          cudaDataType Atype, int lda, long long strideA, const void* B,
                                          cudaDataType Btype, int ldb, long long strideB, const void* beta,
                                          void* C, cudaDataType Ctype, int ldc, long long strideC,
                                          int batchCount, cublasComputeType_t ct, cublasGemmAlgo_t algo) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (blas_time_on()) blas_stat_shape("GemmStridedBatchedEx", m, n, k, batchCount);
    BlasClock clk("GemmStridedBatchedEx", m, n, k, batchCount);
    const size_t ea = elem_bytes(Atype), eb = elem_bytes(Btype), ec = elem_bytes(Ctype);
    if (!ea || !eb || !ec) return CUBLAS_STATUS_NOT_SUPPORTED;
    {
        const float a = load_scalar(alpha, CUDA_R_32F, 0), b = load_scalar(beta, CUDA_R_32F, 0);
        if (gemm_gpu(ta, tb, m, n, k, a, A, Atype, lda, strideA, B, Btype, ldb, strideB,
                     b, C, Ctype, ldc, strideC, batchCount)) return CUBLAS_STATUS_SUCCESS;
    }
    for (int b2 = 0; b2 < batchCount; ++b2) {
        cublasStatus_t s = cublasGemmEx(handle, ta, tb, m, n, k, alpha,
                                        (const char*) A + strideA * b2 * (long long) ea, Atype, lda,
                                        (const char*) B + strideB * b2 * (long long) eb, Btype, ldb,
                                        beta, (char*) C + strideC * b2 * (long long) ec, Ctype, ldc, ct, algo);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
                           int m, int n, int k, const __half* alpha, const __half* A, int lda,
                           const __half* B, int ldb, const __half* beta, __half* C, int ldc) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    return gemm_one(*(Handle*) handle, ta, tb, m, n, k, __half2float(*alpha), A, CUDA_R_16F, lda,
                    B, CUDA_R_16F, ldb, __half2float(*beta), C, CUDA_R_16F, ldc);
}

}  // extern "C"
