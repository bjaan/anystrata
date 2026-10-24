// vulkan/src/poly/kernels/attend_batch_vk.cpp - the Vulkan launcher for
// strata::poly::attend_batch (include/strata/poly/kernels.hpp:219; the CUDA body is
// src/poly/kernels.cu's wrapper :2568-2662 and its kernels attn_group_kernel :1581 /
// attn_softmax_block_kernel :1602 / attn_scale_kernel :1631 / attn_ungroup_kernel :1590).
//
// This is the prefill/multi-token attention: the wrapper's chunk loop is transcribed whole -
// the chunk pick (2598-2600), the span arithmetic (2602-2604), the key-block split (2619-2624) -
// and the TWO cublasGemmStridedBatchedEx calls (2630-2637 scores, 2647-2651 values) are copied
// ARG FOR ARG onto the compat CPU cublas (vk_blas.cpp: the exact column-major contract, ordered
// fp32 accumulation, f16 operands converted by the tree's tested helpers). The flash64 tensor
// path (2575-2582) and the kq8/vq8 dequant arms (2587-2592, 2625-2630) trap BY NAME, like
// attend_vk.cpp's unported arms.
//
// The score/weight/og/qg index algebra was checked against the batched-GEMM strides: with
// cols = c*per and batch strideC = cols*m, C_g(row i, col j') lands at score + (g*cols + j')*m
// + i = score + blk*m + i, blk the softmax's group-order row - so the .comp's base = blk*m is
// exact, and the values' B reads weight at the same element index the softmax wrote.
//
// stream: like every launcher here, kernel_launch_sized and the compat cublas act on the
// runtime's queue/staged copies in call order (each launch/copy drains), so the enqueue-order
// law of the batch path holds; the cudaStream_t carries no extra meaning on this backend.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"
#include "strata/vulkan_compat/cublas_v2.h"
#include "strata/vulkan_compat/cuda_fp16.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "attend_batch_group.spv.h"          // attn_group_kernel arm (UNGROUP=0)
#include "attend_batch_group_ungroup.spv.h"  // attn_ungroup_kernel arm
#include "attend_batch_softmax.spv.h"
#include "attend_batch_scale.spv.h"

// kernels.cu:27 - the results cap the wrapper's chunk pick.
constexpr size_t kAttendResults = size_t(8) << 20;

const vk::Kernel& pl_group() {
    static const vk::Kernel k = vk::kernel_load("attend_batch_group", attend_batch_group_spv,
                                                attend_batch_group_spv_len, 2, 64, 256, false);
    return k;
}
const vk::Kernel& pl_ungroup() {
    static const vk::Kernel k = vk::kernel_load("attend_batch_group_ungroup",
                                                attend_batch_group_ungroup_spv,
                                                attend_batch_group_ungroup_spv_len, 2, 64, 256, false);
    return k;
}
const vk::Kernel& pl_softmax_block() {
    static const vk::Kernel k = vk::kernel_load("attend_batch_softmax", attend_batch_softmax_spv,
                                                attend_batch_softmax_spv_len, 3, 64, 128, false);
    return k;
}
const vk::Kernel& pl_scale() {
    static const vk::Kernel k = vk::kernel_load("attend_batch_scale", attend_batch_scale_spv,
                                                attend_batch_scale_spv_len, 2, 64, 256, false);
    return k;
}
inline long long gridll(long long n, long long block) { return (n + block - 1) / block; }
}  // namespace

void attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                  float scale, void* stream, int window, int ring, bool whole, Kvq8 kq8, Kvq8 vq8, uint16_t* kwork,
                  uint16_t* vwork, size_t work_halves) {
    const int per = n_head / kv_heads, width = kv_heads * dim;
    if (rows <= 0 || n_head <= 0 || kv_heads <= 0 || n_head % kv_heads || dim % 2 || dim <= 0)
        throw std::invalid_argument("attend_batch: bad shape");
    const bool kq = kq8.codes != nullptr, vq = vq8.codes != nullptr;
    // CUDA 2575's dim==64 per==4 fast path requires tensor_cores(), which is false on the
    // AMD builds (kernels.cu:1632-1636's own note); the general chain below runs the same math
    // there.  Mirrored: no flash arm here, every shape goes through the batched products.
    if (kq || vq) throw std::invalid_argument("attend_batch: kq8/vq8 quantized-KV arm");
    if (pos < 0 || pos > 0x7FFFFFFFll || pos + rows > 0x7FFFFFFFll ||
        (long long) window > 0x7FFFFFFFll || (long long) ring > 0x7FFFFFFFll)
        throw std::invalid_argument("attend_batch: positions beyond int31 (the .comp kernels read one word)");
    const cublasHandle_t h = (cublasHandle_t) blas;
    if (!h) throw std::invalid_argument("attend_batch: null cuBLAS handle");
    if (score == nullptr || work == nullptr || out == nullptr)
        throw std::invalid_argument("attend_batch: null scratch");
    if (std::getenv("STRATA_VK_TRACE_ATTEND"))
        std::fprintf(stderr, "vk attend_batch: pos=%lld rows=%d dim=%d n_head=%d kv_heads=%d scale=%.9g "
                             "window=%d ring=%d whole=%d score_floats=%zu\n",
                     (long long) pos, rows, dim, n_head, kv_heads, (double) scale, window, ring, int(whole),
                     score_floats);
    const float one = 1.0f, zero = 0.0f;
    const long long last = whole ? pos + rows - 1 : -1;
    // CUDA 2596-2597: two thirds of the scratch for a block's scores, the rest for the f16
    // weights behind them in the same buffer (the overlap is the CUDA design). The weight
    // third is zeroed once a BLOCK (below): a shorter tail block rewrites words the longer
    // block before it left behind, and the .comp's atomicOr builds words from zeros.
    const size_t cells = score_floats / 3 * 2;
    if (cells < (size_t) n_head * 16 || cells >= score_floats)
        throw std::invalid_argument("attend_batch: score scratch too small");
    half* weight = (half*) (score + cells);
    const uint64_t so = strata_vk_compat::off(score), qo = strata_vk_compat::off(q),
                   ko = strata_vk_compat::off(kcache), vo = strata_vk_compat::off(vcache),
                   oo = strata_vk_compat::off(out), wo = strata_vk_compat::off(work);

    for (int first = 0, c = 0; first < rows; first += c) {
        // CUDA 2600-2601's chunk pick, transcribed
        c = (int) std::min({size_t(rows - first), size_t(65535 / n_head), cells / n_head,
                            std::max<size_t>(16, kAttendResults / (size_t(n_head) * dim * sizeof(float)))});
        const long long p0 = pos + first;   // the chunk's first position
        const long long lo0 = window > 0 && p0 >= window ? p0 - window + 1 : 0;
        const int n_keys = int((whole ? last : p0 + c - 1) - lo0) + 1;
        const size_t heads = size_t(c) * n_head;
        half* qg = (half*) work;                      // the queries by group
        float* og = work + (heads * dim + 1) / 2;     // the results by group
        float* top = og + heads * dim;                // CUDA 2615-2617: the running softmax state
        float* sum = top + heads;
        float* shrink = sum + heads;
        const uint64_t work_floats = (heads * dim + 1) / 2 + heads * dim + heads * 3;  // through shrink
        const uint32_t gx = uint32_t(gridll(dim, 256));
        const uint64_t og_bo = ((heads * dim + 1) / 2) * 4;   // og's byte offset in the work arena

        // CUDA 2611-2612: the chunk's queries to the group order (halves the products take).
        {
            const uint64_t bufs[2] = {qo + uint64_t(first) * n_head * dim * 4, wo};
            const uint64_t sizes[2] = {uint64_t(rows - first) * n_head * dim * 4, (heads * dim + 1) * 2};
            const uint32_t pc[16] = {uint32_t(dim), uint32_t(n_head), uint32_t(kv_heads), uint32_t(c), gx};
            vk::kernel_launch_sized(pl_group(), int(gx * heads), bufs, sizes, 2, pc);
        }
        // zero the weight words this block writes (heads*m halves, one extra byte of word
        // round-up): the softmax builds words with atomicOr, and a shorter tail block reuses
        // the region a longer block before it filled.
        // CUDA 2614-2616: the rows go a block at a time so a product has the chunk's every
        // token for its columns however long the context is.
        const long long cols = (long long) c * per;
        const int fit = (int) (cells / heads);
        const int block = fit > 16 ? fit & ~15 : fit;   // the tensor cores are half as fast on a ragged block
        for (int k0 = 0, m = 0; k0 < n_keys; k0 += m) {
            // CUDA 2619-2624: a block ends where a ring starts over (the q8 workspace cap is a
            // trap arm - kq/vq threw above).
            const long long row = ring > 0 ? (lo0 + k0) % ring : lo0 + k0;
            m = (int) std::min<long long>(std::min<long long>(block, n_keys - k0),
                                          ring > 0 ? ring - row : n_keys);
            if (m <= 0) throw std::runtime_error("attend_batch: a key block of zero rows");
            vk::fill_d((void*) (so + cells * 4), 0, (heads * size_t(m) + 1) / 2 * 4, stream);
            const uint16_t* krow = kcache + size_t(row) * width;   // biased compat pointers go to cublas as-is
            const uint16_t* vrow = vcache + size_t(row) * width;
            // CUDA 2630-2637, ARG FOR ARG: score_g = scale * K_g^T Q_g (m x cols, K dim).
            const cublasStatus_t s1 = cublasGemmStridedBatchedEx(
                h, CUBLAS_OP_T, CUBLAS_OP_N, m, (int) cols, dim, &scale,
                krow, CUDA_R_16F, width, dim,
                qg, CUDA_R_16F, dim, cols * dim, &zero,
                score, CUDA_R_32F, m, cols * m,
                kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (s1 != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("attend_batch: attention scores failed in the compat cuBLAS");
            // CUDA 2641-2642: the block's softmax, c*n_head blocks of 128.
            {
                const uint64_t bufs[3] = {so, so + cells * 4, wo};
                const uint64_t sizes[3] = {heads * m * 4, (score_floats - cells) * 4, work_floats * 4};
                const uint32_t pc[16] = {uint32_t(m), uint32_t(c), uint32_t(per), uint32_t(first),
                                         uint32_t(window), uint32_t(k0 == 0),
                                         uint32_t(uint64_t(lo0 + k0)), uint32_t(uint64_t(pos)),
                                         uint32_t(uint64_t(last)),
                                         uint32_t((heads * dim + 1) / 2 + heads * dim),
                                         uint32_t((heads * dim + 1) / 2 + heads * dim + heads),
                                         uint32_t((heads * dim + 1) / 2 + heads * dim + heads * 2)};
                vk::kernel_launch_sized(pl_softmax_block(), int(heads), bufs, sizes, 3, pc);
            }
            if (k0 > 0) {
                // CUDA 2645: the results so far shrink to the new top (times, not divide).
                const uint64_t bufs[2] = {wo + og_bo, wo};
                const uint64_t sizes[2] = {heads * dim * 4, work_floats * 4};
                const uint32_t pc[16] = {uint32_t(dim), gx, 0u,
                                         uint32_t((heads * dim + 1) / 2 + heads * dim + heads * 2)};  // shrink base
                vk::kernel_launch_sized(pl_scale(), int(gx * heads), bufs, sizes, 2, pc);
            }
            // CUDA 2647-2651, ARG FOR ARG: og_g += W_g V_g (dim x cols, K m), accumulating from
            // the second block on.
            const cublasStatus_t s2 = cublasGemmStridedBatchedEx(
                h, CUBLAS_OP_N, CUBLAS_OP_N, dim, (int) cols, m, &one,
                vrow, CUDA_R_16F, width, dim,
                weight, CUDA_R_16F, m, cols * m,
                k0 > 0 ? &one : &zero, og, CUDA_R_32F,
                dim, cols * dim, kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (s2 != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("attend_batch: attention values failed in the compat cuBLAS");
        }
        // CUDA 2653: the final divide by each row's sum.
        {
            const uint64_t bufs[2] = {wo + og_bo, wo};
            const uint64_t sizes[2] = {heads * dim * 4, work_floats * 4};
            const uint32_t pc[16] = {uint32_t(dim), gx, 1u,
                                     uint32_t((heads * dim + 1) / 2 + heads * dim + heads)};  // sum base
            vk::kernel_launch_sized(pl_scale(), int(gx * heads), bufs, sizes, 2, pc);
        }
        // CUDA 2654-2655: the results back to the engine's order.
        {
            const uint64_t bufs[2] = {wo + og_bo, oo + uint64_t(first) * n_head * dim * 4};
            const uint64_t sizes[2] = {heads * dim * 4, uint64_t(rows - first) * n_head * dim * 4};
            const uint32_t pc[16] = {uint32_t(dim), uint32_t(n_head), uint32_t(kv_heads), uint32_t(c), gx};
            vk::kernel_launch_sized(pl_ungroup(), int(gx * heads), bufs, sizes, 2, pc);
        }
    }
    (void) kwork; (void) vwork; (void) work_halves;   // the q8 workspace: that arm traps above
    (void) stream;   // queue/staged copies keep call order
}

}  // namespace strata::poly
