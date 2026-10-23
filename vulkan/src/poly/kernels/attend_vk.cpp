// vulkan/src/poly/kernels/attend_vk.cpp - the Vulkan launcher for strata::poly::attend
// (include/strata/poly/kernels.hpp:197; the CUDA body is src/poly/kernels.cu's wrapper :2449 and
// its general chain: attn_score_reg_kernel :1155 / attn_softmax_kernel :1220 / attn_value_t
// :1499 / attn_value_kernel :1315).
//
// ARM SCOPE (deliberate): this launcher runs the general chain's reg<4,1,false> score arm ONLY -
// the one the CUDA wrapper picks when per % 4 == 0 && dim == 128 (:2488), the model's shape
// (Qwen3.6-35B: n_head 16, kv_heads 2 -> per 8, head dim 128). Everything else traps BY NAME:
//   - attn_split/attn_join (dim 64, per 4, CUDA 2460-2472): "attend: attn_split/join arm"
//   - reg<8,2>/reg<4,2>/reg<4,4>/attn_score_kernel:    "attend: score arm for dim != 128"
//   - kq8/vq8 quantized KV:                            "attend: kq8/vq8 quantized-KV arm"
// The value-arm pick (attn_value_t<2> vs <1> vs plain, CUDA 2518-2548) IS mirrored exactly, and
// window/ring/whole/at replay the wrapper's chunk loop (2473-2477) and the kernels' attn_rows/
// attn_span arithmetic as-is, so the model's decode rows=1 AND prompt rows<=chunk paths run the
// same three launches.
//
// Launch shapes (mirroring the CUDA <<<...>>>): score grid = grid(grid(keys,SCORE_KEYS=8) *
// kv_heads, SCORE_WARPS=8) blocks x 256 threads (2478-2480); softmax grid = n_head * c blocks
// (dim3(n_head, c) flattened x-fastest, the kernel recovers h = flat % n_head, tok = flat /
// n_head so CUDA 1228's (blockIdx.y * gridDim.x + blockIdx.x) indexing lands the same);
// value grid = kv_heads * dims * gy blocks x 1024 threads (dim3 flattened x-fastest, gxdim in
// the pc).
//
// wt predicate (CUDA 2511): per >= 4 && kv_heads * blocks * VALUE_QUERIES <= (chunk - c) *
// n_head -> weights lie behind the launch's scores in the same buffer, so the arena span of
// score is read as: c*n_head*keys (softmax/score/value-plain) + the t-kernel's kv_heads*blocks*
// keys*8 tail (separate binding, the overlap is the CUDA design).
//
// i64 pos/last/at: the kernels read the low u32 word only - the launcher traps positions past
// int31 (the sequence rows stay far under it; the same law as head_norm_rope_vk).
// stream: like every other launcher here, kernel_launch_sized enqueues on the runtime's queue in
// call order; the cudaStream_t carries no extra meaning on this backend.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "attend_score_4_1.spv.h"
#include "attend_score_8_2.spv.h"
#include "attend_score_4_2.spv.h"
#include "attend_score_4_4.spv.h"
#include "attend_softmax.spv.h"
#include "attend_value_t1.spv.h"
#include "attend_value_t2.spv.h"
#include "attend_value_plain.spv.h"

// force_subgroup32=false everywhere (the rms_norm/gdn reasoning): every shuffle offset here is
// <= 16 or an absolute read of lane < 8, and aligned 32-lane blocks never mix under wave64;
// attend_value.comp's plain arm compensates with gl_SubgroupInvocationID's & 32u, which is a
// no-op if the driver ever runs wave32.
// the reg variants the CUDA chain instantiates (2488-2499): (N, STEPS) -> its own compiled .spv.
const vk::Kernel& pl_score(int n, int steps) {
    static std::map<std::pair<int, int>, vk::Kernel> ks;
    auto it = ks.find({n, steps});
    if (it == ks.end()) {
        static const std::map<std::pair<int, int>, std::pair<const uint32_t*, size_t>> spvs = {
            {{4, 1}, {attend_score_4_1_spv, attend_score_4_1_spv_len}},
            {{8, 2}, {attend_score_8_2_spv, attend_score_8_2_spv_len}},
            {{4, 2}, {attend_score_4_2_spv, attend_score_4_2_spv_len}},
            {{4, 4}, {attend_score_4_4_spv, attend_score_4_4_spv_len}},
        };
        auto sp = spvs.find({n, steps});
        if (sp == spvs.end()) throw std::runtime_error("attend: score variant not compiled");
        char name[32];
        std::snprintf(name, sizeof name, "attend_score_%d_%d", n, steps);
        it = ks.emplace(std::make_pair(n, steps),
                        vk::kernel_load(name, sp->second.first, sp->second.second, 4, 64, 256, false)).first;
    }
    return it->second;
}
const vk::Kernel& pl_softmax() {
    static const vk::Kernel k = vk::kernel_load("attend_softmax", attend_softmax_spv, attend_softmax_spv_len,
                                                3, 64, 256, false);
    return k;
}
const vk::Kernel& pl_value(int tiles) {
    if (tiles == 2) {
        static const vk::Kernel k = vk::kernel_load("attend_value_t2", attend_value_t2_spv,
                                                    attend_value_t2_spv_len, 4, 64, 1024, false);
        return k;
    }
    if (tiles == 1) {
        static const vk::Kernel k = vk::kernel_load("attend_value_t1", attend_value_t1_spv,
                                                    attend_value_t1_spv_len, 4, 64, 1024, false);
        return k;
    }
    static const vk::Kernel k = vk::kernel_load("attend_value_plain", attend_value_plain_spv,
                                                attend_value_plain_spv_len, 4, 64, 1024, false);
    return k;
}
inline uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
inline long long gridll(long long n, long long block) { return (n + block - 1) / block; }
}  // namespace

void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window,
            int ring, bool whole, const long long* at, int room, size_t score_floats, Kvq8 kq8, Kvq8 vq8) {
    if (at && (whole || room < 1)) throw std::invalid_argument("attend: a position on the card needs its room");
    if (dim % 4 || uintptr_t(q) % 16 || uintptr_t(kcache) % 8)
        throw std::invalid_argument("attend: queries and keys are read four values at a time");
    if (std::getenv("STRATA_VK_TRACE_ATTEND"))
        std::fprintf(stderr, "vk attend: pos=%lld rows=%d chunk=%d dim=%d n_head=%d kv_heads=%d scale=%.9g "
                             "window=%d ring=%d whole=%d at=%d room=%d score_floats=%zu\n",
                     (long long)pos, rows, chunk, dim, n_head, kv_heads, (double) scale, window, ring,
                     int(whole), int(at != nullptr), room, score_floats);
    if (rows <= 0 || chunk <= 0 || n_head <= 0 || kv_heads <= 0 || n_head % kv_heads)
        throw std::invalid_argument("attend: bad shape");
    const bool kq = kq8.codes != nullptr, vq = vq8.codes != nullptr;
    const int per = n_head / kv_heads;
    // Arms this backend does not carry yet, named with the config that would run them.
    if (kq || vq) throw std::invalid_argument("attend: kq8/vq8 quantized-KV arm");
    // CUDA 2460's split/join gate, kept exact so the trap says which shape it is.
    const long long most = gridll(at ? room : pos + rows, 64);   // SPLIT_KEYS
    if (dim == 64 && per == 4 && window == 0 && ring == 0 && !whole &&
        size_t(chunk) * n_head * most * (dim + 2) <= score_floats)
        throw std::invalid_argument("attend: attn_split/join arm (dim 64, groups of 4, whole conversation)");
    // the CUDA wrapper's reg pick (2488-2499), mirrored exactly
    int sn = 0, ss = 0;
    if (per % 4 == 0 && dim == 128) { sn = 4; ss = 1; }
    else if (per % 8 == 0 && dim == 256) { sn = 8; ss = 2; }
    else if (per % 4 == 0 && dim == 256) { sn = 4; ss = 2; }
    else if (per % 4 == 0 && dim == 512) { sn = 4; ss = 4; }
    else throw std::invalid_argument("attend: plain attn_score_kernel arm (no reg variant)");
    if (pos > 0x7FFFFFFFll || pos + rows > 0x7FFFFFFFll || (long long)room > 0x7FFFFFFF || (long long)window > 0x7FFFFFFF)
        throw std::invalid_argument("attend: positions beyond int31 (kernels read one word of pos/at)");

    if (std::getenv("STRATA_VK_TRACE_ATTEND"))
        std::fprintf(stderr, "vk attend: pos=%lld rows=%d chunk=%d dim=%d n_head=%d kv_heads=%d scale=%.9g "
                             "window=%d ring=%d whole=%d at=%d room=%d score_floats=%zu\n",
                     (long long)pos, rows, chunk, dim, n_head, kv_heads, (double) scale, window, ring,
                     int(whole), int(at != nullptr), room, score_floats);

    const long long last = whole ? pos + rows - 1 : -1;
    const uint32_t posw = uint32_t(uint64_t(pos) & 0xFFFFFFFFull), lastw = uint32_t(uint64_t(last) & 0xFFFFFFFFull);
    const uint64_t qo = strata_vk_compat::off(q), ko = strata_vk_compat::off(kcache),
                   vo = strata_vk_compat::off(vcache), so = strata_vk_compat::off(score),
                   oo = strata_vk_compat::off(out), ayo = at ? strata_vk_compat::off(at) : 0;
    const uint64_t qsz = size_t(rows) * n_head * dim * 4, osz = size_t(rows) * n_head * dim * 4;
    // rows of the cache the kernels can walk: absolute index < pos + rows (window=0), or the
    // ring wraps under ring rows; with at, room bounds the span the launches were sized for.
    const uint64_t cache_rows = uint64_t(std::max<long long>(at ? room : 0, pos + rows));
    // +8: a lane reading the cache's LAST element (odd) also touches the following u32 word,
    // exactly like the CUDA u64 read of two halves.
    const uint64_t kcsz = cache_rows * kv_heads * dim * 2 + 8, vcsz = kcsz;

    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        // the launch's span (CUDA 2475-2477)
        const long long p = pos + first, lo = window > 0 && p >= window ? p - window + 1 : 0;
        const long long hi = whole ? last : p + c - 1;
        const int keys = at ? room : int(hi - lo) + 1;
        if (keys <= 0) continue;
        const uint32_t sb = bits(scale);

        // the value-block count and the wt tail, per CUDA 2510-2512 (VALUE_QUERIES = 8)
        const long long vblocks = gridll(size_t(c) * per, 8);
        const uint64_t wt_floats = size_t(kv_heads) * vblocks * keys * 8;
        float* wt = per >= 4 && size_t(kv_heads) * vblocks * 8 <= size_t(chunk - c) * n_head
                        ? score + size_t(c) * n_head * keys : nullptr;
        const uint64_t scsz = size_t(c) * n_head * keys * 4;

        // ---- score: grid(grid(keys,8)*kv_heads, 8) blocks of 256 (the reg arm)
        {
            const long long units = gridll(keys, 8) * kv_heads;
            const long long blocks = gridll(units, 8);
            if (blocks > 0x7FFFFFFFll) throw std::runtime_error("attend: score grid overflows");
            const uint64_t bufs[4] = {qo, ko, so, at ? ayo : so};
            const uint64_t sizes[4] = {qsz, kcsz, scsz, at ? 8ull : scsz};
            const uint32_t pc[16] = {uint32_t(keys), uint32_t(n_head), uint32_t(kv_heads), posw,
                                     uint32_t(first), uint32_t(c), sb, uint32_t(window), uint32_t(ring),
                                     lastw, at ? 1u : 0u, 0, 0, 0, 0, 0};
            vk::kernel_launch_sized(pl_score(sn, ss), int(blocks), bufs, sizes, 4, pc);
        }

        // ---- softmax: n_head * c blocks of 256
        {
            const long long n = size_t(n_head) * c;
            if (n > 0x7FFFFFFFll) throw std::runtime_error("attend: softmax grid overflows");
            const uint64_t bufs[3] = {so, wt ? so + scsz : so, at ? ayo : so};
            const uint64_t sizes[3] = {scsz, wt ? wt_floats * 4 : scsz, at ? 8ull : scsz};
            const uint32_t pc[16] = {uint32_t(n_head), uint32_t(keys), posw, uint32_t(first), uint32_t(c),
                                     uint32_t(window), lastw, at ? 1u : 0u, uint32_t(per), uint32_t(vblocks),
                                     wt ? 1u : 0u, 0, 0, 0, 0, 0};
            vk::kernel_launch_sized(pl_softmax(), int(n), bufs, sizes, 3, pc);
        }

        // ---- value: the wrapper's pick (CUDA 2518-2548) mirrored: tiles=2 when the launch still
        // has blocks for most of the card, 1 on the wt arm otherwise, plain without wt.
        const long long dims = gridll(dim, 32), gxdim = kv_heads * dims;
        const int tiles = wt ? (vblocks > 1 && kv_heads * dims * gridll(vblocks, 2) >= 32 ? 2 : 1) : 0;
        const long long gy = tiles == 2 ? gridll(vblocks, 2)
                                        : tiles == 1 ? vblocks : gridll(size_t(c) * per, 8);
        {
            const long long n = gxdim * gy;
            if (n > 0x7FFFFFFFll) throw std::runtime_error("attend: value grid overflows");
            const uint64_t bufs[4] = {wt ? so + scsz : so, vo, oo, at ? ayo : oo};
            const uint64_t sizes[4] = {wt ? wt_floats * 4 : scsz, vcsz, osz, at ? 8ull : osz};
            const uint32_t pc[16] = {uint32_t(keys), uint32_t(dim), uint32_t(n_head), uint32_t(kv_heads),
                                     posw, uint32_t(first), uint32_t(c), uint32_t(window), uint32_t(ring),
                                     lastw, at ? 1u : 0u, uint32_t(gxdim), wt ? 1u : 0u, 0, 0, 0};
            vk::kernel_launch_sized(pl_value(tiles), int(n), bufs, sizes, 4, pc);
        }
    }
    (void) stream;   // like every launcher here: the runtime's queue keeps call order
}

}  // namespace strata::poly
