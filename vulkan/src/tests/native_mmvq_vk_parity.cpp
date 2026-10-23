// vulkan/src/tests/native_mmvq_vk_parity.cpp - the IQ2_XXS native_mmvq contract on Vulkan.
//
// Reference: a scalar transcription of iq_kernels.cu's mmvq_kernel<TY=16> + vec_dot_iq2_xxs_q8_1
// (101-125): the same k-lane partition (k = lane, lane+32, ...; kbx = k/8, iqs = 2*(k%8)), the same
// iqs/k0/j order inside the dot, the same integer semantics (vcmpne4/vsub4/dp4a exact, truncation
// toward zero on sumi*ls/8), the same fp order (dw*d8)*sumi, and the warp_sum xor tree 16,8,4,2,1
// emulated lane-for-lane. Every device op is an exact integer op or an unfused fp32 mul/add, and
// iq_kernels.cu:699-701 pins mmvq_kernel bitwise-equal to the Split/multi kernel the CUDA build
// runs — so the assertion here is BITWISE (==), the standard mmvq_multi_parity.cpp holds. Never
// loosen it.
//
// Data (mmvq_multi_parity.cpp's law): random IQ2_XXS blocks (any byte pattern is legal — the grid
// index bytes address the 256-entry table through & 255) with the block fp16 scale d rewritten to
// a normal fp16 in +-[2^-10, 2^-5) so no inf/NaN can mask a difference. Activations are the
// quantize_q8_1.comp's OWN q8_1 bytes for a random vector (engine-shaped: wrap() pointers).
// Shapes: n_in 2048/768/256, n_out 64/3, ncols 1..3 (the launcher loops c < ncols, bitwise the
// same per column as one call).
//
// Type 13 (Q5_K, Task 2.3): the reference is a scalar transcription of native_mmvq.cu's
// q5_q8_dot + q5_q8_dot_impl (243-299: exact word reads from Q5KBlock — dm@0, scales@4, qh@16,
// qs@48 — the 6x16-bit scale decode, the 0x0f0f0f0f/0x10101010 nibble+high-bit assembly, the
// two dp4a pairs per i, dm5f.x*sumf_d - dm5f.y*sumf_m) over the native_q5_k_mmvq_kernel
// partition (309-348): tid = 0..127, kbx = tid/16 + 8t, kqs = 2*(tid%16); rows=1 (n_in>=2048):
// 4 subrow partials summed lane-wise in subrow order then the warp xor tree; rows=4 (SmallK,
// n_in<2048): the same with 4 rows per group. Q5KBlock scales (half2 dm) are rewritten to sane
// fp16s like the iq2 case. mmvq_multi_parity.cpp holds the same partition BITWISE on CUDA.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"
#include "iq2xxs_grid_vk.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

uint16_t f16_rn(float f) {                       // round-to-nearest-even (matches the .comp / CUDA)
    uint32_t b; std::memcpy(&b, &f, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    int exp = int((b >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = b & 0x7FFFFFu, h;
    if (exp >= 31) h = 0x7C00u;
    else if (exp <= 0) {
        man |= 0x800000u;
        const uint32_t sh = uint32_t(1 - exp) + 13u;
        h = (sh >= 32u) ? 0u : (man >> sh);
        const uint32_t rem = (sh >= 32u) ? 0u : (man & ((1u << sh) - 1u));
        if (sh < 32u && (rem > (1u << (sh - 1u)) || (rem == (1u << (sh - 1u)) && (h & 1u) != 0u))) h++;
    } else {
        h = man >> 13u; uint32_t rem = man & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) { h++; if (h == 0x400u) { h = 0u; exp++; if (exp >= 31) h = 0x7C00u; } }
        if (exp < 31) h |= uint32_t(exp) << 10u;
    }
    return uint16_t(sign | h);
}
float f16_to_f32(uint16_t hh) {
    const uint32_t sign = (uint32_t)(hh >> 15) & 1u;
    uint32_t exp = (hh >> 10) & 0x1Fu, man = hh & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else { exp = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; --exp; } man &= 0x3FFu;
               f = (sign << 31) | (exp << 23) | (man << 13); }
    } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
    else f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    float out; std::memcpy(&out, &f, 4); return out;
}

int sbyte(uint32_t a, int i) { const uint32_t b = (a >> (8 * i)) & 255u; return b >= 128u ? int(b) - 256 : int(b); }
uint32_t vcmpne4(uint32_t a) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= (((a >> (8 * i)) & 255u) ? 0xFFu : 0u) << (8 * i);
    return r;
}
uint32_t vsub4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= (((a >> (8 * i)) - (b >> (8 * i))) & 0xFFu) << (8 * i);
    return r;
}
int dp4a(uint32_t x, uint32_t y, int acc) { for (int i = 0; i < 4; ++i) acc += sbyte(x, i) * sbyte(y, i); return acc; }
uint32_t unpack_ksigns(uint32_t v) { return (v ^ ((uint32_t(__builtin_popcount(v)) & 1u) << 7u)) * 0x01010101u; }

// vec_dot_iq2_xxs_q8_1(bq2, bq8_1[iqs/2], iqs) transcribed; q8 points at the block's byte 0
// (word 0 = ds, bytes 4..35 = qs).
float vdot_iq2_xxs(const uint8_t* bq2, const uint8_t* q8, int iqs) {
    uint16_t dh; std::memcpy(&dh, bq2, 2);
    const float dw = f16_to_f32(dh);
    uint32_t q2 = 0, aux32 = 0;
    std::memcpy(&q2,    bq2 + 2 + 4 * iqs,     4);   // qs is uint16[32]: get_int_b2's pair = 4 bytes at 2+4*iqs
    std::memcpy(&aux32, bq2 + 2 + 4 * (iqs + 1), 4);
    const uint8_t* aux8 = (const uint8_t*)&q2;
    int sumi = 0;
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint64_t g = strata::vk::kIq2xxsGrid[aux8[k0 / 2]];
        const uint32_t gx = uint32_t(g), gy = uint32_t(g >> 32);          // ((const uint2*)grid)[idx]
        const uint32_t signs = unpack_ksigns((aux32 >> (7 * (k0 / 2))) & 0xFFu);
        const uint32_t s0 = vcmpne4(signs & 0x08040201u);
        uint32_t u0; std::memcpy(&u0, q8 + 4 + k0 * 4, 4);                // get_int_b4(qs, k0)
        sumi = dp4a(vsub4(gx ^ s0, s0), u0, sumi);
        const uint32_t s1 = vcmpne4(signs & 0x80402010u);
        uint32_t u1; std::memcpy(&u1, q8 + 4 + (k0 + 1) * 4, 4);
        sumi = dp4a(vsub4(gy ^ s1, s1), u1, sumi);
    }
    const int ls = int((aux32 >> 27) | 1u);
    sumi = (sumi * ls) / 8;                                              // C truncation toward zero
    uint16_t d8h; std::memcpy(&d8h, q8, 2);                              // __low2float(ds)
    return dw * f16_to_f32(d8h) * float(sumi);
}

// ---- Q5_K (type 13): transcription of q5_q8_dot / q5_q8_dot_impl (native_mmvq.cu:243-299).
int32_t rd_i32(const uint8_t* p) { int32_t v; std::memcpy(&v, p, 4); return v; }
uint16_t rd_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

float vdot_q5_k(const uint8_t* w, const uint8_t* x, int iqs) {   // w = block bytes, x = q8 chain bytes
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int col = (iqs / 2) % 4;
    const int* ql = nullptr; (void)ql;
    const int vl0 = rd_i32(w + 48 + 16 * bq8_offset + 4 * col);        // qs@48
    const int vl1 = rd_i32(w + 48 + 16 * bq8_offset + 4 * col + 16);   // ql[4]
    const int vh0 = rd_i32(w + 16 + 4 * col) >> bq8_offset;            // qh[0] >> off (arith)
    const int vh1 = rd_i32(w + 16 + 4 * col + 16) >> bq8_offset;       // qh[4]
    const uint16_t* scales; (void)scales;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = rd_u16(w + 4 + 2 * jm);      // scales[jm]
    const uint32_t s2 = rd_u16(w + 4 + 2 * (jm + 2));// scales[jm+2]
    const uint32_t s4 = rd_u16(w + 4 + 2 * (jm + 4));// scales[jm+4]
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* m  = sc + 2;
    float sumf_d = 0.0f, sumf_m = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const uint8_t* bq8i = x + (bq8_offset + i) * 36;
        const float d8 = f16_to_f32(rd_u16(bq8i));                     // __low2float(ds)
        const int u0 = rd_i32(bq8i + 4 + 4 * col);                     // (int*)qs + col
        const int u1 = rd_i32(bq8i + 4 + 4 * col + 16);                // q8[4]
        const int vl0i = (vl0 >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl1 >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh0 >> i) << 4) & 0x10101010;
        const int vh1i = ((vh1 >> i) << 4) & 0x10101010;
        const int dot1 = dp4a(uint32_t(vl0i | vh0i), uint32_t(u0),
                              dp4a(uint32_t(vl1i | vh1i), uint32_t(u1), 0));
        const int dot2 = dp4a(0x01010101u, uint32_t(u0), dp4a(0x01010101u, uint32_t(u1), 0));
        // contraction probe: fused (what CUDA's default fp model produces) unless STRATA_REF_UNFUSED
        sumf_d = std::fma(d8, float(dot1 * int(sc[i])), sumf_d);
        sumf_m = std::fma(d8, float(dot2 * int(m[i])), sumf_m);
    }
    const float dm_x = f16_to_f32(rd_u16(w));                          // __half22float2(dm)
    const float dm_y = f16_to_f32(rd_u16(w + 2));
    // contraction-verified (this box, RADV gfx1013 + CUDA native_mmvq.cu's --use_fast_math fusion
    // model): the final `dm_x*sumf_d - dm_y*sumf_m` answers as fma(-dm_y, sumf_m, dm_x*sumf_d) —
    // the negated-product operand folds into the fma with no separate rounding. Probed all three
    // fusion orders; this is the only bitwise match (the un-fused orders differ on 108-321 outputs).
    return std::fma(-dm_y, sumf_m, dm_x * sumf_d);
}

// emulate native_q5_k_mmvq_kernel<SmallK> for one (row, column): part[subrow][lane] accumulates,
// in the CUDA loop order, the dots tid = subrow*32 + lane performs: kbx = tid/16 + 8t (ascending),
// kqs = 2*(tid%16). rows==1: all four subrows sum one row; rows==4: subrow i does row0+i.
// warp 0 then sums subrow-ascending and runs the xor tree; lane i writes row0+i.
float q5_k_ref(const uint8_t* wr, const uint8_t* xcol, int nb, int row, int rows, int row_in_group) {
    float part[4][32];
    for (int s = 0; s < 4; ++s) for (int l = 0; l < 32; ++l) part[s][l] = 0.f;
    for (int tid = 0; tid < 128; ++tid) {
        const int s = tid / 32, lane = tid % 32, kqs = 2 * (tid % 16);
        const int r = (rows == 1) ? row : row + row_in_group;   // guarded by caller for rows==4
        float acc = 0.f;
        for (int kbx = tid / 16; kbx < nb; kbx += 8)
            acc += vdot_q5_k(wr + (size_t(r) * nb + kbx) * 176,
                             xcol + size_t(kbx * 8) * 36, kqs);
        part[s][lane] += acc;
    }
    float a[32];
    for (int l = 0; l < 32; ++l)
        a[l] = ((part[0][l] + part[1][l]) + part[2][l]) + part[3][l];
    float n[32];
    for (int o = 16; o > 0; o >>= 1) { for (int l = 0; l < 32; ++l) n[l] = a[l] + a[l ^ o]; std::memcpy(a, n, sizeof a); }
    return a[row_in_group];   // lane i's post-tree value answers row0+i
}

// ---- Q6_K (type 14): transcription of q6_q8_dot / q6_q8_dot_impl (native_mmvq.cu:773-803).
// Q6KBlock bytes: ql[128]@0, qh[64]@128, scales[16]@192 (int8), d (half)@208; 210 B, not aligned.
// Returns d and sumf separately: the kbx accumulation `tmp[i] += q6_q8_dot(...)` inlines the dot,
// and CUDA's --use_fast_math / RADV's NIR fuse `acc + d*sumf` to fma(d, sumf, acc) (the q5_k
// contraction lineage, PORT-NOTES law 6) — visible only when a thread takes >1 kbx iteration,
// which is exactly where the unfused host arm disagreed by 1-2 ULP and rows=4 (<=1 iter) matched.
void q6_k_dot_df(const uint8_t* w, const uint8_t* x, int iqs, float& d, float& sumf) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = rd_i32(w + 4 * iqs);                                   // load_int_b2(ql, iqs)
    const int vh = rd_i32(w + 128 + 4 * (8 * (iqs / 16) + iqs % 8)) >> vh_shift;
    const int8_t* sc = (const int8_t*)(w + 192 + scale_offset);           // scales[]
    sumf = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const uint8_t* bq8i = x + (bq8_offset + 2 * i) * 36;
        const int u  = rd_i32(bq8i + 4 + 4 * (iqs % 8));                  // (int*)qs[iqs % 8]
        const float d8 = f16_to_f32(rd_u16(bq8i));                        // __low2float(ds)
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi  = int(vsub4(uint32_t(vil | vih), 0x20202020u));     // vsubss4 (in-range, exact)
        sumf = std::fma(d8, float(dp4a(uint32_t(vi), uint32_t(u), 0) * int(sc[4 * i])), sumf);
    }
    d = f16_to_f32(rd_u16(w + 208));
}

// same partition for Q6_K (native_q6_k_mmvq_kernel :806-842): tid = 0..127, kbx = tid/32 + 4t
// (VDR=1, BLOCKS_PER_ITER=4), kqs = tid%32; rows=1: all four subrows sum one row; rows=4: subrow
// i does row0+i. warp 0 sums subrow-ascending then the xor tree; lane i writes row0+i.
float q6_k_ref(const uint8_t* wr, const uint8_t* xcol, int nb, int row, int rows, int row_in_group) {
    float part[4][32];
    for (int s = 0; s < 4; ++s) for (int l = 0; l < 32; ++l) part[s][l] = 0.f;
    for (int tid = 0; tid < 128; ++tid) {
        const int s = tid / 32, lane = tid % 32, kqs = tid % 32;
        const int r = (rows == 1) ? row : row + row_in_group;
        float acc = 0.f;
        for (int kbx = tid / 32; kbx < nb; kbx += 4) {
            float d, sumf;
            q6_k_dot_df(wr + (size_t(r) * nb + kbx) * 210,
                        xcol + size_t(kbx * 8) * 36, kqs, d, sumf);
            acc = std::fma(d, sumf, acc);   // contraction: tmp += d*sumf -> fma (see above)
        }
        part[s][lane] += acc;
    }
    float a[32];
    for (int l = 0; l < 32; ++l)
        a[l] = ((part[0][l] + part[1][l]) + part[2][l]) + part[3][l];
    float n[32];
    for (int o = 16; o > 0; o >>= 1) { for (int l = 0; l < 32; ++l) n[l] = a[l] + a[l ^ o]; std::memcpy(a, n, sizeof a); }
    return a[row_in_group];   // lane i's post-tree value answers row0+i
}

}  // namespace

using namespace strata;

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
    try { vk::device_info(); } catch (const std::exception& e) {
        std::fprintf(stderr, "native_mmvq_vk_parity: %s\n", e.what());
        std::fprintf(stderr, "native_mmvq_vk_parity SKIP (no usable Vulkan device)\n");
        return selftest ? 0 : 77;
    }
    vk::Arena& a = vk::Arena::instance();
    int fails = 0, compared = 0;

    struct Shape { int n_in, n_out, ncols; };
    const Shape shapes[] = {{2048, 64, 1}, {2048, 64, 2}, {768, 64, 3}, {256, 3, 1}};

    for (const Shape& s : shapes) {
        const int nb = s.n_in / 256;
        std::mt19937 rng(1234 + s.n_in);
        std::uniform_real_distribution<float> sc(0.0009765625f, 0.03125f);   // d in [2^-10, 2^-5)
        std::vector<uint8_t> W(size_t(s.n_out) * nb * 66);
        for (auto& blk : W) blk = uint8_t(rng() & 0xFF);                     // random legal bytes
        for (int r = 0; r < s.n_out; ++r)
            for (int b = 0; b < nb; ++b) {
                const uint16_t dh = f16_rn((rng() & 1) ? -sc(rng) : sc(rng));
                std::memcpy(W.data() + (size_t(r) * nb + b) * 66, &dh, 2);   // finite fp16 scale
            }
        std::vector<float> x(size_t(s.n_in) * s.ncols);
        for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 250.0f;

        const uint64_t w_sz = W.size(), w_pad = (w_sz + 3) & ~3ull;          // words read up to a pad
        const uint64_t x_sz = size_t(s.n_in) / 32 * s.ncols * 36;
        const uint64_t y_sz = size_t(s.ncols) * s.n_out * 4;
        const uint64_t ax = a.alloc(x.size() * 4), aq = a.alloc(x_sz);
        const uint64_t aw = a.alloc(w_pad + 4), ay = a.alloc(y_sz);
        vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
        vk::copy_h2d((void*)aw, W.data(), w_sz, nullptr);
        vk::fill_d((void*)ay, 0xFE, y_sz, nullptr);
        kernels::native_quantize_q8_1((const float*)strata_vk_compat::wrap(ax), strata_vk_compat::wrap(aq),
                                      s.n_in, s.ncols, strata_vk_compat::wrap(1000));
        std::vector<uint8_t> Q(x_sz);
        vk::copy_d2h(Q.data(), (void*)aq, x_sz, nullptr);

        kernels::native_mmvq(16, strata_vk_compat::wrap(aw), strata_vk_compat::wrap(aq),
                             (float*)strata_vk_compat::wrap(ay), s.n_in, s.n_out, s.ncols,
                             strata_vk_compat::wrap(1000));
        std::vector<float> y(size_t(s.ncols) * s.n_out);
        vk::copy_d2h(y.data(), (void*)ay, y_sz, nullptr);

        int shape_fails = 0;
        for (int c = 0; c < s.ncols; ++c)
            for (int r = 0; r < s.n_out; ++r) {
                const uint8_t* wr = W.data() + size_t(r) * nb * 66;
                const uint8_t* xcol = Q.data() + size_t(c) * nb * 8 * 36;
                float part[32] = {0.f};
                for (int k = 0; k < nb * 8; ++k) {                          // lane k%32's call k
                    const int kbx = k / 8, iqs = 2 * (k % 8);
                    part[k % 32] += vdot_iq2_xxs(wr + kbx * 66, xcol + size_t(kbx * 8 + iqs / 2) * 36, iqs);
                }
                float acc[32], nxt[32];
                std::memcpy(acc, part, sizeof part);
                for (int o = 16; o > 0; o >>= 1) {                          // warp_sum's xor tree
                    for (int l = 0; l < 32; ++l) nxt[l] = acc[l] + acc[l ^ o];
                    std::memcpy(acc, nxt, sizeof nxt);
                }
                const float ref = acc[0], got = y[size_t(c) * s.n_out + r];
                ++compared;
                if (got != ref) {
                    uint32_t gb, rb; std::memcpy(&gb, &got, 4); std::memcpy(&rb, &ref, 4);
                    if (shape_fails < 24)
                        std::printf("  MISMATCH n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                    s.n_in, c, r, got, ref, gb, rb);
                    ++shape_fails; ++fails;
                }
            }
        std::printf("native_mmvq iq2_xxs vk vs reference: n_in %d n_out %d ncols %d: %d outputs, %d differ bitwise\n",
                    s.n_in, s.n_out, s.ncols, s.ncols * s.n_out, shape_fails);
    }
    // ---- type 13 (Q5_K): n_in 2048 -> rows=1 (nb 8, SmallK=false); 768/256 -> rows=4 (SmallK=true)
    const struct { int n_in, n_out, ncols; } q5_shapes[] = {
        {2048, 64, 1}, {2048, 64, 4}, {768, 64, 3}, {768, 3, 1}, {256, 3, 2}};
    for (const auto& s : q5_shapes) {
        const int nb = s.n_in / 256;
        const int rows = (nb < 8) ? 4 : 1;            // mirrors the launcher's SmallK pick
        std::mt19937 rng(9900 + s.n_in * 7 + s.ncols);
        std::vector<uint8_t> W(size_t(s.n_out) * nb * 176);
        for (auto& blk : W) blk = uint8_t(rng() & 0xFF);
        for (int r = 0; r < s.n_out; ++r)
            for (int b = 0; b < nb; ++b) {           // sane fp16 dm halves (mmvq_multi_parity's law)
                const uint16_t dh = f16_rn((rng() & 1) ? -0.01f : 0.02f);
                const uint16_t dl = f16_rn((rng() & 1) ? -0.02f : 0.01f);
                std::memcpy(W.data() + (size_t(r) * nb + b) * 176, &dh, 2);
                std::memcpy(W.data() + (size_t(r) * nb + b) * 176 + 2, &dl, 2);
            }
        std::vector<float> x(size_t(s.n_in) * s.ncols);
        for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 250.0f;

        const uint64_t w_sz = W.size(), w_pad = (w_sz + 3) & ~3ull;
        const uint64_t x_sz = size_t(s.n_in) / 32 * s.ncols * 36;
        const uint64_t y_sz = size_t(s.ncols) * s.n_out * 4;
        const uint64_t ax = a.alloc(x.size() * 4), aq = a.alloc(x_sz);
        const uint64_t aw = a.alloc(w_pad + 4), ay = a.alloc(y_sz);
        vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
        vk::copy_h2d((void*)aw, W.data(), w_sz, nullptr);
        vk::fill_d((void*)ay, 0xFE, y_sz, nullptr);
        kernels::native_quantize_q8_1((const float*)strata_vk_compat::wrap(ax), strata_vk_compat::wrap(aq),
                                      s.n_in, s.ncols, strata_vk_compat::wrap(1000));
        std::vector<uint8_t> Q(x_sz);
        vk::copy_d2h(Q.data(), (void*)aq, x_sz, nullptr);

        kernels::native_mmvq(13, strata_vk_compat::wrap(aw), strata_vk_compat::wrap(aq),
                             (float*)strata_vk_compat::wrap(ay), s.n_in, s.n_out, s.ncols,
                             strata_vk_compat::wrap(1000));
        std::vector<float> y(size_t(s.ncols) * s.n_out);
        vk::copy_d2h(y.data(), (void*)ay, y_sz, nullptr);

        int shape_fails = 0;
        for (int c = 0; c < s.ncols; ++c) {
            const uint8_t* xcol = Q.data() + size_t(c) * nb * 8 * 36;
            if (rows == 1) {
                for (int r = 0; r < s.n_out; ++r) {
                    const float ref = q5_k_ref(W.data(), xcol, nb, r, 1, 0);
                    const float got = y[size_t(c) * s.n_out + r];
                    ++compared;
                    if (got != ref) {
                        if (shape_fails < 8) {
                            uint32_t gb, rb; std::memcpy(&gb, &got, 4); std::memcpy(&rb, &ref, 4);
                            std::printf("  MISMATCH q5k n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                        s.n_in, c, r, got, ref, gb, rb);
                        }
                        ++shape_fails; ++fails;
                    }
                }
            } else {
                for (int g = 0; g < (s.n_out + 3) / 4; ++g)
                    for (int i = 0; i < 4 && g * 4 + i < s.n_out; ++i) {
                        const int r = g * 4 + i;
                        const float ref = q5_k_ref(W.data(), xcol, nb, g * 4, 4, i);
                        const float got = y[size_t(c) * s.n_out + r];
                        ++compared;
                        if (got != ref) {
                            if (shape_fails < 8) {
                                uint32_t gb, rb; std::memcpy(&gb, &got, 4); std::memcpy(&rb, &ref, 4);
                                std::printf("  MISMATCH q5k n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                            s.n_in, c, r, got, ref, gb, rb);
                            }
                            ++shape_fails; ++fails;
                        }
                    }
            }
        }
        std::printf("native_mmvq q5_k vk vs reference: n_in %d n_out %d ncols %d rows %d: %d outputs, %d differ bitwise\n",
                    s.n_in, s.n_out, s.ncols, rows, s.ncols * s.n_out, shape_fails);
    }
    // ---- type 14 (Q6_K): n_in 2048 -> rows=1 (nb 8, SmallK=false); 512/256 -> rows=4 (nb<4)
    const struct { int n_in, n_out, ncols; } q6_shapes[] = {
        {2048, 64, 1}, {2048, 64, 4}, {2048, 3, 2}, {512, 64, 3}, {256, 3, 1}, {256, 7, 2}};
    for (const auto& s : q6_shapes) {
        const int nb = s.n_in / 256;
        const int rows = (nb < 4) ? 4 : 1;            // mirrors the launcher's SmallK pick
        std::mt19937 rng(4100 + s.n_in * 13 + s.ncols);
        std::uniform_real_distribution<float> dsc(0.0009765625f, 0.03125f);   // d in [2^-10, 2^-5)
        std::vector<uint8_t> W(size_t(s.n_out) * nb * 210);
        for (auto& blk : W) blk = uint8_t(rng() & 0xFF);
        for (int r = 0; r < s.n_out; ++r)
            for (int b = 0; b < nb; ++b) {           // sane fp16 d (byte 208; mmvq_multi_parity's law)
                const uint16_t dh = f16_rn((rng() & 1) ? -dsc(rng) : dsc(rng));
                std::memcpy(W.data() + (size_t(r) * nb + b) * 210 + 208, &dh, 2);
            }
        std::vector<float> x(size_t(s.n_in) * s.ncols);
        for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 250.0f;

        const uint64_t w_sz = W.size(), w_pad = (w_sz + 3) & ~3ull;
        const uint64_t x_sz = size_t(s.n_in) / 32 * s.ncols * 36;
        const uint64_t y_sz = size_t(s.ncols) * s.n_out * 4;
        const uint64_t ax = a.alloc(x.size() * 4), aq = a.alloc(x_sz);
        const uint64_t aw = a.alloc(w_pad + 4), ay = a.alloc(y_sz);
        vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
        vk::copy_h2d((void*)aw, W.data(), w_sz, nullptr);
        vk::fill_d((void*)ay, 0xFE, y_sz, nullptr);
        kernels::native_quantize_q8_1((const float*)strata_vk_compat::wrap(ax), strata_vk_compat::wrap(aq),
                                      s.n_in, s.ncols, strata_vk_compat::wrap(1000));
        std::vector<uint8_t> Q(x_sz);
        vk::copy_d2h(Q.data(), (void*)aq, x_sz, nullptr);

        kernels::native_mmvq(14, strata_vk_compat::wrap(aw), strata_vk_compat::wrap(aq),
                             (float*)strata_vk_compat::wrap(ay), s.n_in, s.n_out, s.ncols,
                             strata_vk_compat::wrap(1000));
        std::vector<float> y(size_t(s.ncols) * s.n_out);
        vk::copy_d2h(y.data(), (void*)ay, y_sz, nullptr);

        int shape_fails = 0;
        for (int c = 0; c < s.ncols; ++c) {
            const uint8_t* xcol = Q.data() + size_t(c) * nb * 8 * 36;
            if (rows == 1) {
                for (int r = 0; r < s.n_out; ++r) {
                    const float ref = q6_k_ref(W.data(), xcol, nb, r, 1, 0);
                    const float got = y[size_t(c) * s.n_out + r];
                    ++compared;
                    if (got != ref) {
                        if (shape_fails < 8) {
                            uint32_t gb, rb; std::memcpy(&gb, &got, 4); std::memcpy(&rb, &ref, 4);
                            std::printf("  MISMATCH q6k n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                        s.n_in, c, r, got, ref, gb, rb);
                        }
                        ++shape_fails; ++fails;
                    }
                }
            } else {
                for (int g = 0; g < (s.n_out + 3) / 4; ++g)
                    for (int i = 0; i < 4 && g * 4 + i < s.n_out; ++i) {
                        const int r = g * 4 + i;
                        const float ref = q6_k_ref(W.data(), xcol, nb, g * 4, 4, i);
                        const float got = y[size_t(c) * s.n_out + r];
                        ++compared;
                        if (got != ref) {
                            if (shape_fails < 8) {
                                uint32_t gb, rb; std::memcpy(&gb, &got, 4); std::memcpy(&rb, &ref, 4);
                                std::printf("  MISMATCH q6k n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                            s.n_in, c, r, got, ref, gb, rb);
                            }
                            ++shape_fails; ++fails;
                        }
                    }
            }
        }
        std::printf("native_mmvq q6_k vk vs reference: n_in %d n_out %d ncols %d rows %d: %d outputs, %d differ bitwise\n",
                    s.n_in, s.n_out, s.ncols, rows, s.ncols * s.n_out, shape_fails);
    }
    std::printf("native_mmvq_vk_parity: %d outputs compared bitwise, %d differ -> %s\n",
                compared, fails, fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
