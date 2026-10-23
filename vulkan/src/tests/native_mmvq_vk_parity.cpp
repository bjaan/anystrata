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
#include "iq3xxs_grid_vk.h"
#include "iq2s_grid_vk.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <bit>
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

// vec_dot_iq3_xxs_q8_1 (iq_kernels.cu:191-214) transcribed verbatim; bq3 = block bytes (98 B: half d
// at 0, 64 grid-index bytes at 2..65, aux words at 66..97 - get_int_b2(qs, 16 + iqs/2) = 4 bytes at
// 2 + 4*(16 + iqs/2) = 66 + 2*iqs), q8 = the q8_1 block's byte 0 (word 0 = ds, bytes 4..35 = qs).
float vdot_iq3_xxs(const uint8_t* bq3, const uint8_t* q8, int iqs) {
    uint16_t dh; std::memcpy(&dh, bq3, 2);
    const float dw = f16_to_f32(dh);
    uint32_t q3[2];
    std::memcpy(q3,    bq3 + 2 + 4 * iqs,       8);   // make_int2(get_int_b2(qs,iqs), get_int_b2(qs,iqs+1))
    const uint8_t* qb = (const uint8_t*) q3;          // q3[] indexes bytes l0, l0+1 of the pair
    uint32_t aux32;
    std::memcpy(&aux32, bq3 + 66 + 2 * iqs, 4);       // get_int_b2(qs, QK_K/16 + iqs/2)
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t gx = strata::vk::kIq3xxsGrid[qb[l0 + 0]];       // word itself = four int8 lanes
        const uint32_t gy = strata::vk::kIq3xxsGrid[qb[l0 + 1]];
        const uint32_t signs = unpack_ksigns((aux32 >> (7 * (l0 / 2))) & 0xFFu);
        const uint32_t s0 = vcmpne4(signs & 0x08040201u);
        uint32_t u0; std::memcpy(&u0, q8 + 4 + l0 * 4, 4);             // get_int_b4(bq8->qs, l0)
        sumi = dp4a(vsub4(gx ^ s0, s0), u0, sumi);
        const uint32_t s1 = vcmpne4(signs & 0x80402010u);
        uint32_t u1; std::memcpy(&u1, q8 + 4 + (l0 + 1) * 4, 4);
        sumi = dp4a(vsub4(gy ^ s1, s1), u1, sumi);
    }
    const int ls = int(aux32 >> 28);                                   // logical shift; 4-bit value
    sumi = (ls * sumi + sumi / 2) / 2;                                 // C truncation toward zero, both divs
    uint16_t d8h; std::memcpy(&d8h, q8, 2);                            // __low2float(ds)
    return dw * f16_to_f32(d8h) * float(sumi);
}

// vec_dot_iq2_s_q8_1 (iq_kernels.cu:158-189, vecdotq.cuh:1124-1167) transcribed verbatim;
// bq2 = block bytes (82 B: half d @0, qs[64] @2..65 -- grid-index bytes at 2+4*(iqs/2)+au and
// sign bytes at 34+2*iqs+au == "get_int_b2(qs, 8 + iqs/2)", qh[8] @66, scales[8] @74), q8 = the
// q8_1 block's byte 0 (word 0 = ds, bytes 4..35 = qs).
float vdot_iq2_s(const uint8_t* bq2, const uint8_t* q8, int iqs) {
    uint16_t dh; std::memcpy(&dh, bq2, 2);
    const float dw = f16_to_f32(dh);
    const uint8_t* qs = bq2 + 2 + 4 * (iqs / 2);        // qs_packed's 4 bytes
    const uint8_t* sp = bq2 + 2 + 4 * (8 + iqs / 2);    // signs_packed_32's 4 bytes = 34 + 2*iqs
    const int qh  = bq2[66 + iqs / 2];
    const uint8_t scale = bq2[74 + iqs / 2];
    const int ls0 = scale & 0x0F, ls1 = scale >> 4;
    int sumi0 = 0, sumi1 = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint64_t g = strata::vk::kIq2sGrid[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const uint32_t gx = uint32_t(g), gy = uint32_t(g >> 32);      // (const int*) entry halves
        const uint32_t s0 = vcmpne4(((sp[l0 / 2] & 0x03) << 7) | ((sp[l0 / 2] & 0x0C) << 21));
        const uint32_t s1 = vcmpne4(((sp[l0 / 2] & 0x30) << 3) | ((sp[l0 / 2] & 0xC0) << 17));
        uint32_t u0; std::memcpy(&u0, q8 + 4 + l0 * 4, 4);            // get_int_b4(bq8->qs, l0)
        uint32_t u1; std::memcpy(&u1, q8 + 4 + (l0 + 1) * 4, 4);
        if (l0 < 4) {
            sumi0 = dp4a(vsub4(gx ^ s0, s0), u0, sumi0);
            sumi0 = dp4a(vsub4(gy ^ s1, s1), u1, sumi0);
        } else {
            sumi1 = dp4a(vsub4(gx ^ s0, s0), u0, sumi1);
            sumi1 = dp4a(vsub4(gy ^ s1, s1), u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;   // C truncation, both divs
    uint16_t d8h; std::memcpy(&d8h, q8, 2);                                   // __low2float(ds)
    return dw * f16_to_f32(d8h) * float(sumi);
}

// ---- IQ4_XS (type 23, Task 2.12): kvalues_iq4nl (ggml-common.h:1120-1122) as a host table, and
// vec_dot_iq4_xs_q8_1 (iq_kernels.cu:296-313) transcribed verbatim. block_iq4_xs is 146 B (half d
// @0, uint16 scales_h @2, uint8 scales_l[4] @4, uint8 qs[128] @8); q8 = the q8_1 block's byte 0
// (word 0 = ds, bytes 4..35 = qs). No grid table — the nibbles decode through kvalues_iq4nl.
const int8_t kIq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

float vdot_iq4_xs(const uint8_t* bq4, const uint8_t* q8, int iqs) {
    uint16_t dh; std::memcpy(&dh, bq4, 2);
    const float dw = f16_to_f32(dh);
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t aux_q4; std::memcpy(&aux_q4, bq4 + 8 + 4 * (iqs + j), 4);   // get_int_b4(qs, iqs+j)
        // get_int_from_table_16: v.x = even nibbles, v.y = odd nibbles (packed as bytes)
        uint32_t vx = 0, vy = 0;
        for (int b = 0; b < 4; ++b) {
            vx |= uint32_t(uint8_t(kIq4nl[(aux_q4 >> (8 * b)) & 15u])) << (8 * b);
            vy |= uint32_t(uint8_t(kIq4nl[(aux_q4 >> (4 + 8 * b)) & 15u])) << (8 * b);
        }
        uint32_t u0; std::memcpy(&u0, q8 + 4 + 4 * j, 4);        // get_int_b4(bq8_1[iqs/4].qs, j)
        uint32_t u1; std::memcpy(&u1, q8 + 4 + 4 * (j + 4), 4);  // get_int_b4(..., j + 4)
        sumi = dp4a(vx, u0, sumi);
        sumi = dp4a(vy, u1, sumi);
    }
    uint16_t sh16; std::memcpy(&sh16, bq4 + 2, 2);
    const int ls = int(((bq4[4 + iqs / 8] >> (iqs & 0x04)) & 0x0F) |
                       ((uint32_t(sh16) >> (iqs / 2)) & 0x03u) << 4);
    sumi *= ls - 32;
    uint16_t d8h; std::memcpy(&d8h, q8, 2);                                // __low2float(ds)
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

// ---- Q8_0 (type 8, Task 2.13): transcription of small_q8_dot(Q80Block) (native_mmvq.cu:888-900).
// Q80Block bytes: half d @0, int8 qs[32] @2 (34 B). v[i] = load_int_b2(qs, iqs+i) = the two u16
// pair (exact as a u32 read); u[i] = (int*)(x->qs)[iqs+i]; sumi = the two-dp4a chain (exact int);
// d0*d1 is exact in fp32 (11+11-bit mantissas). Returns dm and sumi: the kbx accumulation
// `tmp += dm * float(sumi)` answers UNFUSED on this device (contraction probe, 2026-10-23: the
// fma(dm, sumi, acc) host arm differed on 36-3649/rows=1 outputs; the unfused arm below matches
// BITWISE on all 7879 q8_0 outputs — glslc's OpFMul+OpFAdd survives NIR without contracting here).
void q8_0_dot_df(const uint8_t* w, const uint8_t* x, int iqs, float& dm, int& sumi) {  // x = one q8_1 block
    sumi = 0;
    for (int i = 0; i < 2; ++i)
        sumi = dp4a(uint32_t(rd_i32(w + 2 + 4 * (iqs + i))),
                    uint32_t(rd_i32(x + 4 + 4 * (iqs + i))), sumi);
    dm = f16_to_f32(rd_u16(w)) * f16_to_f32(rd_u16(x));
}

// emulate native_small_mmvq_kernel<Q80Block, 8, SmallK> for one (row, column): tid = 0..127,
// kbx = tid/4 + 32t (Qi=8 -> T=4, BLOCKS_PER_ITER=32), kqs = 2*(tid%4); x block = kbx (KBY=1).
// rows==1: all four subrows sum one row; rows==4: subrow i does row0+i. warp 0 sums
// subrow-ascending then the xor tree; lane i writes row0+i.
float q8_0_ref(const uint8_t* wr, const uint8_t* xcol, int nb, int row, int rows, int row_in_group) {
    float part[4][32];
    for (int s = 0; s < 4; ++s) for (int l = 0; l < 32; ++l) part[s][l] = 0.f;
    for (int tid = 0; tid < 128; ++tid) {
        const int s = tid / 32, lane = tid % 32, kqs = 2 * (tid % 4);
        const int r = (rows == 1) ? row : row + row_in_group;
        float acc = 0.f;
        for (int kbx = tid / 4; kbx < nb; kbx += 32) {
            float dm; int sumi;
            q8_0_dot_df(wr + (size_t(r) * nb + kbx) * 34, xcol + size_t(kbx) * 36, kqs, dm, sumi);
            // probe: UNFUSED accumulation (glslc kept OpFMul+OpFAdd for this dot shape)
            const float term = dm * float(sumi);
            acc = acc + term;
        }
        part[s][lane] = acc;
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
    // ---- type 18 (IQ3_XXS, Task 2.9): Fmt<18> = {qk 256, ipb 8, step 2}, ROWS=4 grid ceil(n_out/4)
    // always (mmvq_kernel<18> default, iq_kernels.cu:3126 iq_mmvq -> the Split/multi kernel pinned
    // bitwise-equal per column). Blocks are 98 B.
    const Shape iq3_shapes[] = {{2048, 64, 1}, {2048, 64, 2}, {2048, 768, 1}, {768, 64, 3}, {256, 3, 1}};
    for (const Shape& s : iq3_shapes) {
        const int nb = s.n_in / 256;
        std::mt19937 rng(4200 + s.n_in * 13 + s.ncols);
        std::uniform_real_distribution<float> sc(0.0009765625f, 0.03125f);   // d in [2^-10, 2^-5)
        std::vector<uint8_t> W(size_t(s.n_out) * nb * 98);
        for (auto& blk : W) blk = uint8_t(rng() & 0xFF);                     // random legal bytes
        for (int r = 0; r < s.n_out; ++r)
            for (int b = 0; b < nb; ++b) {
                const uint16_t dh = f16_rn((rng() & 1) ? -sc(rng) : sc(rng));
                std::memcpy(W.data() + (size_t(r) * nb + b) * 98, &dh, 2);   // finite fp16 scale
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

        kernels::native_mmvq(18, strata_vk_compat::wrap(aw), strata_vk_compat::wrap(aq),
                             (float*)strata_vk_compat::wrap(ay), s.n_in, s.n_out, s.ncols,
                             strata_vk_compat::wrap(1000));
        std::vector<float> y(size_t(s.ncols) * s.n_out);
        vk::copy_d2h(y.data(), (void*)ay, y_sz, nullptr);

        int shape_fails = 0;
        for (int c = 0; c < s.ncols; ++c)
            for (int r = 0; r < s.n_out; ++r) {
                const uint8_t* wr = W.data() + size_t(r) * nb * 98;
                const uint8_t* xcol = Q.data() + size_t(c) * nb * 8 * 36;
                float part[32] = {0.f};
                for (int k = 0; k < nb * 8; ++k) {                          // lane k%32's call k
                    const int kbx = k / 8, iqs = 2 * (k % 8);
                    part[k % 32] += vdot_iq3_xxs(wr + kbx * 98, xcol + size_t(kbx * 8 + iqs / 2) * 36, iqs);
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
                        std::printf("  MISMATCH t18 n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                    s.n_in, c, r, got, ref, gb, rb);
                    ++shape_fails; ++fails;
                }
            }
        std::printf("native_mmvq iq3_xxs vk vs reference: n_in %d n_out %d ncols %d: %d outputs, %d differ bitwise\n",
                    s.n_in, s.n_out, s.ncols, s.ncols * s.n_out, shape_fails);
    }
    // ---- type 23 (IQ4_XS, Task 2.12): Fmt<23> = {qk 256, ipb 8, step 4} - same ROWS=4 grid
    // ceil(n_out/4) launch as 16/18; blocks are 146 B, no grid table (kvalues_iq4nl inlined).
    const Shape iq4_shapes[] = {{2048, 64, 1}, {2048, 64, 2}, {2048, 768, 1}, {768, 64, 3}, {256, 3, 1}};
    for (const Shape& s : iq4_shapes) {
        const int nb = s.n_in / 256;
        std::mt19937 rng(5300 + s.n_in * 13 + s.ncols);
        std::uniform_real_distribution<float> sc(0.0009765625f, 0.03125f);   // d in [2^-10, 2^-5)
        std::vector<uint8_t> W(size_t(s.n_out) * nb * 146);
        for (auto& blk : W) blk = uint8_t(rng() & 0xFF);                     // random legal bytes
        for (int r = 0; r < s.n_out; ++r)
            for (int b = 0; b < nb; ++b) {
                const uint16_t dh = f16_rn((rng() & 1) ? -sc(rng) : sc(rng));
                std::memcpy(W.data() + (size_t(r) * nb + b) * 146, &dh, 2);  // finite fp16 scale
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

        kernels::native_mmvq(23, strata_vk_compat::wrap(aw), strata_vk_compat::wrap(aq),
                             (float*)strata_vk_compat::wrap(ay), s.n_in, s.n_out, s.ncols,
                             strata_vk_compat::wrap(1000));
        std::vector<float> y(size_t(s.ncols) * s.n_out);
        vk::copy_d2h(y.data(), (void*)ay, y_sz, nullptr);

        int shape_fails = 0;
        for (int c = 0; c < s.ncols; ++c)
            for (int r = 0; r < s.n_out; ++r) {
                const uint8_t* wr = W.data() + size_t(r) * nb * 146;
                const uint8_t* xcol = Q.data() + size_t(c) * nb * 8 * 36;
                float part[32] = {0.f};
                for (int k = 0; k < nb * 8; ++k) {                          // lane k%32's call k
                    const int kbx = k / 8, iqs = 4 * (k % 8);               // Fmt<23>: step 4
                    part[k % 32] += vdot_iq4_xs(wr + kbx * 146, xcol + size_t(kbx * 8 + iqs / 4) * 36, iqs);
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
                        std::printf("  MISMATCH t23 n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                    s.n_in, c, r, got, ref, gb, rb);
                    ++shape_fails; ++fails;
                }
            }
        std::printf("native_mmvq iq4_xs vk vs reference: n_in %d n_out %d ncols %d: %d outputs, %d differ bitwise\n",
                    s.n_in, s.n_out, s.ncols, s.ncols * s.n_out, shape_fails);
    }
    // ---- type 22 (IQ2_S, Task 2.14): Fmt<22> = {qk 256, ipb 8, step 2} - same ROWS=4 grid
    // ceil(n_out/4) launch as 16/18; blocks are 82 B (half d + qs[64] + qh[8] + scales[8]), the
    // uint64 iq2s_grid table (1024 entries) rides the arena-upload route bound as uvec2 pairs.
    const Shape iq2s_shapes[] = {{2048, 64, 1}, {2048, 768, 1}, {2048, 64, 2}, {768, 64, 3}, {256, 64, 1}, {256, 3, 1}};
    for (const Shape& s : iq2s_shapes) {
        const int nb = s.n_in / 256;
        std::mt19937 rng(6100 + s.n_in * 13 + s.ncols);
        std::uniform_real_distribution<float> sc(0.0009765625f, 0.03125f);   // d in [2^-10, 2^-5)
        std::vector<uint8_t> W(size_t(s.n_out) * nb * 82);
        for (auto& blk : W) blk = uint8_t(rng() & 0xFF);                     // random legal bytes
        for (int r = 0; r < s.n_out; ++r)
            for (int b = 0; b < nb; ++b) {
                const uint16_t dh = f16_rn((rng() & 1) ? -sc(rng) : sc(rng));
                std::memcpy(W.data() + (size_t(r) * nb + b) * 82, &dh, 2);   // finite fp16 scale
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

        kernels::native_mmvq(22, strata_vk_compat::wrap(aw), strata_vk_compat::wrap(aq),
                             (float*)strata_vk_compat::wrap(ay), s.n_in, s.n_out, s.ncols,
                             strata_vk_compat::wrap(1000));
        std::vector<float> y(size_t(s.ncols) * s.n_out);
        vk::copy_d2h(y.data(), (void*)ay, y_sz, nullptr);

        int shape_fails = 0;
        for (int c = 0; c < s.ncols; ++c)
            for (int r = 0; r < s.n_out; ++r) {
                const uint8_t* wr = W.data() + size_t(r) * nb * 82;
                const uint8_t* xcol = Q.data() + size_t(c) * nb * 8 * 36;
                float part[32] = {0.f};
                for (int k = 0; k < nb * 8; ++k) {                          // lane k%32's call k
                    const int kbx = k / 8, iqs = 2 * (k % 8);               // Fmt<22>: step 2
                    part[k % 32] += vdot_iq2_s(wr + kbx * 82, xcol + size_t(kbx * 8 + iqs / 2) * 36, iqs);
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
                        std::printf("  MISMATCH t22 n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                    s.n_in, c, r, got, ref, gb, rb);
                    ++shape_fails; ++fails;
                }
            }
        std::printf("native_mmvq iq2_s vk vs reference: n_in %d n_out %d ncols %d: %d outputs, %d differ bitwise\n",
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
    // ---- Task 2.8: native_mmvq_slots. Per the CUDA contract (native_mmvq.hpp:140-149, kernel
    // :1710-1755 / iq mmvq_slots_kernel :3491-3501): matrix i is w[i] times x[i] into y[i], one
    // column each, and per-slot work is the SAME partition as the single-launch kernel — so
    // y_slots == y_single BITWISE is the strongest check; we additionally assert the host refs.
    // ---- type 8 (Q8_0, Task 2.13): n_in 2048 -> nb 64 >= 32 -> rows=1 (SmallK=false);
    // 768/256 -> nb 24/8 < 32 -> rows=4 (SmallK=true). x side is always q8_1 (36 B), W is Q8_0 (34 B).
    const struct { int n_in, n_out, ncols; } q8_shapes[] = {
        {2048, 64, 1}, {2048, 768, 2}, {2048, 2048, 3}, {768, 64, 2}, {256, 7, 1}};
    for (const auto& s : q8_shapes) {
        const int nb = s.n_in / 32;
        const int rows = (nb < 32) ? 4 : 1;           // mirrors the launcher's SmallK pick
        std::mt19937 rng(7700 + s.n_in * 5 + s.ncols);
        std::uniform_real_distribution<float> dsc(0.0009765625f, 0.03125f);   // d in [2^-10, 2^-5)
        std::vector<uint8_t> W(size_t(s.n_out) * nb * 34);
        for (auto& blk : W) blk = uint8_t(rng() & 0xFF);                      // random int8 qs
        for (int r = 0; r < s.n_out; ++r)
            for (int b = 0; b < nb; ++b) {            // sane fp16 d (bytes 0-1)
                const uint16_t dh = f16_rn((rng() & 1) ? -dsc(rng) : dsc(rng));
                std::memcpy(W.data() + (size_t(r) * nb + b) * 34, &dh, 2);
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

        kernels::native_mmvq(8, strata_vk_compat::wrap(aw), strata_vk_compat::wrap(aq),
                             (float*)strata_vk_compat::wrap(ay), s.n_in, s.n_out, s.ncols,
                             strata_vk_compat::wrap(1000));
        std::vector<float> y(size_t(s.ncols) * s.n_out);
        vk::copy_d2h(y.data(), (void*)ay, y_sz, nullptr);

        int shape_fails = 0;
        for (int c = 0; c < s.ncols; ++c) {
            const uint8_t* xcol = Q.data() + size_t(c) * nb * 36;
            if (rows == 1) {
                for (int r = 0; r < s.n_out; ++r) {
                    const float ref = q8_0_ref(W.data(), xcol, nb, r, 1, 0);
                    const float got = y[size_t(c) * s.n_out + r];
                    ++compared;
                    if (got != ref) {
                        if (shape_fails < 8) {
                            uint32_t gb, rb; std::memcpy(&gb, &got, 4); std::memcpy(&rb, &ref, 4);
                            std::printf("  MISMATCH q80 n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                        s.n_in, c, r, got, ref, gb, rb);
                        }
                        ++shape_fails; ++fails;
                    }
                }
            } else {
                for (int g = 0; g < (s.n_out + 3) / 4; ++g)
                    for (int i = 0; i < 4 && g * 4 + i < s.n_out; ++i) {
                        const int r = g * 4 + i;
                        const float ref = q8_0_ref(W.data(), xcol, nb, g * 4, 4, i);
                        const float got = y[size_t(c) * s.n_out + r];
                        ++compared;
                        if (got != ref) {
                            if (shape_fails < 8) {
                                uint32_t gb, rb; std::memcpy(&gb, &got, 4); std::memcpy(&rb, &ref, 4);
                                std::printf("  MISMATCH q80 n_in %d ncols %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                            s.n_in, c, r, got, ref, gb, rb);
                            }
                            ++shape_fails; ++fails;
                        }
                    }
            }
        }
        std::printf("native_mmvq q8_0 vk vs reference: n_in %d n_out %d ncols %d rows %d: %d outputs, %d differ bitwise\n",
                    s.n_in, s.n_out, s.ncols, rows, s.ncols * s.n_out, shape_fails);
    }
    // Mixed slots across types: iq2_xxs expert dims (2048 x 768), q5_k (2048 x 2048 style), q6_k.
    const int slots_cases = 0; (void)slots_cases;
    {
        struct SlotCase { int type, count, n_in, n_out; int rb; };   // rb = row block bytes
        // 2048: ROWS=1 arm for 13/14; 512: the SmallK ROWS=2 arm (nb 2 < 8/4) — the host refs
        // model the single-call ROWS=4 partition, provably the same per-row lane sums (see above).
        const SlotCase cases[] = { {16, 3, 2048, 768, 66}, {13, 2, 2048, 512, 176}, {14, 2, 2048, 2048, 210},
                                   {14, 3, 512, 64, 210}, {13, 2, 512, 7, 176} };
        for (const auto& sc : cases) {
            const int nb = sc.n_in / 256;
            std::mt19937 rng(7700 + sc.type * 101 + sc.count);
            std::vector<std::vector<uint8_t>> W(sc.count);
            for (auto& w : W) {
                w.resize(size_t(sc.n_out) * nb * sc.rb);
                for (auto& blk : w) blk = uint8_t(rng() & 0xFF);
            }
            if (sc.type == 16) for (auto& w : W) for (int r = 0; r < sc.n_out; ++r) for (int b = 0; b < nb; ++b) {
                const uint16_t dh = f16_rn((rng() & 1) ? -0.01f : 0.02f);
                std::memcpy(w.data() + (size_t(r) * nb + b) * 66, &dh, 2);
            } else if (sc.type == 13) for (auto& w : W) for (int r = 0; r < sc.n_out; ++r) for (int b = 0; b < nb; ++b) {
                const uint16_t dh = f16_rn((rng() & 1) ? -0.01f : 0.02f), dl = f16_rn((rng() & 1) ? -0.02f : 0.01f);
                std::memcpy(w.data() + (size_t(r) * nb + b) * 176, &dh, 2);
                std::memcpy(w.data() + (size_t(r) * nb + b) * 176 + 2, &dl, 2);
            } else for (auto& w : W) for (int r = 0; r < sc.n_out; ++r) for (int b = 0; b < nb; ++b) {
                const uint16_t dh = f16_rn((rng() & 1) ? -0.01f : 0.02f);
                std::memcpy(w.data() + (size_t(r) * nb + b) * 210 + 208, &dh, 2);
            }
            // one shared activation column (the engine repeats x[i]): quantize once
            std::vector<float> x(sc.n_in);
            for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 250.0f;
            const uint64_t x_sz = size_t(nb) * 8 * 36;
            const uint64_t ax = a.alloc(x.size() * 4), aq = a.alloc(x_sz);
            vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
            kernels::native_quantize_q8_1((const float*)strata_vk_compat::wrap(ax), strata_vk_compat::wrap(aq),
                                          sc.n_in, 1, strata_vk_compat::wrap(1000));
            std::vector<uint8_t> Q(x_sz);
            vk::copy_d2h(Q.data(), (void*)aq, x_sz, nullptr);

            kernels::MmvqSlots slots{};
            std::vector<uint64_t> aw(sc.count), ay(sc.count);
            const uint64_t y_sz = size_t(sc.n_out) * 4;
            for (int i = 0; i < sc.count; ++i) {
                const uint64_t w_pad = (W[i].size() + 3) & ~3ull;
                aw[i] = a.alloc(w_pad + 4);
                ay[i] = a.alloc(y_sz);
                vk::copy_h2d((void*)aw[i], W[i].data(), W[i].size(), nullptr);
                vk::fill_d((void*)ay[i], 0xFE, y_sz, nullptr);
                slots.w[i] = strata_vk_compat::wrap(aw[i]);
                slots.x[i] = strata_vk_compat::wrap(aq);      // the same column in every slot
                slots.y[i] = (float*)strata_vk_compat::wrap(ay[i]);
            }
            // single-launch reference outputs (the base kernel for the same type/shape)
            std::vector<std::vector<float>> y_single(sc.count);
            for (int i = 0; i < sc.count; ++i) {
                kernels::native_mmvq(sc.type, slots.w[i], slots.x[i], slots.y[i], sc.n_in, sc.n_out, 1,
                                     strata_vk_compat::wrap(1000));
                y_single[i].resize(sc.n_out);
                vk::copy_d2h(y_single[i].data(), (void*)ay[i], y_sz, nullptr);
                vk::fill_d((void*)ay[i], 0xFE, y_sz, nullptr);   // re-poison for the slots run
            }
            if (!kernels::native_mmvq_slots(sc.type, slots, sc.count, sc.n_in, sc.n_out,
                                            strata_vk_compat::wrap(1000))) {
                std::printf("slots: native_mmvq_slots(%d) returned FALSE — FAIL\n", sc.type);
                ++fails;
                continue;
            }
            int shape_fails = 0;
            for (int i = 0; i < sc.count; ++i) {
                std::vector<float> y(sc.n_out);
                vk::copy_d2h(y.data(), (void*)ay[i], y_sz, nullptr);
                for (int r = 0; r < sc.n_out; ++r) {
                    // (a) bitwise the single-launch kernel's answer
                    if (y[r] != y_single[i][r]) {
                        if (shape_fails < 8) {
                            uint32_t gb, sb; std::memcpy(&gb, &y[r], 4); std::memcpy(&sb, &y_single[i][r], 4);
                            std::printf("  MISMATCH slots t %d slot %d r %d: slots %.9g single %.9g (bits %08x/%08x)\n",
                                        sc.type, i, r, y[r], y_single[i][r], gb, sb);
                        }
                        ++shape_fails; ++fails;
                    }
                    // (b) bitwise the host transcription of the same partition
                    float ref;
                    if (sc.type == 16) {
                        const uint8_t* wr = W[i].data() + size_t(r) * nb * 66;
                        float part[32] = {0.f};
                        for (int k = 0; k < nb * 8; ++k)
                            part[k % 32] += vdot_iq2_xxs(wr + (k / 8) * 66, Q.data() + size_t(k) * 36, 2 * (k % 8));
                        float acc[32], nxt[32]; std::memcpy(acc, part, sizeof acc);
                        for (int o = 16; o > 0; o >>= 1) { for (int l = 0; l < 32; ++l) nxt[l] = acc[l] + acc[l ^ o]; std::memcpy(acc, nxt, sizeof nxt); }
                        ref = acc[0];
                    } else if (sc.type == 13) {
                        const int rows = (nb < 8) ? 4 : 1;
                        ref = q5_k_ref(W[i].data(), Q.data(), nb, rows == 1 ? r : (r / 4) * 4, rows, r % 4);
                    } else {
                        const int rows = (nb < 4) ? 4 : 1;
                        ref = q6_k_ref(W[i].data(), Q.data(), nb, rows == 1 ? r : (r / 4) * 4, rows, r % 4);
                    }
                    ++compared;
                    if (y[r] != ref) {
                        if (shape_fails < 8) {
                            uint32_t gb, rb2; std::memcpy(&gb, &y[r], 4); std::memcpy(&rb2, &ref, 4);
                            std::printf("  MISMATCH slots-ref t %d slot %d r %d: got %.9g ref %.9g (bits %08x/%08x)\n",
                                        sc.type, i, r, y[r], ref, gb, rb2);
                        }
                        ++shape_fails; ++fails;
                    }
                }
            }
            std::printf("native_mmvq_slots t %d count %d n_in %d n_out %d: %d outputs, %d differ bitwise\n",
                        sc.type, sc.count, sc.n_in, sc.n_out, sc.count * sc.n_out, shape_fails);
        }
        // ---- an unported format: false, with NOTHING launched (CUDA wrapper :3179's slots_kind
        // leg) — y stays poisoned. Type 12 (Q4_K) is ported nowhere here (single-matrix traps).
        {
            kernels::MmvqSlots slots{};
            const uint64_t aw0 = a.alloc(256), ax0 = a.alloc(256), ay = a.alloc(64 * 4);
            vk::fill_d((void*)ay, 0xFE, 64 * 4, nullptr);
            slots.w[0] = strata_vk_compat::wrap(aw0);
            slots.x[0] = strata_vk_compat::wrap(ax0);
            slots.y[0] = (float*)strata_vk_compat::wrap(ay);
            if (kernels::native_mmvq_slots(12, slots, 1, 2048, 64, strata_vk_compat::wrap(1000))) {
                std::printf("slots: type 12 returned TRUE — FAIL\n"); ++fails;
            }
            std::vector<float> y(64);
            vk::copy_d2h(y.data(), (void*)ay, 64 * 4, nullptr);
            for (int r = 0; r < 64; ++r)
                if (std::bit_cast<uint32_t>(y[r]) != 0xFEFEFEFEu) { std::printf("slots t12 poisoned y touched — FAIL\n"); ++fails; break; }
            std::printf("native_mmvq_slots unported type 12: false, nothing launched — checked\n");
        }
    }
    std::printf("native_mmvq_vk_parity: %d outputs compared bitwise, %d differ -> %s\n",
                compared, fails, fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
