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
    std::printf("native_mmvq_vk_parity: %d outputs compared bitwise, %d differ -> %s\n",
                compared, fails, fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
