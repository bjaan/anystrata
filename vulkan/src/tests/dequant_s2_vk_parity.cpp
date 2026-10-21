// vulkan/src/tests/dequant_s2_vk_parity.cpp - the tree's dequant_s2 contract on Vulkan, at zero tolerance.
//
// src/kernels/dequant_s2_parity.cpp's rule applies verbatim: the decode is exact integer-to-float
// arithmetic (the -1 rides on the code), so this asserts BIT EQUALITY against the scalar reference -
// no tolerance, because none is earned. The scalar reference here re-transcribes docs/pack-format.md's
// contract (upstream's own validated dequantizer, strata::dequantize_s2, is checked against ggml by
// bench/micro/dequant_xcheck, which lives on the CUDA tree).
#include "strata/kernels/dequant_s2.hpp"
#include "strata/vulkan_runtime.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int QK = 64;
}  // namespace

using namespace strata;

int main(int argc, char** argv) {
    bool selftest = false;
    long long n_blocks = 4096;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--n-blocks") == 0 && i + 1 < argc) n_blocks = std::atoll(argv[++i]);
        else { std::fprintf(stderr, "usage: dequant_s2_vk_parity [--selftest] [--n-blocks N]\n"); return 2; }
    }
    (void)selftest;

    std::vector<uint8_t> codes((size_t)n_blocks * (QK / 4));
    std::mt19937 rng(4242);
    for (auto& v : codes) v = (uint8_t)(rng() & 0xFF);
    std::vector<float> scales((size_t)n_blocks);
    for (auto& v : scales) {
        // finite, exact-representable scales: a power of two times a small odd, so the one multiply
        // cannot round in either implementation and any mismatch is a real bug.
        const int e = (int)(rng() % 27) - 13;
        v = (float)(1 + 2 * (rng() % 64)) * std::ldexp(1.0f, e);
    }

    try { vk::device_info(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "dequant_s2_vk_parity: %s\n", e.what());
        std::fprintf(stderr, "dequant_s2_vk_parity SKIP (no usable Vulkan device)\n");
        return 77;
    }

    // host reference: the pack-format contract, code-1 in the INTEGER domain then ONE multiply.
    std::vector<float> ref((size_t)n_blocks * QK);
    for (long long b = 0; b < n_blocks; ++b) {
        const float d = scales[(size_t)b];
        const uint8_t* c = codes.data() + b * (QK / 4);
        for (int j = 0; j < QK; ++j) {
            const int code = (c[j / 4] >> ((j % 4) * 2)) & 0x03;
            ref[(size_t)(b * QK + j)] = (float)(code - 1) * d;
        }
    }

    vk::Arena& a = vk::Arena::instance();
    const uint64_t sc = (uint64_t)n_blocks * (QK / 4), ss = (uint64_t)n_blocks * 4,
                   so = (uint64_t)n_blocks * QK * 4;
    const uint64_t dc = a.alloc(sc), ds = a.alloc(ss), d_o = a.alloc(so);
    vk::copy_h2d((void*)(uintptr_t)dc, codes.data(), sc, nullptr);
    vk::copy_h2d((void*)(uintptr_t)ds, scales.data(), ss, nullptr);
    vk::fill_d((void*)(uintptr_t)d_o, 0xFE, so, nullptr);

    strata::kernels::dequant_s2((const uint8_t*)(uintptr_t)dc, (const float*)(uintptr_t)ds,
                                (float*)(uintptr_t)d_o, n_blocks);

    std::vector<float> got((size_t)n_blocks * QK);
    vk::copy_d2h(got.data(), (void*)(uintptr_t)d_o, so, nullptr);

    long long bad = 0;
    for (size_t i = 0; i < got.size() && bad < 4; ++i)
        if (std::memcmp(&got[i], &ref[i], 4) != 0) {
            std::printf("  elem %zu: ref %.9g got %.9g\n", i, ref[i], got[i]); ++bad;
        }
    for (size_t i = 0; i < got.size(); ++i)
        if (std::memcmp(&got[i], &ref[i], 4) != 0) ++bad;
    std::printf("dequant_s2 vk: %lld blocks (%lld elements), mismatches %lld\n", n_blocks, n_blocks * QK, bad);
    const bool pass = (bad == 0);
    std::printf("dequant_s2_vk_parity %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
