// vulkan/src/tests/s2_gemv_vk_parity.cpp - the tree's own s2_gemv parity contract on Vulkan.
//
// Same shape as src/kernels/s2_gemv_parity.cpp: the reference is the validated scalar transcription
// (include/strata/artifact/dequant.hpp's fp16_to_fp32 plus the dot in the same term order as the CUDA
// kernel), the inputs are the parity test's exact patterns and sizes (n_in 2560, n_out 640 - the real
// expert shapes), and the assertion is the same RELATIVE bound with the measured number printed.
// The one deliberate difference: the device side is vk (arena offsets instead of cudaMalloc pointers),
// which is exactly what this parity test exists to pin down.
#include "strata/kernels/s2_gemv.hpp"
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

float fp16_to_fp32(uint16_t h) {           // the validated transcription (artifact/dequant.hpp)
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
    else f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    float out; std::memcpy(&out, &f, 4); return out;
}

// the same 27 patterns as the CUDA-side test: powers of two AND full 11-bit mantissas, so the
// arithmetic CAN round and the tolerance means something.
const uint16_t kScales[] = {
    0x3C00, 0x3800, 0x3400, 0x3000, 0x2C00, 0x2800, 0x2000,
    0xBC00, 0xB800, 0xB400, 0xB000, 0xAC00, 0x1800, 0x1400,
    0x3E00, 0x3555, 0x3C01, 0x4248, 0x4123, 0x2AAA, 0x4A2B,
    0xBE00, 0xB555, 0xC248, 0x2AAB, 0x4A2C, 0x2AAB};
}  // namespace

using namespace strata;

int main(int argc, char** argv) {
    long long n_in = 2560, n_out = 640;
    double tol = 1e-5;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--n-in") == 0 && i + 1 < argc) n_in = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--n-out") == 0 && i + 1 < argc) n_out = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc) tol = std::atof(argv[++i]);
        else { std::fprintf(stderr, "usage: s2_gemv_vk_parity [--selftest] [--n-in N] [--n-out N] [--tol T]\n"); return 2; }
    }
    if (n_in % QK) { std::fprintf(stderr, "--n-in must be a multiple of %d\n", QK); return 2; }
    const long long nb = n_in / QK;
    const int n_scales = (int)(sizeof(kScales) / sizeof(kScales[0]));

    std::mt19937 rng(999);
    std::vector<uint16_t> x((size_t)n_in);
    for (auto& v : x) v = kScales[rng() % n_scales];
    std::vector<uint8_t> codes((size_t)n_out * nb * (QK / 4));
    for (auto& v : codes) v = (uint8_t)(rng() & 0xFF);
    std::vector<float> scales((size_t)n_out * nb);
    for (auto& v : scales) v = fp16_to_fp32(kScales[rng() % n_scales]);

    // host reference: the CUDA kernel's term order, without contraction (the same choice the CUDA-side
    // parity test documents; the measured error is what bounds the difference).
    std::vector<float> yref((size_t)n_out);
    for (long long o = 0; o < n_out; ++o) {
        const uint8_t* c = codes.data() + o * nb * (QK / 4);
        const float* s = scales.data() + o * nb;
        float acc = 0.0f;
        for (long long b = 0; b < nb; ++b) {
            const float d = s[b];
            for (int j = 0; j < QK; ++j) {
                const int code = (c[b * (QK / 4) + (j >> 2)] >> ((j & 3) * 2)) & 0x03;
                acc += (float)(code - 1) * d * fp16_to_fp32(x[(size_t)(b * QK + j)]);
            }
        }
        yref[(size_t)o] = acc;
    }

    // device side: arena offsets, the .comp's packed views (xs uints = 2 halfs, cs uints = 4 bytes).
    try {
        vk::device_info();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "s2_gemv_vk_parity: %s\n", e.what());
        std::fprintf(stderr, "s2_gemv_vk_parity SKIP (no usable Vulkan device)\n");
        return selftest ? 0 : 77;   // the tree's ctest SKIP_RETURN_CODE style
    }
    vk::Arena& a = vk::Arena::instance();
    const uint64_t sx = (uint64_t)n_in * 2, sc = (uint64_t)n_out * nb * (QK / 4),
                   ss = (uint64_t)n_out * nb * 4, sy = (uint64_t)n_out * 4;
    const uint64_t dx = a.alloc(sx), dc = a.alloc(sc), ds = a.alloc(ss), dy = a.alloc(sy);
    vk::copy_h2d((void*)(uintptr_t)dx, x.data(), sx, nullptr);
    vk::copy_h2d((void*)(uintptr_t)dc, codes.data(), sc, nullptr);
    vk::copy_h2d((void*)(uintptr_t)ds, scales.data(), ss, nullptr);
    vk::fill_d((void*)(uintptr_t)dy, 0xFE, sy, nullptr);

    strata::kernels::s2_gemv((const uint16_t*)(uintptr_t)dx, (const uint8_t*)(uintptr_t)dc,
                             (const float*)(uintptr_t)ds, (float*)(uintptr_t)dy, n_in, n_out);

    std::vector<float> yvk((size_t)n_out);
    vk::copy_d2h(yvk.data(), (void*)(uintptr_t)dy, sy, nullptr);

    double worst_rel = 0.0; long long worst_at = -1; long long exact = 0;
    for (long long i = 0; i < n_out; ++i) {
        const float r = yref[(size_t)i], v = yvk[(size_t)i];
        if (r == v) { ++exact; continue; }
        const double rel = std::fabs((double)r - (double)v) / std::max(1e-3, std::fabs((double)r));
        if (rel > worst_rel) { worst_rel = rel; worst_at = i; }
    }
    std::printf("s2_gemv vk vs reference: rows %lld, bit-exact %lld, worst rel %.3e (row %lld: ref %.6f vk %.6f), tol %.0e\n",
                n_out, exact, worst_rel, worst_at, yref[(size_t)(worst_at < 0 ? 0 : worst_at)],
                yvk[(size_t)(worst_at < 0 ? 0 : worst_at)], tol);
    const bool pass = worst_rel < tol;
    std::printf("s2_gemv_vk_parity %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
