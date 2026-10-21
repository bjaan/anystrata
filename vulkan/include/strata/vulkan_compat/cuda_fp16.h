// cuda_fp16.h for a Vulkan build: __half as a 16-bit wrapper with software IEEE conversion.
//
// hip_compat gets these from <hip/hip_fp16.h>; there is no vendor header here, so the conversion is spelled
// out.  Host-side conversions only ever touch metadata paths (the engine's bit-exact converters live in
// include/strata/kernels/f16_bits.hpp and are the ones the kernel paths use); these wrappers exist so the
// host TUs that name __half/__half2float compile and behave, matching CUDA's semantics closely enough that
// the same f16_bits.hpp tests pass on them.  Conversion is round-to-nearest-even with fp16 inf saturation,
// the same regime table as f16_bits.hpp.
#pragma once

#include <cstdint>
#include <cstring>

struct __half {
    uint16_t x;
    __half() = default;
    explicit __half(uint16_t bits) : x(bits) {}
};
struct __half2 { __half x, y; };
typedef __half half;
typedef __half2 half2;

inline float strata_vk_f32_from_f16_bits(uint16_t h) {
    // Verbatim transcription of include/strata/kernels/f16_bits.hpp's f32_from_f16 (the tree's tested,
    // subnormal-exact converter; __builtin_memcpy branches dropped: this is host code only).
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t ex = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t out;
    if (ex == 0) {
        if (man == 0) {
            out = sign;
        } else {
            uint32_t m = man;
            int s = 0;
            while ((m & 0x400u) == 0) { m <<= 1; ++s; }
            out = sign | ((uint32_t) (113 - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (ex == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((ex - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}
inline uint16_t strata_vk_f16_bits_from_f32(float f) {
    // Verbatim transcription of f16_bits.hpp's f16_from_f32 (RNE, finite-overflow -> inf; the regime
    // comments and the numpy-oracle history are in that header).
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    int exp = (int) rawexp - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (rawexp == 0xFFu) return (uint16_t) (sign | 0x7C00u | (man ? 0x200u : 0u));   // inf, or a quiet/NaN
    if (exp >= 31) return (uint16_t) (sign | 0x7C00u);                              // finite overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;                                // underflows to a signed zero
        man |= 0x800000u;                                                     // restore the implicit 1
        const uint32_t sh = (uint32_t) (14 - exp);
        uint32_t h = (man >> sh) & 0x3FFu;
        const uint32_t rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    uint16_t h = (uint16_t) (sign | ((uint32_t) exp << 10) | (man >> 13));
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;                   // the carry may bump the exponent
    return h;
}

inline float __half2float(__half h) { return strata_vk_f32_from_f16_bits(h.x); }
inline __half __float2half(float f) { return __half(strata_vk_f16_bits_from_f32(f)); }
inline __half __ushort_as_half(unsigned short bits) { return __half(bits); }
inline unsigned short __half_as_ushort(__half h) { return h.x; }
inline __half operator+(__half a, __half b) { return __float2half(__half2float(a) + __half2float(b)); }
inline __half operator-(__half a, __half b) { return __float2half(__half2float(a) - __half2float(b)); }
inline __half operator*(__half a, __half b) { return __float2half(__half2float(a) * __half2float(b)); }
inline __half operator/(__half a, __half b) { return __float2half(__half2float(a) / __half2float(b)); }
inline bool operator<(__half a, __half b) { return __half2float(a) < __half2float(b); }
inline bool operator>(__half a, __half b) { return __half2float(a) > __half2float(b); }
inline bool operator<=(__half a, __half b) { return __half2float(a) <= __half2float(b); }
inline bool operator>=(__half a, __half b) { return __half2float(a) >= __half2float(b); }
inline bool operator==(__half a, __half b) { return a.x == b.x; }
inline bool operator!=(__half a, __half b) { return a.x != b.x; }
inline __half2 make_half2(__half a, __half b) { return {a, b}; }
inline float __low2float(__half2 h) { return __half2float(h.x); }
inline float __high2float(__half2 h) { return __half2float(h.y); }
