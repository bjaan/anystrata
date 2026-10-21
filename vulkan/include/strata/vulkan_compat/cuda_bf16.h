// cuda_bf16.h for a Vulkan build: __nv_bfloat16, the same shape as cuda_fp16.h's wrapper.  bf16 is a plain
// truncation-with-RNE of f32's top 16 bits (see include/strata/kernels/bf16_bits.hpp, the bit-exact engine
// converter this mirrors for the host metadata paths).
#pragma once

#include <cstdint>
#include <cstring>

struct __nv_bfloat16 {
    uint16_t x;
    __nv_bfloat16() = default;
    explicit __nv_bfloat16(uint16_t bits) : x(bits) {}
};
typedef __nv_bfloat16 bf16;

inline float __bfloat162float(__nv_bfloat16 h) {
    uint32_t x = (uint32_t) h.x << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}
inline __nv_bfloat16 __float2bfloat16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t r = x + 0x7FFFu + ((x >> 16) & 1u);       // round-to-nearest-even, then take the high half
    if (((x >> 23) & 0xFFu) == 0xFFu && (x & 0x7FFFFFu)) r = x + 0x1000u;  // keep NaNs quiet
    return __nv_bfloat16((uint16_t)(r >> 16));
}
inline __nv_bfloat16 __ushort_as_bfloat16(unsigned short bits) { return __nv_bfloat16(bits); }
inline unsigned short __bfloat16_as_ushort(__nv_bfloat16 h) { return h.x; }
inline bool operator<(__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat162float(a) < __bfloat162float(b); }
inline bool operator>(__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat162float(a) > __bfloat162float(b); }
inline bool operator==(__nv_bfloat16 a, __nv_bfloat16 b) { return a.x == b.x; }
inline bool operator!=(__nv_bfloat16 a, __nv_bfloat16 b) { return a.x != b.x; }
