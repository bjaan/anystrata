// scratch: quantize one known block, dump raw 36 bytes.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace strata;
int main() {
    vk::device_info();
    auto& a = vk::Arena::instance();
    const int n = 32;
    float x[32];
    for (int j = 0; j < 32; ++j) x[j] = (j == 0) ? -500.0f : 1.0f;   // amax=500, sum=501? no: -500+31 = -469
    const uint64_t ax = a.alloc(128), aq = a.alloc(36);
    vk::copy_h2d((void*)ax, x, 128, nullptr);
    vk::fill_d((void*)aq, 0, 36, nullptr);
    kernels::native_quantize_q8_1((const float*)strata_vk_compat::wrap(ax), strata_vk_compat::wrap(aq),
                                  n, 1, strata_vk_compat::wrap(1000));
    unsigned char q[36];
    vk::copy_d2h(q, (void*)aq, 36, nullptr);
    std::printf("ds bytes: %02x %02x %02x %02x  (d f16 bits lo,hi ; sum f16 lo,hi)\n", q[0],q[1],q[2],q[3]);
    unsigned d16 = q[0] | (q[1]<<8), s16 = q[2] | (q[3]<<8);
    auto dec=[](unsigned h){ unsigned sign=(h>>15)&1, exp=(h>>10)&31, man=h&1023; double v = exp? std::ldexp((double)(1024+man), (int)exp-25) : (double)man/16384.0; return sign? -v:v; };
    std::printf("d = %.7f (0x%04x), sum = %.4f (0x%04x)\n", dec(d16), d16, dec(s16), s16);
    std::printf("qs[0..5]: %d %d %d %d %d %d\n", (signed char)q[4],(signed char)q[5],(signed char)q[6],(signed char)q[7],(signed char)q[8],(signed char)q[9]);
    return 0;
}
