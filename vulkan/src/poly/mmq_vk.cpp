// vulkan/src/poly/mmq_vk.cpp - the Task-1.4 twin of src/poly/mmq.cu for the strata-poly-vk link.
//
// strata_poly_mmq on CUDA is llama.cpp's ggml-cuda MMQ: quantize.cu plus 18 template-instances, all
// __global__ code that cannot compile without nvcc. Per the plan, the prompt path's MMQ kernels are
// TRAPPED for now (PORT-STATUS: "mmq prompt kernels: trapped (Phase 3 covers prefill)"). The trap
// pattern is missing.cpp's: a call names itself and stops the run (exit 86) — but the load path asks
// Mmq::fits() BEFORE any kernel is demanded (engine.cpp's tiles decision), and the CUDA engine's
// answer for a card with no tile config is false with nothing built: on Vulkan no MMQ tile exists, so
// fits() answers false HONESTLY and the engine keeps its batched path off. The rest — quantize(),
// mul(), the constructor — are the kernel sites: they trap by name. q8_bytes() is pure size
// arithmetic (pad512 * block_q8_1_mmq) and is reached only when fits() said yes, so it never runs;
// it is defined unchanged for the contract.
#include "mmq.hpp"   // the directory of this file's CUDA twin: vulkan/src/poly sits beside src/poly's shape;
                     // CMake also puts ${STRATA_ROOT}/src/poly on this target's include path.

namespace strata::vk_engine { void missing(const char* name); }

namespace strata::poly {

Mmq::Mmq() { strata::vk_engine::missing("strata::poly::Mmq (ggml-cuda mmq context)"); }
Mmq::~Mmq() {}

bool Mmq::fits(int, int64_t, int64_t) {
    // No MMQ tile is built for this backend yet (Phase 3, the plan's Task 3.1): the honest answer is
    // "this build does not multiply that type", which is what CUDA's fits() returns when the card has
    // no configuration for it. The engine then runs those layers through its non-batched path.
    return false;
}

size_t Mmq::q8_bytes(int64_t rows, int64_t cols) {
    // The same arithmetic as src/poly/mmq.cu: rows of pad512(cols) values in 32-value Q8_1 blocks
    // (block_q8_1_mmq is 132 bytes on llama.cpp at this pin), plus the tile tail. Reached only where
    // fits() said true, which on this backend is nowhere; defined so the contract links.
    const int64_t padded = (cols + 511) / 512 * 512;
    return (size_t) rows * (size_t) (padded / 32) * 132 + 128 * 132;
}

void Mmq::quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*) {
    strata::vk_engine::missing("strata::poly::Mmq::quantize (ggml-cuda quantize_mmq_q8_1)");
}

void Mmq::mul(const void*, int, int64_t, int64_t, size_t, int, const void*, const int32_t*, const int32_t*,
              int64_t, int64_t, float*, int64_t, void*) {
    strata::vk_engine::missing("strata::poly::Mmq::mul (ggml-cuda mul_mat_q_case)");
}

}  // namespace strata::poly
