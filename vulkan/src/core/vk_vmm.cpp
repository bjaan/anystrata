// vulkan/src/core/vk_vmm.cpp — the Vulkan twin of src/core/vmm.cpp (Task 1.3): grow-never, arena-backed.
//
// CUDA's VMM (cuMemAddressReserve/cuMemMap: a fixed address range whose 2 MiB chunks get mapped and
// unmapped by physical handle) lets the elastic K/V and the expert cache trade physical VRAM without
// any pointer moving. RADV has no sparse-buffer path this port uses (the plan: "sparse cuMemMap
// equivalent = VK_KHR_maintenance5/no sparse"), so the same interface lands on the piece-chain arena:
//
//   * reserve(bytes): allocate the WHOLE span now (grow-never: everything the range can ever map is
//     physical from the start — 35B-A3B at 32K ctx of int8 KV is ~2-3 GiB, it fits the 15.7 GiB heap),
//   * mapped(i): true from the start — map_range finds every chunk already mapped and returns true
//     without touching the arena (the engine's kvg_ensure/pool_grow paths keep working, they just stop
//     growing),
//   * unmap(i)/vmm_chunk_free: free NOTHING (bump arena, the same --no-free rule as everywhere): unmap
//     reports its chunk so the accounting moves, and a shrink returns the handles the give() callback
//     drops. Physical VRAM is never reclaimed — PORT-STATUS records the row as
//     `host-free fallback (grow-never)`: the elastic tier reports full at all times, and --kv-grow
//     costs its maximum context up front instead of at token 4000.
//
// The chunk "handle" is the arena offset of the chunk's own bytes when the engine supplies one through
// a take() callback (a spare donated from another range or the cache): those bytes stay allocated where
// they are — the mapped position keeps its own reserve-time memory, so a take()-donated range wastes
// the donated bytes (bounded by the spare-list size; the honest cost of grow-never without sparse
// mapping, recorded here rather than hidden).
//
// src/core/vmm.cpp (the CUDA one) is NOT compiled by this backend — its driver entry points resolve to
// null through the compat layer anyway, which is the engine's own documented fallback; this twin gives
// the elastic tier a working arena-backed form instead of silently switching it off.
#include "strata/core/vmm.hpp"

#include "strata/vulkan_compat/strata_vk.h"

#include <cstdio>
#include <stdexcept>

namespace strata::core {

namespace {
constexpr uint64_t kGran = 2ull << 20;   // the CUDA VMM granularity this interface assumed (2 MiB)
}  // namespace

bool vmm_available() { return true; }    // the fallback IS available: grow-never, arena-backed
uint64_t vmm_granularity() { return kGran; }

VmmChunk vmm_chunk_new() {
    try {
        return (VmmChunk)(uintptr_t) strata_vk_compat::wrap(strata::vk::Arena::instance().alloc(kGran));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-vk vmm: chunk alloc failed: %s\n", e.what());
        return 0;   // the interface's out-of-memory answer
    }
}
void vmm_chunk_free(VmmChunk) {}         // bump arena: frees nothing (grow-never)

bool VmmRange::reserve(uint64_t bytes) {
    release();
    if (bytes == 0) return false;
    const uint64_t n = (bytes + kGran - 1) / kGran;
    try {
        base_ = (unsigned long long)(uintptr_t)
            strata_vk_compat::wrap(strata::vk::Arena::instance().alloc(n * kGran));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-vk vmm: reserve of %llu MiB failed: %s\n",
                     (unsigned long long)((n * kGran) >> 20), e.what());
        return false;
    }
    // grow-never: every chunk is physical now
    h_.assign((size_t) n, 1);            // 1 = mapped (the handle identity does not matter here: it is
                                         // this range's own arena span, and unmap reports it, not frees it)
    return true;
}

void VmmRange::release() {
    base_ = 0;
    h_.clear();                          // bump arena: the bytes stay (grow-never, no reclaim)
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    if (base_ == 0 || i < 0 || i >= (int64_t) h_.size()) return false;
    h_[(size_t) i] = h != 0 ? h : 1;
    return true;
}
bool VmmRange::set_access(int64_t, int64_t) { return base_ != 0; }
bool VmmRange::commit_run(int64_t lo, int64_t hi) { return lo >= hi || set_access(lo, hi); }

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const VmmChunk h = h_[(size_t) i];
    h_[(size_t) i] = 0;                  // bookkeeping only: nothing is released (grow-never)
    return h;
}

}  // namespace strata::core
