// vulkan/src/tests/vk_core_smoke.cpp — Task 1.3's compile+run proof: the device/pinned/vmm twins.
//   * strata::core::device_info / device_summary / device_count / gpu_arch_problem against the real
//     device (the numbers printed are THIS box's: the claim the ledger records);
//   * core::DeviceArena: bump allocation, the pointer contract a cudaMemcpy round-trips through the
//     compat layer (write pattern -> copy -> read back), overflow throws named;
//   * core::PinnedRing (pinned_ring.hpp, header-inline over the compat calls): h2d through a pinned
//     slot lands the bytes in the arena;
//   * core::PinnedArena construct/teardown (the UMA twin path) and the vmm fallback contract:
//     vmm_available()/granularity()/reserve/mapped — grow-never reports every chunk mapped.
// Exit 0 = all green.
#include "strata/core/device.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/pinned_ring.hpp"
#include "strata/core/vmm.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::core;

static int fails = 0;
#define CHECK(cond, what) do { if (!(cond)) { \
    std::printf("FAIL: %s\n", what); ++fails; } else std::printf("ok: %s\n", what); } while (0)

int main() {
    // ---- device twin
    CHECK(device_count() == 1, "device_count() == 1");
    set_primary_device(0);
    CHECK(primary_device() == 0, "primary_device() round-trips");
    DeviceInfo d = device_info(0);
    std::printf("info: ordinal %d, %s, cc %d.%d, %llu MiB total / %llu MiB free, SMs %d, arch %s\n",
                d.ordinal, d.name.c_str(), d.cc_major, d.cc_minor,
                (unsigned long long)(d.total_bytes >> 20), (unsigned long long)(d.free_bytes >> 20),
                d.multi_processor_count, d.arch.c_str());
    CHECK(d.cc_major == 10 && d.cc_minor == 1, "cc is the RDNA1 family code 10.1");
    CHECK(d.arch == "gfx1013", "arch string is gfx1013");
    CHECK(d.multi_processor_count == 24, "multi_processor_count 24 (VK_AMD_shader_core_properties: 2 engines x 2 arrays x 6 CUs, measured)");
    CHECK(gpu_arch_problem(0).empty(), "gpu_arch_problem(0) == \"\" (probe gates passed)");
    std::string name, detail;
    CHECK(device_summary(0, name, detail), "device_summary reports");
    std::printf("summary: %s: %s\n", name.c_str(), detail.c_str());
    CHECK(compiled_gpu_archs() == std::string("gfx1013"), "compiled_gpu_archs names the card");
    bool threw = false;
    try { (void) device_info(5); } catch (const CudaError&) { threw = true; }
    CHECK(threw, "device_info(5) throws CudaError (ordinal out of range)");

    // ---- DeviceArena: bump + the pointer contract through the compat copies
    DeviceArena arena(4ull << 20, 0, false);
    void* p0 = arena.alloc(1000);                 // default align 256
    void* p1 = arena.alloc(300, 64);
    CHECK(((std::uintptr_t) p1 & 63) == 0 && (char*) p1 > (char*) p0, "bump sub-allocation honors align");
    CHECK(arena.capacity() == (4ull << 20) && arena.used() >= 1300, "capacity/used track the bump");
    std::vector<uint32_t> pat(256);
    for (size_t i = 0; i < pat.size(); i++) pat[i] = (uint32_t)(i * 7919u + 13u);
    CHECK(cudaMemcpy(p1, pat.data(), pat.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess, "cudaMalloc-style device ptr: h2d");
    std::vector<uint32_t> back(256, 0);
    CHECK(cudaMemcpy(back.data(), p1, back.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess && back == pat,
          "device ptr round-trip is bit-exact");
    threw = false;
    try { (void) arena.alloc(8ull << 20); } catch (const CudaError& e) {
        threw = std::strstr(e.what(), "DeviceArena out of memory") != nullptr;
    }
    CHECK(threw, "over-capacity alloc throws named");

    // ---- PinnedRing over the compat host path (header-inline twin: cudaHostAlloc = aligned RAM)
    {
        PinnedRing ring;
        std::vector<uint8_t> src(70000);
        for (size_t i = 0; i < src.size(); i++) src[i] = (uint8_t)(i ^ 0x5a);
        void* dst = nullptr;
        CHECK(cudaMalloc(&dst, src.size()) == cudaSuccess, "arena alloc for the ring's destination");
        CHECK(ring.h2d(dst, src.data(), src.size(), nullptr) == cudaSuccess, "PinnedRing::h2d (pinned slot path)");
        std::vector<uint8_t> rb(src.size(), 0);
        CHECK(cudaMemcpy(rb.data(), dst, rb.size(), cudaMemcpyDeviceToHost) == cudaSuccess && rb == src,
              "PinnedRing upload landed bit-exact");
        CHECK(ring.reserve(1 << 20), "PinnedRing::reserve grows the slots");
    }

    // ---- PinnedArena (the UMA twin: huge pages or 4K, reported in `note`)
    {
        PinnedArena pa(8ull << 20);
        CHECK(pa.valid(), "PinnedArena of 8 MiB is valid");
        std::printf("arena backing: %s%s\n",
                    pa.backing == PageBacking::LargePages ? "LargePages"
                    : pa.backing == PageBacking::NormalPages ? "NormalPages" : "PinnedByCuda",
                    ("; " + pa.note).c_str());
        pa.data()[0] = 42;
        CHECK(pa.data()[0] == 42, "arena pages are writable");
    }

    // ---- vmm fallback: grow-never, every chunk mapped from the start
    CHECK(vmm_available() && vmm_granularity() == (2ull << 20), "vmm_available + 2 MiB granularity");
    {
        VmmRange r;
        CHECK(r.reserve(5ull << 20), "VmmRange::reserve(5 MiB) -> 3 chunks");
        CHECK(r.chunks() == 3 && r.mapped(0) && r.mapped(2) && !r.mapped(3), "all chunks mapped at reserve (grow-never)");
        CHECK(r.map_range(0, 3, [] { return (VmmChunk) 0; }), "map_range is a no-op over an all-mapped range");
        CHECK(r.unmap(1) != 0 && !r.mapped(1), "unmap moves bookkeeping (bytes stay: bump arena)");
        CHECK(r.base() != nullptr, "base() hands a device pointer");
        float one = 1.5f;
        CHECK(cudaMemset(r.base(), 0, 4) == cudaSuccess, "memset through base() (compat device ptr)");
        float zero = 9.f;
        CHECK(cudaMemcpy(&zero, r.base(), 4, cudaMemcpyDeviceToHost) == cudaSuccess && zero == 0.f,
              "memset visible via d2h");
    }

    std::printf("vk_core_smoke %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
