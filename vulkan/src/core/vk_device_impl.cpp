// vulkan/src/core/vk_device_impl.cpp — the Vulkan twin of src/core/device.cu (Task 1.3).
//
// Same include/strata/core/device.hpp contract, no CUDA: the facts come from strata::vk::device_info
// (vk_device.cpp) and the arena bump over strata::vk::Arena. device.cu is NOT compiled by this backend
// (the plan's rule: the CUDA/HIP trees stay byte-identical; each platform supplies its own twin).
//
// The arch story this twin tells, honestly:
//   * cc_major/cc_minor = 10/1: the RDNA1 family code the plan settled on (the compat layer's
//     cudaDeviceProp carries the same pair; the engine's gfx probes read gcnArchName, not these).
//   * arch = "gfx1013": the engine's HIP-style field, taken from this backend's device-name match —
//     vk_device.cpp refuses any device that is not the dot-product+subgroup-control class this port
//     was written for. The probe (vk_device_probe) asserted dp4a-forcing and wave32 by EXECUTION;
//     gpu_arch_problem() stays "" while that is true.
//   * device_code_error(): this backend has no device-symbol image to query (the CUDA build's
//     cudaFuncGetAttributes(poison_kernel) stand-in); the honest answer is "" — vk_device.cpp's
//     capability refusals at init are this check's equivalent, and they run before any allocation.
#include "strata/core/device.hpp"

#include "strata/vulkan_compat/strata_vk.h"   // the device-pointer wrap (arena offset + kPtrBias)

#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {
void check_vk(bool ok, const char* what) {
    if (!ok) throw CudaError(std::string(what) + ": strata-vk runtime failure", -1);
}
}  // namespace

const char* compiled_gpu_archs() { return "gfx1013"; }   // this twin is built for one card (docs/VULKAN.md)

int device_count() {
    try {
        return strata::vk::device_count();
    } catch (const std::exception&) {
        return 0;   // no usable Vulkan device: the header's "0 when there is none (or no usable runtime)"
    }
}

// STRATA_PRIMARY_DEVICE: single-device backend — the ordinal is remembered (a caller asking for device 1
// gets the same refusal device.cu's callers see), nothing switches.
static int g_primary_device = 0;
void set_primary_device(int ordinal) { g_primary_device = ordinal; }
int primary_device() { return g_primary_device; }

bool device_summary(int ordinal, std::string& name, std::string& detail) {
    if (ordinal < 0 || ordinal >= device_count()) return false;
    try {
        const strata::vk::DeviceInfo& d = strata::vk::device_info(ordinal);
        char buf[160];
        std::snprintf(buf, sizeof buf, "arch gfx1013, %.1f GiB, wave32-forced (hw 64)",
                      (double) d.total_bytes / (1024.0 * 1024 * 1024));
        name = d.name;
        detail = buf;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string gpu_arch_problem(int ordinal) {
    // The Vulkan twin is compiled for exactly the card class vk_device.cpp admits (it throws at init
    // for anything without shader-integer-dot-product + subgroup-size-control, and vk_device_probe
    // EXECUTES the dp4a and forced-32 checks). A device that got this far runs this binary.
    (void) ordinal;
    return "";
}

std::string device_code_error() { return ""; }   // see the header comment: init-time refusals stand in

DeviceInfo device_info(int ordinal) {
    const int count = device_count();
    if (count == 0)
        throw CudaError("no Vulkan device is present; Strata's Vulkan backend needs a RADV gfx1013-class "
                        "device (VK_KHR_shader_integer_dot_product + VK_EXT_subgroup_size_control)", -1);
    if (ordinal < 0 || ordinal >= count)
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(count) + ")", -1);
    const strata::vk::DeviceInfo& v = strata::vk::device_info(ordinal);
    DeviceInfo d;
    d.ordinal = ordinal;
    d.name = v.name;
    d.cc_major = 10;                  // RDNA1 family code (the plan; compat cudaDeviceProp agrees)
    d.cc_minor = 1;
    d.total_bytes = v.total_bytes;    // the Vulkan heaps' sum (this box: 15.4 GiB unified)
    d.free_bytes = v.free_bytes;      // VK_EXT_memory_budget's budget-usage, queried at init
    d.driver_version = 12040;         // the compat layer's claimed CUDART (the loader is Vulkan 1.4)
    d.runtime_version = 12040;
    d.multi_processor_count = v.multi_processor_count;   // VK_AMD_shader_core_properties (24 on this box)
    d.arch = "gfx1013";
    return d;
}

// ---- DeviceArena: one arena region over strata::vk::Arena's piece chain (bump, no frees — the same
// contract device.hpp prints). The returned pointers are the compat layer's device pointers: what
// cudaMalloc hands the host files (strata_vk_compat::wrap, arena offset + the 64-byte bias).
DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    check_vk(ordinal >= 0 && ordinal < device_count(), "DeviceArena ordinal");
    try {
        base_ = strata_vk_compat::wrap(strata::vk::Arena::instance().alloc(bytes));
    } catch (const std::exception& e) {
        throw CudaError(std::string("DeviceArena: ") + e.what(), -1);   // device.cu's own code for arena failures
    }
    used_ = 0;
    if (poison_) {
        // the NaN-ish pattern the header asks for (a debug build's guard against reading uninitialised
        // VRAM as plausible zeros): fill_d walks the pieces; 0x7f7f7f7f per float word is a NaN. The
        // CUDA twin's poison_kernel cannot run here (no device-symbol image), and a byte fill of 0x7f
        // gives every float in the region the same "clearly not a real value" shape.
        try {
            strata::vk::fill_d((void*)(std::uintptr_t) strata_vk_compat::off(base_), 0x7f, bytes, nullptr);
        } catch (const std::exception& e) {
            throw CudaError(std::string("DeviceArena poison: ") + e.what(), -1);
        }
    }
}

DeviceArena::~DeviceArena() {}   // bump arena: no frees (the CUDA build's --no-free rule)

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0)
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
