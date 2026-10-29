// vulkan/include/strata/vulkan_runtime.hpp - the Vulkan backend's device, arena, queues and events.
//
// The contract mirrors what src/core/{device,pinned}.cu and the kernels' launchers need:
//   * one physical device picked at start (STRATA_VK_DEVICE by name substring, else the first with compute),
//   * DeviceArena: ONE big VkBuffer per region with bump sub-allocation below it and no frees, exactly like
//     src/core/device.hpp's DeviceArena: the memory plan is fixed at startup and an allocator that could free
//     would add fragmentation the engine does not have. A 4 GiB buffer is split across several allocations on
//     drivers whose maxMemoryAllocationSize is smaller (RADV on the BC-250: 512 MiB per allocation); spans stay
//     device-indexable because one VkDeviceMemory binds at a time inside one buffer, and the engine's copies
//     go through vkCmdCopyBuffer, which needs no address aliasing.
//   * VkStream: a VkQueue with its own command buffers, the meaning of a cudaStream_t;
//   * VkEvent: a timeline semaphore, the meaning of a cudaEvent_t.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::vk {

struct DeviceInfo {
    int ordinal = -1;
    std::string name;
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;
    uint32_t subgroup_size = 0;          // the hardware width (64 on RDNA1/BC-250)
    uint32_t max_workgroup_invocations = 0;
    uint64_t max_allocation_size = 0;
    bool integer_dot_product = false;
    bool subgroup_size_control = false;
    bool buffer_device_address = false;
    bool unified_memory = false;
    std::string driver;
    // Task 1.3 (step 5): the facts cudaDeviceProp exposes that vk_device.cpp did not carry.
    // multi_processor_count comes from VK_AMD_shader_core_properties when the driver answers it
    // (radv on the BC-250: shaderEngineCount 2 x shaderArraysPerEngineCount 2 x computeUnitsPerShaderArray 6
    // = 24 CUs; a BC-250 BIOS may unlock to 40 — the driver's figure is what kernels plan against, and
    // STRATA_VK_CU_OVERRIDE rewrites it for occupancy experiments). l2_bytes: no Vulkan query exists for
    // it (radv exposes no L2 property), so it stays 0 and the compat layer says so — readers only tune
    // prefetch/bench sizes with it (src/kernels/s2_expert_grouped_parity.cpp:532).
    int multi_processor_count = 0;
    int l2_bytes = 0;
};

// Throws when there is no usable Vulkan device. Init once per process.
const DeviceInfo& device_info(int ordinal = 0);
int device_count();

// Raw handles for the compat layer and the kernel launchers.
void* vk_handle();          // VkDevice
void* vk_physical_handle();// VkPhysicalDevice
void* vk_default_queue();   // VkQueue (the null-stream's queue)

// ---- device pointers: an offset in the arena's address space (never a host pointer into a VkDeviceMemory map).
using DevicePtr = uint64_t;
constexpr DevicePtr kNull = 0;

class Arena {
public:
    static Arena& instance();
    DevicePtr alloc(uint64_t bytes);                    // bump; throws when out of VRAM
    uint64_t used() const;
    uint64_t capacity() const;

    void* vk_device() const;    // VkDevice handle, for the compat layer
    // internal: find the VkBuffer an offset lands in (may straddle allocations)
    struct Extent { DevicePtr ptr; uint64_t bytes; };
    void buffers_for(uint64_t off, uint64_t bytes, std::vector<Extent>& out) const;

private:
    struct Impl;
    Arena();
    ~Arena();
    Impl* p_;
};

// ---- the stream/event layer. These return raw handles the compat layer hands to ../src.
void* stream_create();                                   // opaque VkStream*
void  stream_destroy(void* s);
void  stream_sync(void* s);
void  device_sync();
void* event_create();
void  event_destroy(void* e);
void  event_record(void* e, void* s);
void  event_wait(void* e, void* s);                      // stream waits on event
float event_elapsed(void* from, void* to);

// ---- copies between the arena and host memory (pinned or ordinary; the loader handles staging).
// ---- Phase 4.2: graph capture (compat cudaGraph* family drives these).
void graph_begin();               // open a capture on the default queue
uint32_t graph_end();             // close it; returns the recorded dispatch count, registers a graph
void graph_launch(uint32_t gi);   // replay graph gi (stream order: pending + its own prior replay first)
void graph_destroy(uint32_t gi);  // free CB + pools after waiting its last replay
bool capturing();                 // a capture is open (on any thread)
bool cap_here();                  // a capture is open ON THIS THREAD (ThreadLocal mode: other
                                  // threads' eager copies/syncs during it proceed normally)
void graph_record_h2d(uint64_t arena_off, const void* host, uint64_t bytes);
void graph_record_d2h(uint64_t arena_off, void* host_dst, uint64_t bytes);
void graph_record_d2d(uint64_t dst_arena, uint64_t src_arena, uint64_t bytes);
uint32_t capture_graph_count();   // registered graphs (launch indices are this order)
uint32_t graph_dispatches(uint32_t gi);   // launches a captured graph recorded (its node count)

uint64_t piece_tail(uint64_t arena_off);  // bytes from off to its piece end (whole-buffer bound)

void copy_h2d(void* dst_arena, const void* src_host, uint64_t bytes, void* stream);
void copy_d2h(void* dst_host, const void* src_arena, uint64_t bytes, void* stream);
void fill_d(void* dst_arena, uint8_t byte, uint64_t bytes, void* stream);

// ---- kernels.
// Compile-time-embedded SPIR-V (see tools/embed_spv.py); one registry call per module, cached.
struct Kernel {
    void* pipeline;             // opaque VkPipeline*
    void* layout;               // opaque VkPipelineLayout*
};
Kernel kernel_load(const char* name, const uint32_t* spv, size_t spv_bytes,
                   int n_storage_buffers, int push_constant_bytes, int workgroup_size,
                   bool force_subgroup32);
// `buffers` are arena offsets with their byte sizes; each span must sit inside one arena piece.
void kernel_launch_sized(const Kernel& k, int groups_x, const uint64_t* buffers, const uint64_t* sizes,
                         int n_buffers, const void* push_constants);
// Barrier-elision arm: rw bit i = 1 if the kernel WRITES buffer i, 0 if it only reads it.
// The runtime skips the post-dispatch full-RW barrier when this launch provably shares no
// RAW/WAR/WAW span with the barrier-free cluster recorded in the pending CB, letting RADV
// overlap independent dispatches (the per-dispatch barrier is ~12us of a 500k-dispatch
// prefill). Callers must pass the EXACT write set; a wrong mask is a race, not a slowdown.
void kernel_launch_sized_rw(const Kernel& k, int groups_x, const uint64_t* buffers, const uint64_t* sizes,
                            uint32_t rw_mask, int n_buffers, const void* push_constants);

}  // namespace strata::vk
