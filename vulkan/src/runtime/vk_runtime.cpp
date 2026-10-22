// vulkan/src/runtime/vk_runtime.cpp - arena, copies, kernel registry and launch.
//
// An arena address is a byte offset into a chain of VkBuffers: RADV caps one memory allocation at
// maxMemoryAllocationSize (512 MiB on the BC-250), so a region larger than that becomes several pieces.
// A kernel argument must land inside one piece (weights regions are laid out so that each tensor's planes
// do; a straddling span throws at launch with a named reason rather than reading wrong bytes). Copies walk
// the pieces. That keeps the first cut honest where CUDA's flat cudaMalloc address space is not available.
#include "strata/vulkan_runtime.hpp"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::vk {

#define RT(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    throw std::runtime_error(std::string(#x) + " -> " + std::to_string((int)r_)); } } while (0)

namespace { } // (queue family exported by vk_device.cpp as strata::vk::g_vk_qf)
extern uint32_t g_vk_qf;
uint32_t vk_queue_family() { return g_vk_qf; }

std::mutex& g_km();

namespace {

struct Piece { uint64_t base; uint64_t size; VkBuffer buf; VkDeviceMemory mem; };
std::vector<Piece> g_pieces;
uint64_t g_cursor = 0;
std::mutex g_m;

uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties((VkPhysicalDevice)vk_physical_handle(), &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return i;
    throw std::runtime_error("vk_runtime: no usable memory type");
}

VkBuffer piece_of(uint64_t off, uint64_t bytes, uint64_t& piece_off) {
    for (auto& p : g_pieces)
        if (off >= p.base && off + bytes <= p.base + p.size) { piece_off = off - p.base; return p.buf; }
    return VK_NULL_HANDLE;
}

// Bytes left in the piece that starts under `off` (0 when no piece holds it). The copy/fill chunkers
// clip to this so a span NEVER crosses a piece boundary (Task 1.4: the engine's 512 MiB allocations
// land as max-allocation piece chains, and a chunk that straddles one used to throw - the region
// itself is fine; the chunking just has to respect the chain).
uint64_t piece_room(uint64_t off) {
    for (auto& p : g_pieces)
        if (off >= p.base && off < p.base + p.size) return p.base + p.size - off;
    return 0;
}

void arena_add(uint64_t bytes) {
    VkDevice dev = (VkDevice)vk_handle();
    const DeviceInfo& d = device_info();
    uint64_t max_alloc = d.max_allocation_size ? d.max_allocation_size : (64ull << 20);
    const uint64_t slack = getenv("STRATA_VK_SLACK") ? (uint64_t)atoll(getenv("STRATA_VK_SLACK")) : (256ull << 20);
    if (g_cursor + bytes + slack > d.free_bytes)
        throw std::runtime_error("vk_runtime: out of VRAM growing the arena by " + std::to_string(bytes >> 20) + " MiB");
    while (bytes) {
        uint64_t sz = bytes > max_alloc ? max_alloc : bytes;
        Piece p; p.base = g_cursor; p.size = sz;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = sz;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        RT(vkCreateBuffer(dev, &bci, nullptr, &p.buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, p.buf, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        RT(vkAllocateMemory(dev, &mai, nullptr, &p.mem));
        RT(vkBindBufferMemory(dev, p.buf, p.mem, 0));
        g_pieces.push_back(p);
        g_cursor += sz;
        bytes -= sz;
    }
}

VkBuffer g_stage = VK_NULL_HANDLE;
VkDeviceMemory g_stage_mem = VK_NULL_HANDLE;
void* g_stage_map = nullptr;
uint64_t g_stage_size = 0;
void stage_ensure(uint64_t bytes) {
    if (g_stage_size >= bytes) return;
    VkDevice dev = (VkDevice)vk_handle();
    if (g_stage) { vkDestroyBuffer(dev, g_stage, nullptr); vkFreeMemory(dev, g_stage_mem, nullptr); g_stage = VK_NULL_HANDLE; }
    if (bytes > (2ull << 20)) bytes = 2ull << 20;      // copies chunk through a 2 MiB stage
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    RT(vkCreateBuffer(dev, &bci, nullptr, &g_stage));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, g_stage, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mem_type(mr.memoryTypeBits,
        (VkMemoryPropertyFlags)(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    RT(vkAllocateMemory(dev, &mai, nullptr, &g_stage_mem));
    RT(vkBindBufferMemory(dev, g_stage, g_stage_mem, 0));
    RT(vkMapMemory(dev, g_stage_mem, 0, mai.allocationSize, 0, &g_stage_map));
    g_stage_size = bytes;
}

VkCommandPool g_pool = VK_NULL_HANDLE;
VkCommandBuffer begin() {
    VkDevice dev = (VkDevice)vk_handle();
    if (!g_pool) {
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.queueFamilyIndex = vk_queue_family();
        RT(vkCreateCommandPool(dev, &cpi, nullptr, &g_pool));
    }
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    RT(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    RT(vkBeginCommandBuffer(cb, &bi));
    return cb;
}
void end_submit(VkCommandBuffer cb) {
    VkDevice dev = (VkDevice)vk_handle();
    RT(vkEndCommandBuffer(cb));
    VkQueue q = (VkQueue)vk_default_queue();
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    RT(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    RT(vkQueueWaitIdle(q));
    vkFreeCommandBuffers(dev, g_pool, 1, &cb);
}

void copy_chunk(bool h2d, void* host, uint64_t arena_off, uint64_t bytes) {
    uint64_t piece_off; VkBuffer pb = piece_of(arena_off, bytes, piece_off);
    if (!pb)
        throw std::runtime_error("vk_runtime: copy span (" + std::to_string(bytes) + " B at " +
                                 std::to_string(arena_off) + ") crosses an arena piece; the region needs a piece-aligned split");
    VkCommandBuffer cb = begin();
    VkBufferCopy reg;
    if (h2d) { reg.srcOffset = 0; reg.dstOffset = piece_off; reg.size = bytes; vkCmdCopyBuffer(cb, g_stage, pb, 1, &reg); }
    else     { reg.srcOffset = piece_off; reg.dstOffset = 0; reg.size = bytes; vkCmdCopyBuffer(cb, pb, g_stage, 1, &reg); }
    end_submit(cb);
    if (h2d) { /* already staged */ }
}

struct KernelImpl {
    VkPipeline pipe; VkPipelineLayout layout; VkDescriptorSetLayout dsl;
    int n_buffers; int pc_bytes; int wg;
};
std::unordered_map<uint64_t, KernelImpl> g_kernels;    // keyed by pipeline handle

}  // namespace

DevicePtr Arena::alloc(uint64_t bytes) {
    std::lock_guard<std::mutex> lk(g_m);
    uint64_t at = g_cursor;
    arena_add(bytes);
    return at;
}
uint64_t Arena::used() const { std::lock_guard<std::mutex> lk(g_m); return g_cursor; }
uint64_t Arena::capacity() const { std::lock_guard<std::mutex> lk(g_m); return g_cursor; }
void* Arena::vk_device() const { return vk_handle(); }
void Arena::buffers_for(uint64_t, uint64_t, std::vector<Extent>&) const {}
Arena::Arena() {}
Arena::~Arena() {}
Arena& Arena::instance() { static Arena a; return a; }

void* stream_create() { return nullptr; }
void stream_destroy(void*) {}
void stream_sync(void*) { RT(vkQueueWaitIdle((VkQueue)vk_default_queue())); }
void device_sync() { RT(vkDeviceWaitIdle((VkDevice)vk_handle())); }

struct Ev { };
void* event_create() { return new Ev{}; }
void event_destroy(void* e) { delete (Ev*)e; }
void event_record(void*, void*) { RT(vkQueueWaitIdle((VkQueue)vk_default_queue())); }
void event_wait(void*, void*) {}
float event_elapsed(void*, void*) { return 0.f; }

void copy_h2d(void* dst, const void* src, uint64_t bytes, void*) {
    std::lock_guard<std::mutex> lk(g_m);
    uint64_t at = (uint64_t)(uintptr_t)dst, off = 0;
    while (off < bytes) {
        uint64_t n = bytes - off < (2ull << 20) ? bytes - off : (2ull << 20);
        if (uint64_t room = piece_room(at + off)) { if (n > room) n = room; }
        stage_ensure(n);
        std::memcpy(g_stage_map, (const char*)src + off, n);
        copy_chunk(true, nullptr, at + off, n);
        off += n;
    }
}
void copy_d2h(void* dst, const void* src, uint64_t bytes, void*) {
    std::lock_guard<std::mutex> lk(g_m);
    uint64_t at = (uint64_t)(uintptr_t)src, off = 0;
    while (off < bytes) {
        uint64_t n = bytes - off < (2ull << 20) ? bytes - off : (2ull << 20);
        if (uint64_t room = piece_room(at + off)) { if (n > room) n = room; }
        stage_ensure(n);
        copy_chunk(false, nullptr, at + off, n);
        std::memcpy((char*)dst + off, g_stage_map, n);
        off += n;
    }
}
void fill_d(void* dst, uint8_t byte, uint64_t bytes, void*) {
    std::lock_guard<std::mutex> lk(g_m);
    stage_ensure(1ull << 20);
    const uint64_t chunk = g_stage_size;
    uint64_t off = 0;
    while (off < bytes) {
        uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (uint64_t room = piece_room((uint64_t)(uintptr_t)dst + off)) { if (n > room) n = room; }
        std::memset(g_stage_map, byte, n);
        copy_chunk(true, nullptr, (uint64_t)(uintptr_t)dst + off, n);
        off += n;
    }
}

std::mutex& g_km() { static std::mutex m; return m; }

Kernel kernel_load(const char* name, const uint32_t* spv, size_t spv_bytes, int n_storage_buffers,
                   int push_constant_bytes, int workgroup_size, bool force_subgroup32) {
    std::lock_guard<std::mutex> lk(g_km());
    VkDevice dev = (VkDevice)vk_handle();
    VkShaderModule sm;
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    si.codeSize = spv_bytes; si.pCode = spv;
    RT(vkCreateShaderModule(dev, &si, nullptr, &sm));
    std::vector<VkDescriptorSetLayoutBinding> bs(n_storage_buffers);
    for (int i = 0; i < n_storage_buffers; i++)
        bs[i] = VkDescriptorSetLayoutBinding{(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                             VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = (uint32_t)bs.size(); dli.pBindings = bs.data();
    VkDescriptorSetLayout dsl; RT(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkPipelineLayout pl; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_constant_bytes};
    if (push_constant_bytes) { pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr; }
    RT(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    req.requiredSubgroupSize = 32;
    VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    st.stage = VK_SHADER_STAGE_COMPUTE_BIT; st.module = sm; st.pName = "main";
    if (force_subgroup32) st.pNext = &req;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = st; ci.layout = pl;
    VkPipeline pipe; RT(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
    vkDestroyShaderModule(dev, sm, nullptr);
    KernelImpl impl{pipe, pl, dsl, n_storage_buffers, push_constant_bytes, workgroup_size};
    g_kernels[(uint64_t)pipe] = impl;
    return {pipe, pl};
}

// `buffers` are arena offsets; `sizes` the same length, each span required to sit inside one arena piece.
void kernel_launch_sized(const Kernel& k, int groups_x, const uint64_t* buffers, const uint64_t* sizes,
                         int n_buffers, const void* push_constants) {
    std::lock_guard<std::mutex> lk(g_km());
    auto it = g_kernels.find((uint64_t)k.pipeline);
    if (it == g_kernels.end()) throw std::runtime_error("vk_runtime: launch of an unloaded kernel");
    const KernelImpl& impl = it->second;
    if (n_buffers != impl.n_buffers) throw std::runtime_error("vk_runtime: kernel buffer count mismatch");
    VkDevice dev = (VkDevice)vk_handle();
    std::vector<VkDescriptorBufferInfo> dbi(n_buffers);
    for (int i = 0; i < n_buffers; i++) {
        uint64_t po; VkBuffer pb = piece_of(buffers[i], sizes[i], po);
        if (!pb)
            throw std::runtime_error("vk_runtime: kernel argument " + std::to_string(i) +
                                     " crosses an arena piece (offset " + std::to_string(buffers[i]) +
                                     ", " + std::to_string(sizes[i]) + " bytes)");
        dbi[i] = VkDescriptorBufferInfo{pb, po, sizes[i]};
    }
    VkDescriptorPool dp; VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)n_buffers};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz;
    RT(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSet ds; VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &impl.dsl;
    RT(vkAllocateDescriptorSets(dev, &dai, &ds));
    std::vector<VkWriteDescriptorSet> wr(n_buffers);
    for (int i = 0; i < n_buffers; i++) {
        wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wr[i].dstSet = ds; wr[i].dstBinding = (uint32_t)i; wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, (uint32_t)n_buffers, wr.data(), 0, nullptr);
    VkCommandBuffer cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, impl.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, impl.layout, 0, 1, &ds, 0, nullptr);
    if (impl.pc_bytes && push_constants)
        vkCmdPushConstants(cb, impl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, impl.pc_bytes, push_constants);
    vkCmdDispatch(cb, (uint32_t)groups_x, 1, 1);
    end_submit(cb);
    vkDestroyDescriptorPool(dev, dp, nullptr);
}

}  // namespace strata::vk
