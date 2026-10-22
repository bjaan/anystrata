// vulkan/src/probe/doorbell_probe.cpp — Task 1.3 Step 1: the doorbell handshake, MEASURED on this box.
//
// The engine's token graph hands work between host and GPU through mapped pinned words the GPU spins on
// (include/strata/kernels/elementwise.hpp: doorbell_ring / doorbell_wait — "one thread spins until the
// host-written *d_flag equals the ring value"). The tree's sycl/ port carries the landmine
// (sycl/include/strata/sycl_queue.hpp:43): "On an Arc A750 (Alchemist, i915) a kernel never sees a host
// store to ordinary host USM, with or without system-scope atomics: every ring wait runs to its bound".
// Whether RADV on the BC-250 delivers host stores to a RUNNING kernel decides how the pinned twins are
// built, so it is probed before anything is built on it.
//
// Arm 1 (the live question): a HOST_VISIBLE|HOST_COHERENT buffer — exactly the backing the compat
// layer's cudaHostAlloc hands out, and what the twins will keep — is bound to probe_doorbell.comp
// (40M atomicAdd-spin reads ≈ 0.74 s on this part). This program submits WITHOUT waiting, flips w[0]
// from a host thread 200 ms into the spin (a plain store to coherent mapped memory), waits the queue,
// and reads back whether the kernel observed the flip and echoed the sentinel.
//
// Arm 2 (the fallback handshake, must be green whichever way arm 1 lands): the SAME spin kernel bound
// to a DEVICE_LOCAL buffer whose flag word was staged in by vkCmdCopyBuffer BEFORE the dispatch (the
// same road vk_runtime's copies take). If the sentinel appears, "flush the flag into device memory per
// ring, then launch the waiter" is a working doorbell on this driver.
//
// Self-contained raw Vulkan on purpose: the runtime's every-submit-waits path (end_submit) would make
// "while it runs" unmeasurable through that layer.
//
// Exit 0 = arm 2 green (the twins' contract either way); the arm-1 verdict (VISIBLE / NEVER SEEN) is
// what PORT-STATUS records as the box's truth.
#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "probe_doorbell.spv.h"    // embedded by tools/embed_spv.py

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    std::fprintf(stderr, "doorbell: %s -> %d\n", #x, (int)r_); std::exit(2); } } while (0)

namespace {

VkInstance g_inst;
VkPhysicalDevice g_phys;
VkDevice g_dev;
VkQueue g_queue;
uint32_t g_qf = 0;

uint32_t pick_memtype(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    std::fprintf(stderr, "doorbell: no memory type for flags %x\n", (unsigned)want);
    std::exit(2);
}

struct Buf { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; void* map = nullptr; };
Buf make_buf(bool host_visible) {
    Buf c{};
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 64;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CK(vkCreateBuffer(g_dev, &bci, nullptr, &c.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev, c.buf, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = host_visible
        ? pick_memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : pick_memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CK(vkAllocateMemory(g_dev, &mai, nullptr, &c.mem));
    CK(vkBindBufferMemory(g_dev, c.buf, c.mem, 0));
    if (host_visible) CK(vkMapMemory(g_dev, c.mem, 0, VK_WHOLE_SIZE, 0, &c.map));
    return c;
}

void submit_copy(VkBuffer dst, VkBuffer src, VkDeviceSize bytes) {
    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpc.queueFamilyIndex = g_qf;
    VkCommandPool cp; CK(vkCreateCommandPool(g_dev, &cpc, nullptr, &cp));
    VkCommandBuffer cb; VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = cp; ca.commandBufferCount = 1;
    CK(vkAllocateCommandBuffers(g_dev, &ca, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy reg{0, 0, bytes};
    vkCmdCopyBuffer(cb, src, dst, 1, &reg);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    CK(vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE));
    CK(vkQueueWaitIdle(g_queue));
    vkDestroyCommandPool(g_dev, cp, nullptr);
}

void clear_buf(Buf& b) {
    Buf stage = make_buf(true);
    std::memset(stage.map, 0, 64);
    submit_copy(b.buf, stage.buf, 64);
    vkUnmapMemory(g_dev, stage.mem);
    vkDestroyBuffer(g_dev, stage.buf, nullptr);
    vkFreeMemory(g_dev, stage.mem, nullptr);
}

struct Pipe { VkPipeline pipe; VkPipelineLayout layout; VkDescriptorSet ds; VkCommandBuffer cb; };
Pipe make_pipe(VkBuffer bound) {
    Pipe p{};
    VkShaderModule sm;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = probe_doorbell_spv_len; smi.pCode = probe_doorbell_spv;
    CK(vkCreateShaderModule(g_dev, &smi, nullptr, &sm));
    VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 1; dli.pBindings = &b0;
    VkDescriptorSetLayout dsl;
    CK(vkCreateDescriptorSetLayout(g_dev, &dli, nullptr, &dsl));   // pLayout set BEFORE pipeline create (the RADV landmine)
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    CK(vkCreatePipelineLayout(g_dev, &pli, nullptr, &p.layout));
    VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    st.stage = VK_SHADER_STAGE_COMPUTE_BIT; st.module = sm; st.pName = "main";
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = st; cpi.layout = p.layout;
    CK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipe));
    VkDescriptorPool dp; VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz;
    CK(vkCreateDescriptorPool(g_dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    CK(vkAllocateDescriptorSets(g_dev, &dai, &p.ds));
    VkDescriptorBufferInfo dbi{bound, 0, 64};
    VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr.dstSet = p.ds; wr.dstBinding = 0; wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(g_dev, 1, &wr, 0, nullptr);
    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpc.queueFamilyIndex = g_qf;
    VkCommandPool cp; CK(vkCreateCommandPool(g_dev, &cpc, nullptr, &cp));
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = cp; ca.commandBufferCount = 1;
    CK(vkAllocateCommandBuffers(g_dev, &ca, &p.cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(p.cb, &bi));
    vkCmdBindPipeline(p.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(p.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.ds, 0, nullptr);
    vkCmdDispatch(p.cb, 1, 1, 1);
    CK(vkEndCommandBuffer(p.cb));
    return p;
}

}  // namespace

int main() {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "strata-vk-doorbell";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    const char* iext[] = {"VK_KHR_portability_enumeration", "VK_KHR_get_physical_device_properties2"};
    ici.enabledExtensionCount = 2; ici.ppEnabledExtensionNames = iext;
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    CK(vkCreateInstance(&ici, nullptr, &g_inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(g_inst, &n, pds.data());
    VkPhysicalDeviceProperties props{};
    for (auto pd : pds) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd, &p);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs.data());
        bool compute = false;
        for (auto& q : qs) if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) { compute = true; break; }
        if (!compute) continue;
        g_phys = pd; props = p;
        for (uint32_t i = 0; i < qn; i++) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_qf = i; break; }
        break;
    }
    if (!g_phys) { std::fprintf(stderr, "doorbell: no compute device\n"); std::exit(2); }
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    CK(vkCreateDevice(g_phys, &dci, nullptr, &g_dev));
    vkGetDeviceQueue(g_dev, g_qf, 0, &g_queue);
    std::printf("doorbell: %s (RADV, probe case E)\n", props.deviceName);

    constexpr uint32_t kRing = 7u;

    // ---- arm 1: live host store into HOST_VISIBLE|HOST_COHERENT memory, mid-spin.
    Buf hb = make_buf(true);
    std::memset(hb.map, 0, 64);
    volatile uint32_t* hw = (volatile uint32_t*) hb.map;   // arm 4 reads what arm 1's ring left behind
    {
        Pipe p1 = make_pipe(hb.buf);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence f; CK(vkCreateFence(g_dev, &fci, nullptr, &f));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &p1.cb;
        CK(vkQueueSubmit(g_queue, 1, &si, f));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        hw[0] = kRing;                       // the host's doorbell ring, mid-kernel
        const auto rang = std::chrono::steady_clock::now();
        CK(vkWaitForFences(g_dev, 1, &f, VK_TRUE, ~0ull));
        const double after_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - rang).count();
        const bool visible = (hw[3] == 0x5A5A5A5Au) && (hw[1] == kRing) && (hw[4] == 0u);
        std::printf("doorbell arm1 (host store -> running kernel): flag=%u observed=%u reads=%u "
                    "sentinel=%08x timeout=%u — the spin ran %.1f ms past the ring (its full bound)\n",
                    hw[0], hw[1], hw[2], hw[3], hw[4], after_ms);
        std::printf("doorbell arm1: host stores to a RUNNING kernel are %s\n",
                    visible ? "VISIBLE" : "NEVER SEEN (the spin hit its 40M-read bound)");
    }

    // ---- arm 3 (one more live question the twins could route through): HOST_VISIBLE *uncached*
    // (write-combined on RADV: no HOST_CACHED, no HOST_COHERENT) + explicit vkFlushMappedMemoryRanges.
    {
        Buf wc{};
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = 64;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        CK(vkCreateBuffer(g_dev, &bci, nullptr, &wc.buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev, wc.buf, &mr);
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
        int32_t mt = -1;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i))
                && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                && !(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) { mt = (int)i; break; }
        if (mt >= 0) {
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = mr.size; mai.memoryTypeIndex = (uint32_t) mt;
            CK(vkAllocateMemory(g_dev, &mai, nullptr, &wc.mem));
            CK(vkBindBufferMemory(g_dev, wc.buf, wc.mem, 0));
            CK(vkMapMemory(g_dev, wc.mem, 0, VK_WHOLE_SIZE, 0, &wc.map));
            std::memset(wc.map, 0, 64);
            Pipe p3 = make_pipe(wc.buf);
            volatile uint32_t* ww = (volatile uint32_t*) wc.map;
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkFence f; CK(vkCreateFence(g_dev, &fci, nullptr, &f));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1; si.pCommandBuffers = &p3.cb;
            CK(vkQueueSubmit(g_queue, 1, &si, f));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            ww[0] = kRing;
            VkMappedMemoryRange fr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            fr.memory = wc.mem; fr.offset = 0; fr.size = 64;
            vkFlushMappedMemoryRanges(g_dev, 1, &fr);
            CK(vkWaitForFences(g_dev, 1, &f, VK_TRUE, ~0ull));
            const bool vis = (ww[3] == 0x5A5A5A5Au) && (ww[1] == kRing) && (ww[4] == 0u);
            std::printf("doorbell arm3 (uncached/WC host store + flush -> running kernel): flag=%u observed=%u "
                        "reads=%u sentinel=%08x timeout=%u -> %s\n",
                        ww[0], ww[1], ww[2], ww[3], ww[4], vis ? "VISIBLE" : "NEVER SEEN");
        } else {
            std::printf("doorbell arm3: no uncached HOST_VISIBLE memory type on this driver (skipped)\n");
        }
    }

    // ---- arm 4 (the shape a "waiter kernel launched after the ring" takes): the flag was stored into
    // the HOST_VISIBLE|HOST_COHERENT buffer during arm 1's dead spin; a NEW launch on the same buffer
    // must observe it — if it does, the twins' doorbell is "ring = host store, then launch the waiter"
    // (no copy stage needed); if not, every ring needs the explicit staged flush of arm 2.
    {
        Pipe p4 = make_pipe(hb.buf);            // hb.map[0] is already kRing from arm 1
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence f; CK(vkCreateFence(g_dev, &fci, nullptr, &f));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &p4.cb;
        CK(vkQueueSubmit(g_queue, 1, &si, f));
        CK(vkWaitForFences(g_dev, 1, &f, VK_TRUE, ~0ull));
        const bool vis = (hw[3] == 0x5A5A5A5Au) && (hw[1] == kRing);
        std::printf("doorbell arm4 (host-stored flag -> NEXT launch, same coherent buffer): observed=%u "
                    "reads=%u sentinel=%08x -> %s\n", hw[1], hw[2], hw[3],
                    vis ? "VISIBLE at next launch" : "STALE even for the next launch (needs the copy path)");
    }

    // ---- arm 2: the staged-flush handshake. The flag travels into a DEVICE_LOCAL buffer by
    // vkCmdCopyBuffer BEFORE the spin launches; the waiter must see it on its first reads.
    Buf db = make_buf(false);
    clear_buf(db);
    {
        Buf stage = make_buf(true);
        std::memset(stage.map, 0, 64);
        ((uint32_t*) stage.map)[0] = kRing;              // the flag, staged
        submit_copy(db.buf, stage.buf, 64);              // flushed to device memory before launch
        vkUnmapMemory(g_dev, stage.mem);
        vkDestroyBuffer(g_dev, stage.buf, nullptr);
        vkFreeMemory(g_dev, stage.mem, nullptr);
    }
    Pipe p2 = make_pipe(db.buf);
    VkFenceCreateInfo fci2{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence f2; CK(vkCreateFence(g_dev, &fci2, nullptr, &f2));
    VkSubmitInfo si2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si2.commandBufferCount = 1; si2.pCommandBuffers = &p2.cb;
    CK(vkQueueSubmit(g_queue, 1, &si2, f2));
    CK(vkWaitForFences(g_dev, 1, &f2, VK_TRUE, ~0ull));
    // read back through a host-coherent scratch
    Buf rb = make_buf(true);
    submit_copy(rb.buf, db.buf, 64);
    const uint32_t* o = (const uint32_t*) rb.map;
    std::printf("doorbell arm2 (staged flag -> spin at next launch): flag=%u observed=%u reads=%u "
                "sentinel=%08x timeout=%u\n", o[0], o[1], o[2], o[3], o[4]);
    const bool staged_ok = (o[3] == 0x5A5A5A5Au) && (o[1] == kRing);
    std::printf("doorbell arm2: staged-flush handshake %s\n", staged_ok ? "OK" : "BROKEN");
    std::printf("vk_doorbell_probe %s\n", staged_ok ? "PASS" : "FAIL");
    return staged_ok ? 0 : 1;
}
