// vulkan/src/runtime/vk_device.cpp - device selection and the facts the engine logs and plans against.
//
// The checks below are not decoration: the port's kernels are written against two device capabilities
// (VK_KHR_shader_integer_dot_product for the quantized dots, VK_EXT_subgroup_size_control to force the
// wave32 the CUDA kernels assume on wave64-era hardware like the BC-250's GFX1013). A device without them
// fails at start with a named reason instead of producing wrong tokens - the same rule device.cu applies
// to CUDA/HIP archs.
#include "strata/vulkan_runtime.hpp"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::vk {

uint32_t g_vk_qf = 0;

namespace {

#define VKCK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    throw std::runtime_error(std::string(#x) + " -> " + std::to_string((int)r_)); } } while (0)

VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_phys = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
uint32_t g_queue_family = 0;
DeviceInfo g_info;
bool g_init = false;

void init() {
    if (g_init) return;
    g_init = true;
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "strata";
    const char* iext[] = {"VK_KHR_portability_enumeration", "VK_KHR_get_physical_device_properties2"};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = iext;
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    VKCK(vkCreateInstance(&ici, nullptr, &g_instance));

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    if (n) vkEnumeratePhysicalDevices(g_instance, &n, pds.data());

    const char* want = std::getenv("STRATA_VK_DEVICE");   // substring match on the device name
    int chosen = -1;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &qn, qs.data());
        bool compute = false;
        for (auto& q : qs) if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) { compute = true; break; }
        if (!compute) continue;
        if (want && !std::strstr(p.deviceName, want)) continue;
        if (chosen < 0) chosen = (int)i;
    }
    if (chosen < 0)
        throw std::runtime_error(want ? std::string("no Vulkan device matching STRATA_VK_DEVICE='") + want + "'"
                                      : std::string("no Vulkan device with a compute queue"));
    g_phys = pds[chosen];

    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(g_phys, &p);

    // ---- the required extensions, queried before the device exists so the reason can be named.
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> eps(en);
    vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &en, eps.data());
    auto has = [&](const char* w) {
        for (auto& e : eps) if (!std::strcmp(e.extensionName, w)) return true;
        return false;
    };
    const bool idp = has("VK_KHR_shader_integer_dot_product");
    const bool ssc = has("VK_EXT_subgroup_size_control");
    if (!idp || !ssc) {
        std::string miss;
        if (!idp) miss += " VK_KHR_shader_integer_dot_product";
        if (!ssc) miss += " VK_EXT_subgroup_size_control";
        throw std::runtime_error(std::string("Vulkan device ") + p.deviceName + " lacks:" + miss +
                                 " - the port's quantized dots and wave32 subgroups need both");
    }

    VkPhysicalDeviceSubgroupSizeControlProperties sscp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sscp;
    vkGetPhysicalDeviceProperties2(g_phys, &p2);
    if (sscp.maxSubgroupSize < 32 || sscp.minSubgroupSize > 32)
        throw std::runtime_error(std::string(p.deviceName) + ": cannot run a 32-wide subgroup (min " +
                                 std::to_string(sscp.minSubgroupSize) + ", max " + std::to_string(sscp.maxSubgroupSize) + ")");

    // ---- device with the features ON (probe10's lesson: capabilities present but features not enabled crash
    // the driver at pipeline create; enable exactly what the kernels use).
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, qs.data());
    for (uint32_t i = 0; i < qn; i++) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_queue_family = i; break; }
    g_vk_qf = g_queue_family;
    // Task 4.2: the timestamp facts BEFORE the device exists — the feature means little if the
    // chosen queue family's timestamps aren't valid (BC-250: the compute family reports
    // timestampValidBits 64; the sparse family 0 — the per-family check is what separates a
    // working query pool from one that returns garbage).
    const uint32_t ts_bits = qs[g_queue_family].timestampValidBits;
    const bool ts_feature = p.limits.timestampComputeAndGraphics != 0;
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_queue_family; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    const char* dext[] = {"VK_KHR_shader_integer_dot_product", "VK_EXT_subgroup_size_control",
                          "VK_KHR_get_memory_requirements2"};
    VkPhysicalDeviceShaderIntegerDotProductFeatures idpf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeatures sscf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    idpf.shaderIntegerDotProduct = VK_TRUE;
    sscf.subgroupSizeControl = VK_TRUE;
    sscf.computeFullSubgroups = VK_TRUE;
    idpf.pNext = &sscf;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 3; dci.ppEnabledExtensionNames = dext;
    dci.pNext = &idpf;
    // Task 4.2: nothing to enable for timestamps — VkPhysicalDeviceLimits::
    // timestampComputeAndGraphics is a fact, not a feature bit (Vulkan 1.0 core), and RADV
    // writes CS timestamps for every submit. The per-family timestampValidBits check above
    // is the whole gate.
    VKCK(vkCreateDevice(g_phys, &dci, nullptr, &g_device));
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    // ---- the facts.
    g_info.ordinal = chosen;
    g_info.name = p.deviceName;
        VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    { VkPhysicalDeviceProperties2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; q2.pNext=&drv;
      vkGetPhysicalDeviceProperties2(g_phys,&q2); }
    g_info.driver = drv.driverName ? drv.driverName : "?";
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    sg.supportedStages = (VkShaderStageFlags)VK_SHADER_STAGE_COMPUTE_BIT;
    { VkPhysicalDeviceProperties2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; q2.pNext=&sg;
      vkGetPhysicalDeviceProperties2(g_phys,&q2); }
    g_info.subgroup_size = sg.subgroupSize;
    g_info.max_workgroup_invocations = p.limits.maxComputeWorkGroupInvocations;
        // NOT a driver limit: Vulkan exposes no maxAllocationSize field and the "512 MiB on RADV"
        // figure was a session-start guess, DISPROVEN on the BC-250 (probe: buffer allocations of
        // 512 MiB, 1, 2, 4, 8 GiB all succeed; 12 GiB fails on the memory budget - llama.cpp serves
        // the 11 GB model in big buffers here). The default is now the budget itself: arena_add
        // grows only within free_bytes (its own check) and halve-retries any piece the driver
        // refuses, so chaining is failure-driven, not a fixed cap. (The GSQ-hybrid bring-up: the
        // old 6 GiB default chained the 8.4 GiB expert cache and an expert row straddled the
        // boundary -> named straddle rejection on every mmvq binding of that row.) overridable
        // with STRATA_VK_MAX_ALLOC (the probe pins 512 MiB to keep exercising the piece-chain
        // paths deterministically).
        g_info.max_allocation_size = (16ull << 30);
    if (const char* e = getenv("STRATA_VK_MAX_ALLOC")) g_info.max_allocation_size = (uint64_t)atoll(e);
    g_info.integer_dot_product = idp;
    g_info.subgroup_size_control = ssc;
    g_info.buffer_device_address = has("VK_EXT_buffer_device_address");
    // Task 4.2: the timestamp facts (feature + this queue family's valid bits + the tick period).
    // The FINAL word is the runtime's: vk_runtime's event layer proves vkCmdWriteTimestamp
    // returns real ticks with its first event pair (vk_event_timing pins the same by execution).
    g_info.gpu_timestamps = ts_feature && ts_bits >= 32;
    g_info.timestamp_period_ns = p.limits.timestampPeriod;
    // ---- CU count (Task 1.3 step 5): VK_AMD_shader_core_properties through the properties2 chain —
    // queryable WITHOUT enabling the device extension (measured on this box: engines 2 x arrays/engine 2
    // x CUs/array 6 = 24). STRATA_VK_CU_OVERRIDE=N rewrites it for occupancy experiments (a BC-250 BIOS
    // unlock to 40 CUs is NOT visible through this query — the driver reports what its power hints say).
    {
        VkPhysicalDeviceShaderCorePropertiesAMD scp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD};
        VkPhysicalDeviceProperties2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        q2.pNext = &scp;
        vkGetPhysicalDeviceProperties2(g_phys, &q2);
        uint64_t cu = (uint64_t) scp.shaderEngineCount * scp.shaderArraysPerEngineCount *
                      scp.computeUnitsPerShaderArray;
        if (const char* e = getenv("STRATA_VK_CU_OVERRIDE")) cu = (uint64_t) atoll(e);
        g_info.multi_processor_count = cu > 0 && cu < 1024 ? (int) cu : 0;
    }
    g_info.l2_bytes = 0;   // no Vulkan query for it; radv exposes no L2 property (documented, not guessed)
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
    g_info.unified_memory = false;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        const auto vis = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        if ((mp.memoryTypes[i].propertyFlags & vis) == vis) { g_info.unified_memory = true; break; }
    }
    // VRAM figures, Vulkan 1.4's memory budget through the device properties2 chain.
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    mp2.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(g_phys, &mp2);
    uint64_t total = 0, free_b = 0;
    for (uint32_t h = 0; h < mp2.memoryProperties.memoryHeapCount; h++) {
        total += mp2.memoryProperties.memoryHeaps[h].size;
        const int64_t b = (int64_t)budget.heapBudget[h] - (int64_t)budget.heapUsage[h];
        if (b > 0) free_b += (uint64_t)b;
    }
    g_info.total_bytes = total;
    g_info.free_bytes = free_b;

    std::fprintf(stderr, "strata-vk: %s (%s), vulkan %u.%u.%u, subgroup %u (forced 32), %llu MiB total / %llu MiB free, dot=%d ssc=%d uma=%d\n",
                 g_info.name.c_str(), g_info.driver.c_str(),
                 VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion),
                 g_info.subgroup_size, (unsigned long long)(total >> 20), (unsigned long long)(free_b >> 20),
                 (int)idp, (int)ssc, (int)g_info.unified_memory);
}

}  // namespace

const DeviceInfo& device_info(int) { init(); return g_info; }

int device_count() {
    init();
    return 1;   // first cut: one device, like strata-device's HIP-visible count for an APU
}

void* vk_handle() { init(); return (void*)g_device; }
void* vk_physical_handle() { init(); return (void*)g_phys; }
void* vk_default_queue() { init(); return (void*)g_queue; }

}  // namespace strata::vk
