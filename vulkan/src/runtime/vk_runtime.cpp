// vulkan/src/runtime/vk_runtime.cpp - arena, copies, kernel registry and launch.
//
// An arena address is a byte offset into a chain of VkBuffers: the piece cap (DeviceInfo::
// max_allocation_size, 6 GiB by default on this box; the old "512 MiB on RADV" figure was a
// fictional guess, disproven by measurement - 8 GiB allocates fine, the limit is the memory
// budget), splits regions larger than a piece into several. arena_add() keeps EVERY region that
// fits one piece single-piece by construction (the cursor bumps past a short tail), so a kernel
// span computed from a region's own numbers can never straddle; only genuinely oversized regions
// chain.
// A kernel argument must land inside one piece (weights regions are laid out so that each tensor's planes
// do; a straddling span throws at launch with a named reason rather than reading wrong bytes). Copies walk
// the pieces. That keeps the first cut honest where CUDA's flat cudaMalloc address space is not available.
#include "strata/vulkan_runtime.hpp"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <execinfo.h>
#include <map>
#include <atomic>
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

uint64_t arena_add(uint64_t bytes) {   // returns the base the region was PLACED at (cursor
                                       // bump applied under g_m BEFORE placement - Task 4.1:
                                       // alloc() used to capture g_cursor before the bump, so a
                                       // concurrent alloc that ended a piece returned an offset
                                       // in the dead tail while the copy landed at the bumped
                                       // base -> straddled spans / garbage tensors under batching)
    VkDevice dev = (VkDevice)vk_handle();
    const DeviceInfo& d = device_info();
    uint64_t max_alloc = d.max_allocation_size ? d.max_allocation_size : (64ull << 20);
    const uint64_t slack = getenv("STRATA_VK_SLACK") ? (uint64_t)atoll(getenv("STRATA_VK_SLACK")) : (256ull << 20);
    // Piece-aligned regions (Task 2.11 law): a region that fits inside ONE max-allocation piece
    // never starts where a piece runs short - the cursor bumps to the next piece boundary and
    // every such allocation is single-piece by construction. This kills the whole class of
    // layout-dependent "kernel argument crosses an arena piece" traps: a kernel span computed
    // from the region's own numbers can then never straddle. Regions LARGER than max_alloc
    // still chain (piece_of throws for their internal sub-spans, unchanged) and the probe's
    // oversized-straddle case keeps that rejection. Wasted address space <= one piece per chain.
    if (bytes && bytes <= max_alloc && !g_pieces.empty()) {
        const Piece& last = g_pieces.back();
        const uint64_t room = last.base + last.size - g_cursor;
        if (room < bytes) {
            if (getenv("STRATA_VK_DEBUG_ARENA"))
                std::fprintf(stderr, "arena_add(%llu B): bump %llu -> %llu\n",
                             (unsigned long long)bytes, (unsigned long long)g_cursor,
                             (unsigned long long)(last.base + last.size));
            g_cursor = last.base + last.size;   // end the piece under the cursor
        }
    }
    const uint64_t placed_at = g_cursor;
    if (g_cursor + bytes + slack > d.free_bytes)
        throw std::runtime_error("vk_runtime: out of VRAM growing the arena by " + std::to_string(bytes >> 20) + " MiB");
    while (bytes) {
        // Budget-aware pieces: chain only when a piece of the requested size does not fit the
        // memory budget (the GSQ-hybrid bring-up: the engine's 8.4 GiB expert cache chained at the
        // old 6 GiB default and an expert row straddled the boundary -> every mmvq binding of that
        // row hit the named straddle rejection). Try sz first, halve on allocation failure;
        // max_allocation_size (env-overridable; the probe pins 512 MiB) stays the cap.
        uint64_t sz = bytes > max_alloc ? max_alloc : bytes;
        VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE;
        for (;;) {
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = sz;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (vkCreateBuffer(dev, &bci, nullptr, &buf) != VK_SUCCESS) {
                if (sz < (32ull << 20)) throw std::runtime_error("vkCreateBuffer: arena piece exhausted halving");
                sz /= 2; continue;
            }
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = mr.size;
            mai.memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (vkAllocateMemory(dev, &mai, nullptr, &mem) == VK_SUCCESS) break;
            vkDestroyBuffer(dev, buf, nullptr); buf = VK_NULL_HANDLE;
            if (sz < (32ull << 20)) throw std::runtime_error("vkAllocateMemory -> out of memory: arena piece exhausted halving");
            sz /= 2;
        }
        RT(vkBindBufferMemory(dev, buf, mem, 0));
        Piece p; p.base = g_cursor; p.size = sz; p.buf = buf; p.mem = mem;
        g_pieces.push_back(p);
        g_cursor += sz;
        bytes -= sz;
    }
    if (getenv("STRATA_VK_DEBUG_ARENA"))
        std::fprintf(stderr, "arena_add: placed_at %llu last_piece_end %llu slack_ok cursor_now %llu\n",
                     (unsigned long long)placed_at, (unsigned long long)g_pieces.back().base,
                     (unsigned long long)g_cursor);
    return placed_at;
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
std::mutex g_pm;   // guards g_pend/g_inflight AND pool/queue ops; ALWAYS the leaf lock
void ensure_cmd_pool() {
    if (g_pool) return;
    VkDevice dev = (VkDevice)vk_handle();
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = vk_queue_family();
    RT(vkCreateCommandPool(dev, &cpi, nullptr, &g_pool));
}
VkCommandBuffer begin() {
    VkDevice dev = (VkDevice)vk_handle();
    std::lock_guard<std::mutex> plk(g_pm);   // g_pool allocate is not thread-safe: the eager
    ensure_cmd_pool();                       // copy path and the batched path share this pool
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    RT(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    RT(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

// ---------- Phase 4.1: deferred-submit batching ----------
// kernel_launch_sized records into a PENDING primary command buffer (ONE_TIME_SUBMIT) with a
// per-pending descriptor POOL (cap kMaxSetsPerCB sets, pool recreated with the next pending
// buffer at flush — never a create/destroy per launch, which was a real slice of the old
// overhead). flush() = end + submit with a VkFence, NO wait; flush_wait() additionally waits
// every in-flight fence, then frees the command buffers and destroys the pools (a pool may
// only be reset/destroyed once its bound sets stop executing). Flush(+wait) points:
// stream_sync/device_sync/event_record, and the top of every copy/fill path.
// The staging hazard (why copies stay EAGER, not recorded): copy_chunk captures the SOURCE
// ADDRESS of g_stage into the command buffer, not its bytes — a later host memcpy into the
// same stage would corrupt a not-yet-submitted copy. So each copy first flush_waits all
// pending GPU work, then runs its own submit+vkQueueWaitIdle as before. The engine only
// copies at token boundaries and staging windows, so eager copies cost little.
// STRATA_VK_BATCH=0 restores the per-launch begin/submit/wait path (A/B toggle).
uint32_t max_sets_per_cb() {
    static uint32_t v = getenv("STRATA_VK_MAX_SETS") ? (uint32_t)atoi(getenv("STRATA_VK_MAX_SETS")) : 512u;
    return v;   // descriptor sets (== dispatches) per pending CB
}
constexpr uint32_t kMaxBufsPerSet = 64;      // pool sizing headroom; kernels use far fewer
uint32_t max_inflight() {   // queued-but-unreaped CBs before pend_flush waits the oldest
    static uint32_t v = getenv("STRATA_VK_MAX_INFLIGHT") ? (uint32_t)atoi(getenv("STRATA_VK_MAX_INFLIGHT")) : 8u;
    return v;
}
struct PendingCB { VkCommandBuffer cb = VK_NULL_HANDLE; VkDescriptorPool dp = VK_NULL_HANDLE;
                   uint32_t sets = 0; };

struct InFlight { VkFence fence; VkCommandBuffer cb; VkDescriptorPool dp; };
PendingCB g_pend;
std::vector<InFlight> g_inflight;
struct ClusterSpan { uint64_t off, bytes; bool wr; };
std::vector<ClusterSpan> g_cluster;   // barrier-free cluster spans of the pending CB
long long g_cluster_saves = 0;        // barriers elided (probe)
uint32_t batch_on() {
    static uint32_t v = getenv("STRATA_VK_BATCH") ? (uint32_t)atoi(getenv("STRATA_VK_BATCH")) : 1u;
    return v;
}
void pend_begin_locked() {   // g_pm held
    VkDevice dev = (VkDevice)vk_handle();
    ensure_cmd_pool();
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = g_pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    RT(vkAllocateCommandBuffers(dev, &cai, &g_pend.cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    RT(vkBeginCommandBuffer(g_pend.cb, &bi));
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_sets_per_cb() * kMaxBufsPerSet};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = max_sets_per_cb(); dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz;
    RT(vkCreateDescriptorPool(dev, &dpi, nullptr, &g_pend.dp));
    g_pend.sets = 0;
    g_cluster.clear();   // new CB: no cluster to overlap (or to fence off)
}
// Prefill-tuning probe counters (STRATA_VK_LAUNCH_TIME=1; printed by the launch dump).
std::map<std::string, long long> g_launch_names;
long long g_flush_calls = 0; double g_flush_ms = 0.0;
long long g_pendwait_calls = 0; double g_pendwait_ms = 0.0;   // depth-cap fence waits in pend_flush
void pend_flush_locked() {   // g_pm held; end+submit with a fence, NO wait
    if (!g_pend.cb) return;
    VkDevice dev = (VkDevice)vk_handle();
    // depth cap: never leave more than kMaxInFlight CBs queued without reaping, so a 128-token
    // prefill can't pile up command buffers (and their pools) faster than the GPU drains.
    if (g_inflight.size() >= max_inflight()) {
        const auto pw0 = std::chrono::steady_clock::now();
        RT(vkWaitForFences(dev, 1, &g_inflight.front().fence, VK_TRUE, ~0ull));
        if (getenv("STRATA_VK_LAUNCH_TIME")) { g_pendwait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pw0).count(); g_pendwait_calls++; }
        vkDestroyFence(dev, g_inflight.front().fence, nullptr);
        vkFreeCommandBuffers(dev, g_pool, 1, &g_inflight.front().cb);
        vkDestroyDescriptorPool(dev, g_inflight.front().dp, nullptr);
        g_inflight.erase(g_inflight.begin());
    }
    RT(vkEndCommandBuffer(g_pend.cb));
    VkFence f; VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    RT(vkCreateFence(dev, &fi, nullptr, &f));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &g_pend.cb;
    RT(vkQueueSubmit((VkQueue)vk_default_queue(), 1, &si, f));
    g_inflight.push_back(InFlight{f, g_pend.cb, g_pend.dp});
    g_pend = PendingCB{};
}
void flush_wait_why(const char* why) {
    // STRATA_VK_FLUSH_WHY=1: count drains by call site (the 4.2-slim measurement: WHO drains).
    static const bool on = getenv("STRATA_VK_FLUSH_WHY") != nullptr;
    if (!on) return;
    static std::map<std::string, long long> counts;
    static struct Dump { ~Dump() {
        for (auto& kv : counts) std::fprintf(stderr, "  flush_wait %s: %lld\n", kv.first.c_str(), kv.second); } } dump;
    counts[why]++;
}
}  // namespace (the pending-CB helpers above; the graph layer below gets strata::vk linkage)
// ---------- Phase 4.2: graph capture (the plan's Phase 4; replaces the compat graph traps) ----------
// A captured graph is ONE pre-recorded command buffer (multi-submit usage) plus the descriptor
// pool its sets were allocated from (kept alive for the graph's life; never reset). BeginCapture
// opens it; launches record into it exactly like the pending path but with the OLD barrier law
// (full-RW after EVERY dispatch - no cluster elision while capturing: bit-safe by construction,
// the eager pre-4.1 ordering replayed as one submit). Copies/events/fills during capture ABORT
// loudly (a drain during capture is illegal; a silently skipped copy would replay nothing).
// EndCapture closes the CB; GraphLaunch submits it (waiting the graph's own previous fence
// first: CUDA's stream semantics serialize replays, and Vulkan forbids re-submitting a CB that
// is still executing). Instantiate is the capture close point in the engine's flow (EndCapture
// then Instantiate then destroy graph): the graph HANDLE carries the CB/pool until instantiate
// moves it to the exec handle.
// pools chain: one small pool per 128 sets (a typical layer graph records tens of dispatches);
// the sets must outlive the CB's record, the CB must stay valid for replays: pools free only at
// graph destroy. (A big shared pool preallocates descriptors RADV never releases; measured cost
// would be hundreds of MB across ~1,600 layer graphs.)
constexpr uint32_t kGraphPoolSets = 128;
struct GraphObj { VkCommandBuffer cb = VK_NULL_HANDLE; std::vector<VkDescriptorPool> dps;
                  uint32_t sets_left = 0; uint32_t dispatches = 0; bool open = false; };
static GraphObj g_cap;
static std::vector<GraphObj> g_graphs;        // instantiated graphs (exec handles index these)
static std::vector<VkFence> g_graph_fence;    // one per exec: the LAST replay of that graph
static std::vector<bool> g_graph_fence_live;
static bool g_capturing = false;
uint32_t capture_dispatches() { return g_graphs.size(); }

void graph_begin() {
    std::lock_guard<std::mutex> plk(g_pm);
    if (g_capturing) throw std::runtime_error("vk_runtime: capture already open");
    pend_flush_locked();                     // prior eager work is submitted first: capture starts clean
    VkDevice dev = (VkDevice)vk_handle();
    ensure_cmd_pool();
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = g_pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    RT(vkAllocateCommandBuffers(dev, &cai, &g_cap.cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = 0;   // NOT one-time: the graph replays this CB many times
    RT(vkBeginCommandBuffer(g_cap.cb, &bi));
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphPoolSets * 16};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = kGraphPoolSets; dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz;
    VkDescriptorPool p0; RT(vkCreateDescriptorPool(dev, &dpi, nullptr, &p0));
    g_cap.dps.push_back(p0); g_cap.sets_left = kGraphPoolSets;
    g_cap.dispatches = 0; g_cap.open = true;
    g_capturing = true;
}
uint32_t graph_end() {   // returns the recorded dispatch count; registers a graph object
    std::lock_guard<std::mutex> plk(g_pm);
    if (!g_capturing) throw std::runtime_error("vk_runtime: graph_end with no capture open");
    RT(vkEndCommandBuffer(g_cap.cb));
    g_graphs.push_back(g_cap);
    g_graph_fence.push_back(VK_NULL_HANDLE);
    g_graph_fence_live.push_back(false);
    const uint32_t d = g_cap.dispatches;
    g_cap = GraphObj{}; g_capturing = false;
    return d;
}
void graph_destroy(uint32_t gi) {
    std::lock_guard<std::mutex> plk(g_pm);
    if (gi >= g_graphs.size()) return;
    VkDevice dev = (VkDevice)vk_handle();
    if (g_graph_fence[gi] != VK_NULL_HANDLE) {
        RT(vkWaitForFences(dev, 1, &g_graph_fence[gi], VK_TRUE, ~0ull));
        vkDestroyFence(dev, g_graph_fence[gi], nullptr);
    }
    for (auto dp : g_graphs[gi].dps) vkDestroyDescriptorPool(dev, dp, nullptr);
    g_graphs[gi].dps.clear();
    vkFreeCommandBuffers(dev, g_pool, 1, &g_graphs[gi].cb);
    g_graphs[gi] = GraphObj{};
}
void graph_launch(uint32_t gi) {
    std::lock_guard<std::mutex> plk(g_pm);
    if (!g_capturing && g_pend.cb) pend_flush_locked();   // eager work first: stream order
    if (gi >= g_graphs.size() || !g_graphs[gi].cb) throw std::runtime_error("vk_runtime: graph launch of an unregistered/destroyed graph");
    VkDevice dev = (VkDevice)vk_handle();
    // CUDA stream semantics: this replay runs after the graph's previous replay completed.
    if (g_graph_fence_live[gi]) {
        RT(vkWaitForFences(dev, 1, &g_graph_fence[gi], VK_TRUE, ~0ull));
        vkDestroyFence(dev, g_graph_fence[gi], nullptr);
    }
    VkFence f; VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    RT(vkCreateFence(dev, &fi, nullptr, &f));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &g_graphs[gi].cb;
    RT(vkQueueSubmit((VkQueue)vk_default_queue(), 1, &si, f));
    g_graph_fence[gi] = f; g_graph_fence_live[gi] = true;
}
void graph_sync_all() {   // device_sync/stream_sync also drain graph fences
    std::lock_guard<std::mutex> plk(g_pm);
    VkDevice dev = (VkDevice)vk_handle();
    for (size_t i = 0; i < g_graph_fence.size(); ++i) if (g_graph_fence_live[i]) {
        vkWaitForFences(dev, 1, &g_graph_fence[i], VK_TRUE, ~0ull);
        vkDestroyFence(dev, g_graph_fence[i], nullptr);
        g_graph_fence[i] = VK_NULL_HANDLE; g_graph_fence_live[i] = false;
    }
}
bool capturing() { return g_capturing; }
uint32_t graph_dispatches(uint32_t gi) { std::lock_guard<std::mutex> plk(g_pm);
    return gi < g_graphs.size() ? g_graphs[gi].dispatches : 0; }
uint32_t capture_graph_count() { std::lock_guard<std::mutex> plk(g_pm); return (uint32_t)g_graphs.size(); }

// (back to the file's anonymous namespace for the flush/launch paths below, which use the
// graph state above and the g_* state further down; declared-visible since they are in this TU)
namespace {
void flush_wait() {          // flush pending, then wait ALL in-flight fences (all queue ops under g_pm)
    if (g_capturing) {       // a sync point inside a capture body: recorded as-is would reorder;
                             // silently dropping it would replay wrong. Loud trap (the port's law).
        std::fprintf(stderr, "strata-vk: flush_wait inside a graph capture - the capture body hit a sync/copy site; refusing\n");
        std::abort();
    }
    // stall time spent draining the queue:
    static const bool ft = getenv("STRATA_VK_LAUNCH_TIME") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> plk(g_pm);
    // graph replays in flight belong to the same stream: drain their fences too
    for (size_t i = 0; i < g_graph_fence.size(); ++i) if (g_graph_fence_live[i]) {
        RT(vkWaitForFences((VkDevice)vk_handle(), 1, &g_graph_fence[i], VK_TRUE, ~0ull));
        vkDestroyFence((VkDevice)vk_handle(), g_graph_fence[i], nullptr);
        g_graph_fence[i] = VK_NULL_HANDLE; g_graph_fence_live[i] = false;
    }
    pend_flush_locked();
    VkQueue q = (VkQueue)vk_default_queue();
    RT(vkQueueWaitIdle(q));   // drains EVERYTHING (incl. other threads' recorded work)
    if (ft) g_flush_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ft) g_flush_calls++;
    if (g_inflight.empty()) return;
    VkDevice dev = (VkDevice)vk_handle();
    for (auto& x : g_inflight) {
        vkDestroyFence(dev, x.fence, nullptr);
        vkFreeCommandBuffers(dev, g_pool, 1, &x.cb);
        vkDestroyDescriptorPool(dev, x.dp, nullptr);
    }
    g_inflight.clear();
}
void end_submit(VkCommandBuffer cb) {
    // ONE g_pm critical section: pool-free and vkQueueSubmit/vkQueueWaitIdle are not thread-safe;
    // with batching, background swap threads (copies) and the main thread (launch flushes)
    // genuinely interleave here.
    std::lock_guard<std::mutex> plk(g_pm);
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
    std::string name;    // debug: which launcher threw the piece error
};
std::unordered_map<uint64_t, KernelImpl> g_kernels;    // keyed by pipeline handle

}  // namespace

DevicePtr Arena::alloc(uint64_t bytes) {
    std::lock_guard<std::mutex> lk(g_m);
    // Task 4.1 fix: the base MUST come from after arena_add()'s piece-boundary bump; the old
    // `at = g_cursor` pre-capture handed the caller a dead-tail offset whenever a concurrent
    // alloc had ended the piece under the cursor.
    return arena_add(bytes);
}
uint64_t Arena::used() const { std::lock_guard<std::mutex> lk(g_m); return g_cursor; }
uint64_t Arena::capacity() const { std::lock_guard<std::mutex> lk(g_m); return g_cursor; }
void* Arena::vk_device() const { return vk_handle(); }
void Arena::buffers_for(uint64_t, uint64_t, std::vector<Extent>&) const {}
Arena::Arena() {}
Arena::~Arena() {}
Arena& Arena::instance() { static Arena a; return a; }

// streams: the device has one queue, but native_mmvq's validate_stream() demands non-null stream
// handles from the engine's cudaStreamCreateWithFlags sites - hand out distinct tokens so host code
// keeps its CUDA-shaped discipline; all of them are the same queue underneath.
std::atomic<uint64_t> g_stream_token{1};
void* stream_create() { return (void*)(uintptr_t) g_stream_token.fetch_add(1); }
void stream_destroy(void*) {}
void stream_sync(void*) { flush_wait_why("stream_sync"); flush_wait(); }
void device_sync() { std::lock_guard<std::mutex> plk(g_pm);
                     for (size_t i = 0; i < g_graph_fence.size(); ++i) if (g_graph_fence_live[i]) {
                         RT(vkWaitForFences((VkDevice)vk_handle(), 1, &g_graph_fence[i], VK_TRUE, ~0ull));
                         vkDestroyFence((VkDevice)vk_handle(), g_graph_fence[i], nullptr);
                         g_graph_fence[i] = VK_NULL_HANDLE; g_graph_fence_live[i] = false;
                     }
                     pend_flush_locked();
                     RT(vkDeviceWaitIdle((VkDevice)vk_handle()));
                     VkDevice dev = (VkDevice)vk_handle();
                     for (auto& x : g_inflight) { vkDestroyFence(dev, x.fence, nullptr);
                         vkFreeCommandBuffers(dev, g_pool, 1, &x.cb);
                         vkDestroyDescriptorPool(dev, x.dp, nullptr); }
                     g_inflight.clear(); }

struct Ev { };
void* event_create() { return new Ev{}; }
void event_destroy(void* e) { delete (Ev*)e; }
// host-clock timing: the queue must be idle when the event is recorded
void event_record(void*, void*) { flush_wait_why("event_record"); flush_wait(); }
void event_wait(void*, void*) {}
float event_elapsed(void*, void*) { return 0.f; }

void copy_h2d(void* dst, const void* src, uint64_t bytes, void*) {
    if (getenv("STRATA_VK_FLUSH_WHY")) {   // (size, offset) histogram of eager h2d copies (probe only)
        static std::map<std::pair<uint64_t,uint64_t>, long long> hist;
        static long long g_bt_done = 0;
        static struct HD { ~HD() {
            std::fprintf(stderr, "  copy_h2d (size, dst) pairs (top 12):\n");
            std::vector<std::pair<std::pair<uint64_t,uint64_t>, long long>> v(hist.begin(), hist.end());
            std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.second > b.second; });
            for (size_t i = 0; i < v.size() && i < 12; ++i)
                std::fprintf(stderr, "    %8llu B  dst +%llu: %lld\n", (unsigned long long)v[i].first.first,
                             (unsigned long long)v[i].first.second, (long long)v[i].second); } } hd;
        hist[{bytes, (uint64_t)(uintptr_t)dst}]++;
        if (bytes == 294912 && g_bt_done < 3) {   // who: one backtrace per site, 3 max
            g_bt_done++;
            void* bt[16]; int n = backtrace(bt, 16);
            std::fprintf(stderr, "copy_h2d 288KiB backtrace:\n");
            backtrace_symbols_fd(bt, n, 2);
        }
    }
    std::lock_guard<std::mutex> lk(g_m);
    flush_wait_why("copy_h2d"); flush_wait();   // eager-copy rule: never let a stage write race a recorded stage copy
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
    flush_wait_why("copy_d2h"); flush_wait();   // the d2h result is what the sampler needs: right sync point
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
    if (g_capturing) {   // Phase 4.2: fills ARE recordable (vkCmdFillBuffer, the CUDA
                         // cudaMemsetAsync analogue): no staging, no drain. Chunk by piece.
        std::lock_guard<std::mutex> lk(g_m);       // pieces: g_m, then g_pm (the leaf), matching
        std::lock_guard<std::mutex> caplk(g_pm);   // the launch path's order
        VkDevice dev = (VkDevice)vk_handle();
        const uint32_t pat = uint32_t(byte) * 0x01010101u;
        uint64_t at = (uint64_t)(uintptr_t)dst, off = 0;
        while (off < bytes) {
            uint64_t n = bytes - off;
            if (uint64_t room = piece_room(at + off)) { if (n > room) n = room; }
            auto it = std::find_if(g_pieces.begin(), g_pieces.end(), [&](const Piece& p){
                return p.base <= at + off && at + off + n <= p.base + p.size; });
            if (it == g_pieces.end()) throw std::runtime_error("vk_runtime: fill crosses an arena piece");
            vkCmdFillBuffer(g_cap.cb, it->buf, (at + off) - it->base, n, pat);
            off += n;
        }
        return;
    }
    std::lock_guard<std::mutex> lk(g_m);
    flush_wait_why("fill_d"); flush_wait();   // eager-copy rule (fill writes the shared stage too)
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
    KernelImpl impl{pipe, pl, dsl, n_storage_buffers, push_constant_bytes, workgroup_size, name};
    g_kernels[(uint64_t)pipe] = impl;
    return {pipe, pl};
}

// `buffers` are arena offsets; `sizes` the same length, each span required to sit inside one arena piece.
void kernel_launch_sized(const Kernel& k, int groups_x, const uint64_t* buffers, const uint64_t* sizes,
                         int n_buffers, const void* push_constants) {
    // The conservative default: every bound buffer counts as written, so the launch always
    // barriers against the pending cluster (the pre-elision law).
    kernel_launch_sized_rw(k, groups_x, buffers, sizes, n_buffers >= 32 ? ~0u : ((1u << n_buffers) - 1u),
                           n_buffers, push_constants);
}

// Cluster tracking for barrier elision (see the header): the members recorded in the PENDING CB
// since the last emitted barrier. rw_mask bit i = kernel WRITES buffer i (it may still read it;
// every bound buffer counts as a read for interference). A new launch barriers first only when
// it provably touches (RAW/WAR/WAW) some cluster member's span; else it records with no barrier
// and RADV may overlap the dispatches. Queue submission ends each CB, so cluster state resets
// with every fresh pending CB.
namespace {
bool spans_overlap(const ClusterSpan& a, const ClusterSpan& b) {
    return a.off < b.off + b.bytes && b.off < a.off + a.bytes;
}
bool interferes(const uint64_t* buffers, const uint64_t* sizes, uint32_t rw_mask, int n_buffers) {
    // Interference is PER PAIR: an overlapping span pair is a hazard iff at least one side
    // WRITES it (RAW: cluster write vs new read; WAR: new write vs cluster read; WAW).
    // Read-read overlap (the same weight tensor through consecutive launches) is benign and
    // must NOT barrier, or the elision buys nothing on the mmvq fan-out path.
    for (const ClusterSpan& c : g_cluster)
        for (int i = 0; i < n_buffers; ++i) {
            const ClusterSpan s{buffers[i], sizes[i], (rw_mask >> i) & 1u};
            if (!c.wr && !s.wr) continue;
            if (spans_overlap(c, s)) return true;
        }
    return false;
}
void cluster_add(const uint64_t* buffers, const uint64_t* sizes, uint32_t rw_mask, int n_buffers) {
    for (int i = 0; i < n_buffers; ++i)
        g_cluster.push_back(ClusterSpan{buffers[i], sizes[i], (rw_mask >> i) & 1u});
}
void emit_full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}
}  // namespace

void kernel_launch_sized_rw(const Kernel& k, int groups_x, const uint64_t* buffers, const uint64_t* sizes,
                            uint32_t rw_mask, int n_buffers, const void* push_constants) {
    // Prefill-tuning probe (STRATA_VK_LAUNCH_TIME=1): host time spent inside the launch path
    // and the total dispatch count. Env-gated chrono only; no behavior change.
    static const bool launch_time = getenv("STRATA_VK_LAUNCH_TIME") != nullptr;
    static long long launch_calls = 0;
    static double launch_ms = 0.0;
    static struct LaunchDump { ~LaunchDump() {
        if (launch_calls) std::fprintf(stderr, "=== kernel_launch_sized: calls=%lld host=%.1fms | barriers elided=%lld | flush_wait: calls=%lld stall=%.1fms | depth-cap waits: %lld / %.1fms ===\n",
                                       launch_calls, launch_ms, g_cluster_saves, g_flush_calls, g_flush_ms, g_pendwait_calls, g_pendwait_ms);
        for (auto& kv : g_launch_names)
            std::fprintf(stderr, "  launch %s: %lld\n", kv.first.c_str(), kv.second); } } g_launch_dump;
    struct LaunchTimer {
        std::chrono::steady_clock::time_point t0; bool on;
        LaunchTimer() : on(launch_time) { if (on) t0 = std::chrono::steady_clock::now(); }
        ~LaunchTimer() {
            if (!on) return;
            launch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (++launch_calls % 50000 == 0)
                std::fprintf(stderr, "vk launch host: calls=%lld ms=%.1f\n", launch_calls, launch_ms);
        }
    } lt;
    std::lock_guard<std::mutex> lk(g_km());
    auto it = g_kernels.find((uint64_t)k.pipeline);
    if (it == g_kernels.end()) throw std::runtime_error("vk_runtime: launch of an unloaded kernel");
    const KernelImpl& impl = it->second;
    if (launch_time) g_launch_names[impl.name]++;
    if (n_buffers != impl.n_buffers) throw std::runtime_error("vk_runtime: kernel buffer count mismatch");
    VkDevice dev = (VkDevice)vk_handle();
    // hot path (789k launches in a 735-tok prefill): no heap alloc below the common buffer count
    VkDescriptorBufferInfo dbi_small[32];
    std::vector<VkDescriptorBufferInfo> dbi_big;
    VkDescriptorBufferInfo* dbi = nullptr;
    if (n_buffers <= 32) dbi = dbi_small;
    else { dbi_big.resize(n_buffers); dbi = dbi_big.data(); }
    std::lock_guard<std::mutex> alk(g_m);   // g_pieces is grown under g_m (Arena::alloc); with
                                            // deferred submits, launches now genuinely overlap
                                            // background arena growth - read pieces under the lock.
    for (int i = 0; i < n_buffers; i++) {
        uint64_t po; VkBuffer pb = piece_of(buffers[i], sizes[i], po);
        if (!pb) {
            std::string pieces = "pieces:";
            for (auto& p : g_pieces)
                if (p.base + p.size > buffers[i] && p.base < buffers[i] + sizes[i])
                    pieces += " [" + std::to_string(p.base) + "," + std::to_string(p.base + p.size) + "]";
            if (getenv("STRATA_VK_DEBUG_ARENA")) {
                for (int d = 0; d < n_buffers; d++)
                    std::fprintf(stderr, "  arg %d: off %llu size %llu\n", d,
                                 (unsigned long long)buffers[d], (unsigned long long)sizes[d]);
                std::fprintf(stderr, "  all pieces:\n");
                for (auto& p : g_pieces)
                    std::fprintf(stderr, "    [%llu,%llu)\n", (unsigned long long)p.base,
                                 (unsigned long long)(p.base + p.size));
            }
            throw std::runtime_error("vk_runtime: kernel argument " + std::to_string(i) +
                                     " of " + impl.name + " crosses an arena piece (offset " + std::to_string(buffers[i]) +
                                     ", " + std::to_string(sizes[i]) + " bytes; " + pieces + ")");
        }
        dbi[i] = VkDescriptorBufferInfo{pb, po, sizes[i]};
    }
    if (!batch_on()) {   // A/B fallback: the old per-launch pool + submit + wait
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
        return;
    }
    // Batched path: allocate the set from the PENDING pool (one per pending CB, recycled at
    // flush) and record bind+push+dispatch into the pending CB. When the CB hits the set cap,
    // end/submit it with a fence (no wait) and open a fresh CB+pool pair.
    if (g_capturing) {
        // capture path: record into the graph CB, its own pool, the OLD per-dispatch barrier law
        std::lock_guard<std::mutex> caplk(g_pm);   // leaf lock, same order as the pending path
        VkDescriptorSet ds; VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        if (g_cap.sets_left == 0) {   // next pool in the chain
            VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kGraphPoolSets * 16};
            VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dpi.maxSets = kGraphPoolSets; dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz;
            VkDescriptorPool p; RT(vkCreateDescriptorPool(dev, &dpi, nullptr, &p));
            g_cap.dps.push_back(p); g_cap.sets_left = kGraphPoolSets;
        }
        dai.descriptorPool = g_cap.dps.back(); dai.descriptorSetCount = 1; dai.pSetLayouts = &impl.dsl;
        RT(vkAllocateDescriptorSets(dev, &dai, &ds));
        g_cap.sets_left--;
        std::vector<VkWriteDescriptorSet> wrc(n_buffers);
        for (int i = 0; i < n_buffers; i++) {
            wrc[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wrc[i].dstSet = ds; wrc[i].dstBinding = (uint32_t)i; wrc[i].descriptorCount = 1;
            wrc[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wrc[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(dev, (uint32_t)n_buffers, wrc.data(), 0, nullptr);
        vkCmdBindPipeline(g_cap.cb, VK_PIPELINE_BIND_POINT_COMPUTE, impl.pipe);
        vkCmdBindDescriptorSets(g_cap.cb, VK_PIPELINE_BIND_POINT_COMPUTE, impl.layout, 0, 1, &ds, 0, nullptr);
        if (impl.pc_bytes && push_constants)
            vkCmdPushConstants(g_cap.cb, impl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, impl.pc_bytes, push_constants);
        vkCmdDispatch(g_cap.cb, (uint32_t)groups_x, 1, 1);
        emit_full_barrier(g_cap.cb);        // the eager ordering, frozen: every dispatch fenced off
        g_cap.dispatches++;
        return;
    }
    std::lock_guard<std::mutex> plk(g_pm);
    if (!g_pend.cb || g_pend.sets >= max_sets_per_cb()) {
        pend_flush_locked();      // submit the full CB (if any), fence, no wait
        pend_begin_locked();      // fresh CB + fresh descriptor pool
    }
    VkDescriptorSet ds; VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g_pend.dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &impl.dsl;
    RT(vkAllocateDescriptorSets(dev, &dai, &ds));
    std::vector<VkWriteDescriptorSet> wr(n_buffers);
    for (int i = 0; i < n_buffers; i++) {
        wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wr[i].dstSet = ds; wr[i].dstBinding = (uint32_t)i; wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, (uint32_t)n_buffers, wr.data(), 0, nullptr);
    vkCmdBindPipeline(g_pend.cb, VK_PIPELINE_BIND_POINT_COMPUTE, impl.pipe);
    vkCmdBindDescriptorSets(g_pend.cb, VK_PIPELINE_BIND_POINT_COMPUTE, impl.layout, 0, 1, &ds, 0, nullptr);
    if (impl.pc_bytes && push_constants)
        vkCmdPushConstants(g_pend.cb, impl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, impl.pc_bytes, push_constants);
    // In CB, consecutive dispatches carry NO declared dependency: RADV may overlap their
    // storage-buffer reads/writes (eager mode's per-launch wait was an implicit barrier). The
    // safe default inserts a full memory barrier after EVERY dispatch. Prefill spends ~12us
    // of that barrier per dispatch (500k+ dispatches), so: when the launch comes through
    // kernel_launch_sized_rw with a truthful write mask AND its spans provably do not touch
    // any member of the pending barrier-free cluster (RAW/WAR/WAW span test), record it with
    // NO barrier and let the cluster overlap. Any touching launch first flushes a barrier
    // covering the whole cluster (the cluster's writes are then globally visible to it and to
    // everything after), then records alone. STRATA_VK_NO_ELIDE=1 restores the per-dispatch law
    // (A/B). Cluster state is per-pending-CB: pend_begin_locked resets it (a submit fences the
    // recorded work off from what follows).
    static const bool no_elide = getenv("STRATA_VK_NO_ELIDE") != nullptr;
    // cap the cluster so the per-launch interference test stays O(1) amortized: at the cap,
    // close it with a barrier even when this launch would have overlapped.
    static const uint32_t cluster_cap = getenv("STRATA_VK_CLUSTER_CAP")
                                           ? (uint32_t)atoi(getenv("STRATA_VK_CLUSTER_CAP")) : 128u;
    if (!no_elide && g_cluster.size() < cluster_cap && !interferes(buffers, sizes, rw_mask, n_buffers)) {
        // Provably no RAW/WAR/WAW with any cluster member: overlap. (A read-read span overlap
        // is fine; so are disjoint worksets, the whole point of the mmvq-per-expert fan-out.)
        vkCmdDispatch(g_pend.cb, (uint32_t)groups_x, 1, 1);
        cluster_add(buffers, sizes, rw_mask, n_buffers);
        if (launch_time) g_cluster_saves++;
    } else {
        // Interferes (or conservative/no-elide): CLOSE the cluster first — the barrier between
        // the cluster's dispatches and this one is what makes the ordering safe — then this
        // dispatch starts a new cluster.
        if (!g_cluster.empty()) emit_full_barrier(g_pend.cb);
        g_cluster.clear();
        vkCmdDispatch(g_pend.cb, (uint32_t)groups_x, 1, 1);
        cluster_add(buffers, sizes, rw_mask, n_buffers);
    }
    g_pend.sets++;
}

}  // namespace strata::vk
