// strata_vk.h - the CUDA runtime names Strata's HOST sources use, mapped onto the Vulkan runtime
// (vulkan/src/runtime/{vk_device,vk_runtime}.cpp) for the BC-250 port.
//
// This directory goes FIRST on the include path of a Vulkan build only (vulkan/CMakeLists.txt,
// strata_vk_compat BEFORE); its `cuda_runtime.h`, `cuda.h`, `cublas_v2.h`, ... include this header instead of
// the NVIDIA ones, so the engine's sources stay one tree.  The shape follows
// include/strata/platform/hip_compat/strata_hip.h; the mappings themselves go onto strata::vk, not onto a
// vendor runtime, because there is no CUDA runtime here at all.
//
// THE DEVICE-POINTER CONTRACT (this is what lets 99 host files compile unchanged):
// a "device pointer" is an offset in the arena's address space (strata::vk::DevicePtr) plus a fixed 64-byte
// bias, cast to void*.  The bias exists because the arena's first offset IS 0 and host code legitimately
// tests freshly cudaMalloc'ed pointers against nullptr - an unbiased first allocation would read as a failure.
// Every compat entry point that touches the arena subtracts the bias back (strata_vk_off).  Kernel launchers
// taking a host-side void* device pointer must apply the same helper (or take strata_vk_offset()).
//
// THE ALLOCATION CONTRACT mirrors src/core/device.hpp's DeviceArena: bump allocation, no frees.  cudaFree is
// a no-op (the same --no-free rule the CUDA build runs under); growth happens piece-by-piece because RADV
// caps one VkDeviceMemory at maxMemoryAllocationSize (512 MiB on this box).  A copy whose span straddles a
// piece throws a NAMED reason in the runtime; it arrives here as cudaErrorUnknown with that message - the
// planner's regions must be sized to pieces (recorded in PORT-NOTES).
//
// TRAPS: the cudaGraph* family aborts with a printed name ("run with --no-graphs") rather than pretend -
// Phase 4 of the plan replaces them with recorded command buffers.  Ambiguous-semantics calls return an
// error that NAMES the gap instead of silently faking it.
#pragma once

#include "strata/vulkan_runtime.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// The engine's host files carry `#if CUDART_VERSION >= ...` capability gates (dma_batch's CUDA-13 batched
// copy path, verify's 12.3 node introspection).  Vulkan is not CUDA 13; claim 12.4 so the newer gates
// compile OUT and the code paths we actually implement stay in.
#define CUDART_VERSION 12040

// ---- error state ----------------------------------------------------------------------------------------
enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorUnknown = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidValue = 11,
    cudaErrorNotReady = 600,
    cudaErrorNotSupported = 801,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorStreamCaptureUnsupported = 900,
};

namespace strata_vk_compat {

struct ErrState { cudaError_t code = cudaSuccess; std::string msg; };
inline ErrState& last_error() { static thread_local ErrState e; return e; }
// The message survives cudaGetLastError()'s clear (Task 1.4): the engine's ck() idiom is
// "cudaGetLastError(); throw ...(cudaGetErrorString(e))" - it CLEARS the error before formatting it,
// and without this the arena's named reason ("copy span straddles a piece", "out of VRAM growing
// the arena") vanished into CUDA's generic "unknown error". The clear stays CUDA-correct for the
// CODE; the reason stays readable.
inline ErrState& last_reason() { static thread_local ErrState r; return r; }
inline void set_error(cudaError_t code, const char* what) {
    if (code != cudaSuccess) { last_error() = ErrState{code, what ? what : "unknown error"};
                               last_reason() = last_error(); }
}
// Every arena-touching runtime call can throw (out of VRAM, straddling piece).  Catch once, per call.
#define STRATA_VK_TRY try {
#define STRATA_VK_CATCH(rc)                                                            \
    } catch (const std::exception& ex_) {                                              \
        strata_vk_compat::set_error((rc), ex_.what());                                 \
        return (rc);                                                                   \
    }

// ---- device pointers: offset+bias in the arena ----------------------------------------------------------
inline constexpr std::uintptr_t kPtrBias = 64;   // keeps a device pointer at offset 0 distinguishable from null
inline strata::vk::DevicePtr off(void* p) { return (strata::vk::DevicePtr)(std::uintptr_t)p - kPtrBias; }
inline strata::vk::DevicePtr off(const void* p) { return off(const_cast<void*>(p)); }
inline void* wrap(strata::vk::DevicePtr o) { return (void*)(std::uintptr_t)(o + kPtrBias); }

// Host pointers that came from cudaHostAlloc/cudaHostRegister/cudaMallocHost (for GetDevicePointer checks).
inline std::unordered_set<void*>& host_map() {
    static thread_local std::unordered_set<void*> m;
    return m;
}
inline std::mutex& host_map_m() { static std::mutex m; return m; }

}  // namespace strata_vk_compat

// A device pointer as the arena sees it (for launchers and tests that hold a host-side void* device pointer).
inline uint64_t strata_vk_offset(void* p) { return strata_vk_compat::off(p); }

// ---- kinds, flags, enums --------------------------------------------------------------------------------
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};
enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
};
enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
};
enum cudaDeviceScheduleFlags {
    cudaDeviceScheduleAuto = 0,
    cudaDeviceScheduleSpin = 1,
    cudaDeviceScheduleYield = 2,
    cudaDeviceScheduleBlockingSync = 4,
    cudaDeviceMapHost = 8,
};
enum cudaDeviceWarpLaunch { cudaWarpSizeLaunchAutomatic = 0, cudaWarpSizeLaunchLegacy = 1 };
constexpr unsigned int cudaHostAllocDefault = 0x00, cudaHostAllocPortable = 0x01,
                       cudaHostAllocMapped = 0x02, cudaHostAllocWriteCombined = 0x04;
// (declared, never used by host code): cudaHostRegister flags below.
constexpr unsigned int cudaHostRegisterDefault = 0x00, cudaHostRegisterPortable = 0x01,
                       cudaHostRegisterMapped = 0x02, cudaHostRegisterReadOnly = 0x08;
constexpr unsigned int cudaEventDefault = 0x00, cudaEventBlockingSync = 0x01,
                       cudaEventDisableTiming = 0x02, cudaEventInterprocess = 0x04;
constexpr unsigned int cudaStreamNonBlocking = 0x01;
// The legacy default stream handle (generate.cpp #550 names it; on this single queue it IS stream 0).
inline void* const cudaStreamLegacy = nullptr;

// cudaDeviceGetAttribute ids: the ones the tree's host files pass (values internal to this layer).
enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxSharedMemoryPerBlock = 3,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrClockRate = 13,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrIntegrated = 83,
};

// ---- opaque handles -------------------------------------------------------------------------------------
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;
typedef void* cudaGraphNode_t;
enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7,
};
struct cudaKernelNodeParams {
    void* func; void** kernelParams; void** extra;
    unsigned gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;
    unsigned sharedMemBytes;
};
enum cudaGraphDependencyType { cudaGraphDependencyTypeDefault = 0, cudaGraphDependencyTypeProgrammatic = 1 };
struct cudaGraphEdgeData { cudaGraphDependencyType type; };

// ---- cudaDeviceProp: the fields the host files READ (device.cu itself is not compiled; its Vulkan twin
// arrives in Task 1.3).  Names exactly as CUDA spells them so `prop.<field>` sites compile.
struct cudaDeviceProp {
    char name[256] = {0};
    char gcnArchName[256] = {0};        // HIP spelling; device.cu/verify.cpp read it even in non-HIP builds
    int major = 0, minor = 0;           // NOT a CUDA cc: the RDNA1 family code (10,x) per the plan
    size_t totalGlobalMem = 0;
    size_t sharedMemPerBlock = 0;
    int regsPerBlock = 0;
    int warpSize = 32;                  // logical warp: the kernels' 32-lane contract (subgroup forced to 32)
    size_t memPitch = 0;
    int maxThreadsPerBlock = 1024;
    int maxThreadsPerMultiProcessor = 2048;
    int multiProcessorCount = 0;        // DeviceInfo does not surface a CU count yet (Task 1.3's twin fills it)
    int l2CacheSize = 0;                // not surfaced by DeviceInfo yet; readers only tune prefetch with it
    int maxSharedMemoryPerMultiprocessor = 65536;   // measured on gfx1013 (probe: 64 KiB LDS)
    int integrated = 0;                 // BC-250 is a discrete GPU on a unified heap (uma=1, not an APU)
    int pciBusID = 0, pciDeviceID = 0, pciDomainID = 0;
    char luid[8] = {0};                 // Windows shared-memory budget query; stays zero here
    int concurrentKernels = 1;
    int clockRate = 0, memoryClockRate = 0, memoryBusWidth = 0;
    int maxPersistingL2CacheSize = 0;
};

// ---- the runtime calls ----------------------------------------------------------------------------------
inline const char* cudaGetErrorString(cudaError_t e) {
    using namespace strata_vk_compat;
    if (e == cudaSuccess) return "no error";
    if (last_error().code == e && !last_error().msg.empty()) return last_error().msg.c_str();
    // ck()-clears-first (see last_reason): the named reason outlives the clear.
    if (last_reason().code == e && !last_reason().msg.empty()) return last_reason().msg.c_str();
    switch ((int) e) {
        case cudaErrorMemoryAllocation: return "out of memory (arena)";
        case cudaErrorNotReady:         return "not ready";
        case cudaErrorNotSupported:     return "not supported";
        default:                        return "error";
    }
}
inline const char* cudaGetErrorName(cudaError_t e) {
    return e == cudaSuccess ? "cudaSuccess" : "cudaError";
}
inline cudaError_t cudaGetLastError() {
    auto e = strata_vk_compat::last_error();
    strata_vk_compat::last_error() = {};      // CUDA's contract: this call CLEARS the error
    return e.code;
}
inline cudaError_t cudaPeekAtLastError() { return strata_vk_compat::last_error().code; }

inline cudaError_t cudaGetDeviceCount(int* count) {
    *count = 0;
    STRATA_VK_TRY
    *count = strata::vk::device_count();
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorInitializationError)
}
inline int& strata_vk_cur_device() { static thread_local int d = 0; return d; }
inline cudaError_t cudaGetDevice(int* dev) { *dev = strata_vk_cur_device(); return cudaSuccess; }
inline cudaError_t cudaSetDevice(int dev) {
    if (dev < 0 || dev >= strata::vk::device_count())
        return cudaErrorInvalidValue;
    strata_vk_cur_device() = dev;   // single-device backend: the ordinal is remembered, nothing switches
    return cudaSuccess;
}
inline cudaError_t cudaSetDeviceFlags(unsigned) { return cudaSuccess; }
inline cudaError_t cudaInitDevice(int, unsigned, unsigned) { return cudaSuccess; }

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int dev) {
    if (!p) return cudaErrorInvalidValue;
    STRATA_VK_TRY
    const strata::vk::DeviceInfo& d = strata::vk::device_info(dev);
    std::memset(p, 0, sizeof *p);
    std::snprintf(p->name, sizeof p->name, "%s", d.name.c_str());
    // gfx1013: this backend only ever runs on the BC-250 family (vk_device.cpp refuses devices without
    // shader-integer-dot-product + subgroup-size-control); the engine's gfx-arch probes read this field.
    std::snprintf(p->gcnArchName, sizeof p->gcnArchName, "gfx1013");
    p->major = 10; p->minor = 1;              // RDNA1 family code, not a CUDA compute capability
    p->totalGlobalMem = (size_t) d.total_bytes;
    p->sharedMemPerBlock = 65536;             // measured maxComputeSharedMemorySize on this part
    p->maxSharedMemoryPerMultiprocessor = 65536;
    p->multiProcessorCount = d.multi_processor_count;   // VK_AMD_shader_core_properties (this box: 24)
    p->l2CacheSize = d.l2_bytes;                        // 0: radv exposes no L2 query (documented gap)
    p->maxThreadsPerBlock = (int) d.max_workgroup_invocations;
    p->integrated = 0;                        // discrete GPU (its heap is unified, but it is not an APU)
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorInitializationError)
}

inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int dev) {
    if (!value) return cudaErrorInvalidValue;
    STRATA_VK_TRY
    const strata::vk::DeviceInfo& d = strata::vk::device_info(dev);
    switch (attr) {
        case cudaDevAttrIntegrated:                    *value = 0; break;
        case cudaDevAttrWarpSize:                      *value = 32; break;
        case cudaDevAttrMaxSharedMemoryPerBlock:
        case cudaDevAttrMaxSharedMemoryPerBlockOptin:  *value = 65536; break;
        case cudaDevAttrMultiProcessorCount:           *value = 0; break;   // surfaced by Task 1.3's twin
        case cudaDevAttrComputeCapabilityMajor:        *value = 10; break;
        case cudaDevAttrComputeCapabilityMinor:        *value = 1; break;
        case cudaDevAttrMaxThreadsPerBlock:            *value = (int) d.max_workgroup_invocations; break;
        case cudaDevAttrClockRate:
        case cudaDevAttrMemoryClockRate:
        case cudaDevAttrGlobalMemoryBusWidth:          *value = 0; break;
        default:                                       *value = 0; break;
    }
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorInitializationError)
}
inline cudaError_t cudaDeviceGetStreamPriorityRange(int* lo, int* hi) {
    if (lo) *lo = 0; if (hi) *hi = 0;   // one queue: every priority is the same queue
    return cudaSuccess;
}
inline cudaError_t cudaDriverGetVersion(int* v) { if (v) *v = 12040; return cudaSuccess; }
inline cudaError_t cudaRuntimeGetVersion(int* v) { if (v) *v = CUDART_VERSION; return cudaSuccess; }

inline cudaError_t cudaDeviceSynchronize() {
    STRATA_VK_TRY
    strata::vk::device_sync();
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorUnknown)
}
// Multi-GPU is out of scope for this port (the plan defers remote/peer experts); say so, do not fake access.
inline cudaError_t cudaDeviceCanAccessPeer(int* can, int, int) { if (can) *can = 0; return cudaSuccess; }
inline cudaError_t cudaDeviceEnablePeerAccess(int, unsigned) {
    strata_vk_compat::set_error(cudaErrorNotSupported, "strata-vk: peer access is out of scope for this backend");
    return cudaErrorNotSupported;
}
inline cudaError_t cudaDeviceDisablePeerAccess(int) { return cudaSuccess; }

// ---- memory ---------------------------------------------------------------------------------------------
inline cudaError_t cudaMalloc(void** p, size_t bytes) {
    if (!p) return cudaErrorInvalidValue;
    STRATA_VK_TRY
    *p = strata_vk_compat::wrap(strata::vk::Arena::instance().alloc(bytes));
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorMemoryAllocation)
}
// CUDA's C++ cuda_runtime.h adds typed overloads; the tree calls cudaMalloc(&float_ptr, n) directly.
template <typename T> inline cudaError_t cudaMalloc(T** p, size_t bytes) {
    return cudaMalloc(reinterpret_cast<void**>(p), bytes);
}
// Bump arena, same rule as CUDA's DeviceArena / the plan: cudaFree frees nothing.
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMallocAsync(void** p, size_t bytes, cudaStream_t) { return cudaMalloc(p, bytes); }
inline cudaError_t cudaFreeAsync(void*, cudaStream_t) { return cudaSuccess; }
// The pool handle: nothing pools - the arena is the allocator.
typedef void* cudaMemPool_t;

inline cudaError_t cudaMemGetInfo(size_t* free_, size_t* total) {
    STRATA_VK_TRY
    const strata::vk::DeviceInfo& d = strata::vk::device_info();
    const uint64_t used = strata::vk::Arena::instance().used();
    if (total) *total = (size_t) d.total_bytes;
    if (free_) *free_ = (size_t) (d.total_bytes > used ? d.total_bytes - used : 0);
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorInitializationError)
}

// The one copy dispatcher: every kind lands on the staged chunked copies of the runtime.
inline cudaError_t strata_memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t s) {
    using namespace strata_vk_compat;
    if (bytes == 0) return cudaSuccess;
    STRATA_VK_TRY
    switch (kind) {
        case cudaMemcpyHostToDevice:
            strata::vk::copy_h2d((void*)(std::uintptr_t) off(dst), src, bytes, s);   // the runtime takes the raw arena offset
            break;
        case cudaMemcpyDeviceToHost:
            strata::vk::copy_d2h(dst, (void*)(std::uintptr_t) off(src), bytes, s);
            break;
        case cudaMemcpyDeviceToDevice: {
            if (strata::vk::cap_here()) {           // Phase 4.2: a real device copy node
                strata::vk::graph_record_d2d((std::uintptr_t) off(dst), (std::uintptr_t) off(src), bytes);
                break;
            }
            // No eager device-side copy primitive: walk it through a host scratch (correct, not
            // fast - speed is a PORT-STATUS todo; D2D moves are small state tables today).
            static thread_local std::vector<uint8_t> scratch;
            constexpr size_t chunk = 2ull << 20;
            scratch.resize(chunk);
            for (size_t at = 0; at < bytes; at += chunk) {
                size_t n = bytes - at < chunk ? bytes - at : chunk;
                strata::vk::copy_d2h(scratch.data(), (void*)(std::uintptr_t)(off(src) + at), n, s);
                strata::vk::copy_h2d((void*)(std::uintptr_t)(off(dst) + at), scratch.data(), n, s);
            }
            break;
        }
        case cudaMemcpyDefault:
            // Inferring direction from pointer ranges is exactly the silent-fake this port refuses.
            set_error(cudaErrorNotSupported, "strata-vk: cudaMemcpyDefault cannot infer direction here; pass an explicit kind");
            return cudaErrorNotSupported;
        default:
            set_error(cudaErrorNotSupported, "strata-vk: cudaMemcpyHostToHost not wired");
            return cudaErrorNotSupported;
    }
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorUnknown)
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    return strata_memcpy(dst, src, bytes, kind, nullptr);
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind,
                                   cudaStream_t s = nullptr) {
    return strata_memcpy(dst, src, bytes, kind, s);   // the runtime submits-and-waits; async IS sync here
}
// Strided copies: one row per staged copy call.
inline cudaError_t strata_memcpy2d(void* dst, size_t dpitch, const void* src, size_t spitch,
                                   size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t s) {
    for (size_t r = 0; r < height; ++r) {
        cudaError_t e = strata_memcpy((char*) dst + r * dpitch, (const char*) src + r * spitch, width, kind, s);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                                size_t width, size_t height, cudaMemcpyKind kind) {
    return strata_memcpy2d(dst, dpitch, src, spitch, width, height, kind, nullptr);
}
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                                     size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t s) {
    return strata_memcpy2d(dst, dpitch, src, spitch, width, height, kind, s);
}
inline cudaError_t cudaMemcpyPeer(void* d, int, const void* s2, int, size_t) {
    strata_vk_compat::set_error(cudaErrorNotSupported, "strata-vk: cudaMemcpyPeer (multi-GPU) not ported");
    return cudaErrorNotSupported;
}
inline cudaError_t cudaMemcpyPeerAsync(void* d, int, const void* s2, int, size_t, cudaStream_t) {
    return cudaMemcpyPeer(d, 0, s2, 0, 0);
}
// Device-symbol copies: there is no device-symbol image here (the CUDA build's __constant__ tables live in
// .cu files this backend does not compile).  Name the gap rather than write to a random offset.
inline cudaError_t cudaMemcpyToSymbol(const void*, const void*, size_t, size_t = 0, cudaMemcpyKind = cudaMemcpyHostToDevice) {
    strata_vk_compat::set_error(cudaErrorNotSupported, "strata-vk: cudaMemcpyToSymbol - no device-symbol image; port the table as an arena allocation");
    return cudaErrorNotSupported;
}
inline cudaError_t cudaMemcpyFromSymbol(void*, const void*, size_t, size_t = 0, cudaMemcpyKind = cudaMemcpyDeviceToHost) {
    strata_vk_compat::set_error(cudaErrorNotSupported, "strata-vk: cudaMemcpyFromSymbol - no device-symbol image; port the table as an arena allocation");
    return cudaErrorNotSupported;
}

inline cudaError_t strata_memset(void* dev, int value, size_t bytes, cudaStream_t s) {
    using namespace strata_vk_compat;
    STRATA_VK_TRY
    strata::vk::fill_d((void*)(std::uintptr_t) off(dev), (uint8_t) value, bytes, s);
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorUnknown)
}
inline cudaError_t cudaMemset(void* dev, int value, size_t bytes) { return strata_memset(dev, value, bytes, nullptr); }
inline cudaError_t cudaMemsetAsync(void* dev, int value, size_t bytes, cudaStream_t s = nullptr) { return strata_memset(dev, value, bytes, s); }

// Host memory.  UMA box: staging and doorbell hosts are ordinary page-aligned RAM the staged copies read
// directly (there is no driver pin to register against, and none is needed for the copies).  The device
// ALIAS for host memory is what the pinned twins (Task 1.3) will wire; until then GetDevicePointer names the
// gap and returns an error - every engine call site around it already has a fallback branch.
inline cudaError_t cudaHostAlloc(void** p, size_t bytes, unsigned) {
    if (!p) return cudaErrorInvalidValue;
    void* m = nullptr;
    if (bytes) {
        size_t r = (bytes + 4095) & ~(size_t) 4095;      // aligned_alloc wants a multiple of the alignment
        m = ::aligned_alloc(4096, r);
    }
    if (!m) { strata_vk_compat::set_error(cudaErrorMemoryAllocation, "strata-vk: cudaHostAlloc (aligned_alloc) failed"); return cudaErrorMemoryAllocation; }
    std::lock_guard<std::mutex> lk(strata_vk_compat::host_map_m());
    strata_vk_compat::host_map().insert(m);
    *p = m;
    return cudaSuccess;
}
template <typename T> inline cudaError_t cudaHostAlloc(T** p, size_t bytes, unsigned f) {
    return cudaHostAlloc(reinterpret_cast<void**>(p), bytes, f);
}
inline cudaError_t cudaMallocHost(void** p, size_t bytes) { return cudaHostAlloc(p, bytes, cudaHostAllocDefault); }
template <typename T> inline cudaError_t cudaMallocHost(T** p, size_t b) { return cudaHostAlloc(reinterpret_cast<void**>(p), b, cudaHostAllocDefault); }
inline cudaError_t cudaFreeHost(void* p) {
    if (!p) return cudaSuccess;
    {
        std::lock_guard<std::mutex> lk(strata_vk_compat::host_map_m());
        strata_vk_compat::host_map().erase(p);
    }
    ::free(p);
    return cudaSuccess;
}
inline cudaError_t cudaHostRegister(void* p, size_t, unsigned) {
    std::lock_guard<std::mutex> lk(strata_vk_compat::host_map_m());
    strata_vk_compat::host_map().insert(p);
    return cudaSuccess;      // already "pinned" in the sense this backend needs: the copies read host RAM
}
inline cudaError_t cudaHostUnregister(void* p) {
    std::lock_guard<std::mutex> lk(strata_vk_compat::host_map_m());
    strata_vk_compat::host_map().erase(p);
    return cudaSuccess;
}
inline cudaError_t cudaHostGetDevicePointer(void** p, void* h, unsigned) {
    if (!h || !p) return cudaErrorInvalidValue;
    // Task 1.3 measured why there is no cheap honest alias (vk_doorbell_probe): RADV never delivers a
    // live host store to a RUNNING kernel, and the arena's offset-as-pointer space does not contain
    // host RAM — handing back the host address would have the doorbell/publish kernels' launchers read
    // it as an arena offset and bind the wrong bytes. Call sites all have fallback branches; the ones
    // that need a real device view of pinned RAM take the staged route (ring -> submit boundary ->
    // launch, proven by the probe's arm 4) when their kernels port.
    strata_vk_compat::set_error(cudaErrorNotSupported,
        "strata-vk: no device alias for host memory (doorbell measured: host stores never reach a running kernel on RADV; use ring-then-launch)");
    return cudaErrorNotSupported;
}

// ---- streams --------------------------------------------------------------------------------------------
// The runtime's streams are opaque queue wrappers (today: one queue; null = the default queue).
inline cudaError_t cudaStreamCreate(cudaStream_t* s) {
    if (!s) return cudaErrorInvalidValue;
    *s = strata::vk::stream_create();
    return cudaSuccess;
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { return cudaStreamCreate(s); }
inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned, int) { return cudaStreamCreate(s); }
inline cudaError_t cudaStreamDestroy(cudaStream_t s) { strata::vk::stream_destroy(s); return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    STRATA_VK_TRY
    strata::vk::stream_sync(s);
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorUnknown)
}
// Every submission waits (the runtime's end_submit does vkQueueWaitIdle), so a stream is always drained.
inline cudaError_t cudaStreamQuery(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* st) {
    if (st) *st = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) {
    return cudaSuccess;      // events are recorded after a queue drain (below): a wait is already satisfied
}
// A host function enqueued on a stream runs AT that point of the stream: submissions are synchronous, so
// calling it now is exactly CUDA's ordering.
typedef void (*cudaHostFn_t)(void*);
inline cudaError_t cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t fn, void* ud) {
    if (fn) fn(ud);
    return cudaSuccess;
}

// ---- events (Task 4.2: real GPU timestamps, with the host-clock arm as fallback) ----------------------
// The runtime's gpu arm records a vkCmdWriteTimestamp node INTO the stream (no drain per record)
// and elapsed() reads the driver's 64-bit ticks scaled by timestampPeriod: those are DEVICE
// timestamps, what --stage-timing's per-stage tables and STRATA_COMMIT_TIMING ask for. When the
// card (or STRATA_VK_NO_GPU_EVENTS=1) selects the host arm, record = a queue drain plus a
// steady_clock stamp — an honest end-to-end interval, never a fake driver timestamp. The compat
// surface below is identical either way; strata::vk::event_* decides per call.
struct StrataVkEvent { };
inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
    if (!e) return cudaErrorInvalidValue;
    *e = strata::vk::event_create();
    return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned f) {
    if (!e) return cudaErrorInvalidValue;
    *e = strata::vk::event_create((f & cudaEventDisableTiming) == 0);
    return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { strata::vk::event_destroy(e); return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = nullptr) {
    if (!e) return cudaErrorInvalidValue;
    STRATA_VK_TRY
    strata::vk::event_record(e, s);
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorUnknown)
}
inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }   // the fetch below waits, so: done
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    if (!e) return cudaErrorInvalidValue;
    STRATA_VK_TRY
    strata::vk::event_synchronize(e);   // gpu arm: submit/drain the CB carrying the stamp
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorUnknown)
}
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t from, cudaEvent_t to) {
    if (!ms || !from || !to) return cudaErrorInvalidValue;
    STRATA_VK_TRY
    *ms = strata::vk::event_elapsed(from, to);
    return cudaSuccess;
    STRATA_VK_CATCH(cudaErrorUnknown)
}

// ---- the graph family: Phase 4.2, recorded command buffers ---------------------------------------------
// vk_runtime's capture layer (graph_begin/end/launch/destroy) drives these. A capture records the
// body's launches into ONE multi-submit command buffer with the eager per-dispatch barrier law;
// replay is one vkQueueSubmit. A sync/copy site inside a capture body aborts LOUDLY (the runtime
// checks `capturing()` at every drain point) - the doorbell law says nothing may be skipped in
// silence. graph.cpp (core) is still not compiled here; engine.cpp's record()/replayed() drive
// this family directly under --graphs.
[[noreturn]] inline void strata_vk_graph_trap(const char* what) {
    std::fprintf(stderr, "strata-vk: %s not ported; run with --no-graphs\n", what);
    std::abort();
}
// handles: (index+1) as a pointer; 0/null is the engine's own "none" value.
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
    strata::vk::graph_begin(); return cudaSuccess; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* g) {
    if (!g) return cudaErrorInvalidValue;
    strata::vk::graph_end();
    // the graph handle is the LAST registered index: engine.cpp ends capture right after each body
    *g = (cudaGraph_t)(uintptr_t)(strata::vk::capture_graph_count());   // 1-based via the count
    return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }   // the exec owns the CB now
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t e) {
    if (!e) return cudaSuccess;
    strata::vk::graph_destroy((uint32_t)((uintptr_t)e - 1)); return cudaSuccess; }
inline cudaError_t cudaGraphGetNodes(cudaGraph_t g, cudaGraphNode_t*, size_t* n) {
    if (n) *n = g ? strata::vk::graph_dispatches((uint32_t)((uintptr_t)g - 1)) : 0; return cudaSuccess; }
inline cudaError_t cudaGraphGetEdges(cudaGraph_t, cudaGraphNode_t*, cudaGraphNode_t*, cudaGraphEdgeData*, size_t*) { strata_vk_graph_trap("cudaGraphGetEdges"); }
inline cudaError_t cudaGraphGetEdges_v2(cudaGraph_t, cudaGraphNode_t*, cudaGraphNode_t*, cudaGraphEdgeData*, size_t*) { strata_vk_graph_trap("cudaGraphGetEdges_v2"); }
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t g, cudaGraphNode_t*, char*, size_t) {
    if (!e || !g) return cudaErrorInvalidValue;
    *e = g;                       // same index: the captured CB becomes the exec directly
    return cudaSuccess; }
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t g, unsigned long long) {
    if (!e || !g) return cudaErrorInvalidValue;
    *e = g; return cudaSuccess; }
inline cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* e, cudaGraph_t g, unsigned long long) {
    if (!e || !g) return cudaErrorInvalidValue;
    *e = g; return cudaSuccess; }
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t e, cudaStream_t) {
    if (!e) return cudaErrorInvalidValue;
    strata::vk::graph_launch((uint32_t)((uintptr_t)e - 1)); return cudaSuccess; }
inline cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t) { strata_vk_graph_trap("cudaGraphUpload"); }
inline cudaError_t cudaGraphNodeGetType(cudaGraphNode_t, cudaGraphNodeType*) { strata_vk_graph_trap("cudaGraphNodeGetType"); }
inline cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t, cudaKernelNodeParams*) { strata_vk_graph_trap("cudaGraphKernelNodeGetParams"); }

// ---- driver-entry-point queries (vmm.cpp / expert_cache.cpp resolve the CUDA VMM driver API this way) ----
// Vulkan has no cuMem* driver functions: answer "symbol not found", which drives the engine's own
// documented fallback (vmm.cpp `api().ok == false`, the plain-allocation tier).  Not a trap: these paths are
// feature-detects, not requirements.
enum cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess = 0,
    cudaDriverEntryPointSymbolNotFound = 1,
    cudaDriverEntryPointVersionNotSufficent = 2,
};
constexpr unsigned cudaEnableDefault = 0x0, cudaEnableLegacy = 0x1, cudaEnablePerThreadDefaultStream = 0x2;
inline cudaError_t cudaGetDriverEntryPoint(const char*, void** fn, unsigned, cudaDriverEntryPointQueryResult* q) {
    if (fn) *fn = nullptr;
    if (q) *q = cudaDriverEntryPointSymbolNotFound;
    return cudaSuccess;
}
inline cudaError_t cudaGetDriverEntryPointByVersion(const char* n, void** fn, unsigned, unsigned f, cudaDriverEntryPointQueryResult* q) {
    return cudaGetDriverEntryPoint(n, fn, f, q);
}
inline cudaError_t cudaFuncGetName(const char** name, const void*) { if (name) *name = nullptr; return cudaErrorNotSupported; }
// Shared-memory opt-in etc.: no device-side function objects exist here; the launches are sized by the
// Vulkan launchers.  Accept and do nothing (the CUDA calls only tune the CUDA launch).
template <typename Kernel>
inline cudaError_t cudaFuncSetAttribute(Kernel, int, int) { return cudaSuccess; }
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8, cudaFuncAttributePreferredSharedMemoryCarveout = 9 };
constexpr int cudaSharedmemCarveoutMaxShared = 100;

// ---- vector types (vector_types.h / dim3: engine host files build launch shapes with them) ----------------
struct dim3 { unsigned x, y, z; dim3(unsigned X = 1, unsigned Y = 1, unsigned Z = 1) : x(X), y(Y), z(Z) {} };
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct int2 { int x, y; };
struct int4 { int x, y, z, w; };
struct uint2 { unsigned x, y; };
struct uint3 { unsigned x, y, z; };
struct uint4 { unsigned x, y, z, w; };
struct char4 { signed char x, y, z, w; };
struct uchar4 { unsigned char x, y, z, w; };
struct short2 { short x, y; };
struct short4 { short x, y, z, w; };
struct uchar2 { unsigned char x, y; };
struct ushort2 { unsigned short x, y; };
struct longlong2 { long long x, y; };
struct ulonglong2 { unsigned long long x, y; };
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline int2 make_int2(int x, int y) { return {x, y}; }
inline int4 make_int4(int x, int y, int z, int w) { return {x, y, z, w}; }
inline uint2 make_uint2(unsigned x, unsigned y) { return {x, y}; }
inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) { return {x, y, z}; }
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return {x, y, z, w}; }
inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) { return {x, y, z, w}; }
inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { return {x, y, z, w}; }
inline uchar2 make_uchar2(unsigned char x, unsigned char y) { return {x, y}; }
inline ushort2 make_ushort2(unsigned short x, unsigned short y) { return {x, y}; }
inline short2 make_short2(short x, short y) { return {x, y}; }
inline short4 make_short4(short x, short y, short z, short w) { return {x, y, z, w}; }

// ---- launch shim (no <<<...>>> on a C++ compiler: the engine's host files never use it; device files are
// replaced by Vulkan twins.  cudaLaunchKernel sites: none in ../src's .cpp files as of the Task-1.1 grep.)
