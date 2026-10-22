// cuda.h for a Vulkan build: the driver-API types src/core/vmm.cpp spells in its (dlopen-resolved) VMM tier.
// vmm.cpp only names these in `decltype(&cuMemAddressReserve)`-style member types and reaches them through
// cudaGetDriverEntryPoint, which reports "symbol not found" (see strata_vk.h), so the VMM tier switches
// itself off and the arena fallback runs.  Task 1.4 adds a DIRECT caller: src/poly/engine.cpp's load()/
// cover() call cuDeviceGet / cuDeviceGetAttribute / cuMem* without the entry-point dance (the CUDA build
// links CUDA::cuda_driver for it). So the declarations became inline definitions answering CUDA's own
// "not supported" convention: engine.cpp's probe (CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
// -> 0, error) leaves vmm_ false on its own terms, and the attention/expert caches take the plain
// cudaMalloc (arena) path - the grow-never fallback the twins already carry. Nothing here ever reports
// success, so a VMM path can never half-run.
#pragma once

#include <cstddef>
#include <cstdint>

typedef int CUresult;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_SUPPORTED 801
typedef int CUdevice;
typedef uint64_t CUdeviceptr;
typedef uint64_t CUmemGenericAllocationHandle;
enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
};
enum CUmemAllocationType { CU_MEM_ALLOCATION_TYPE_PINNED = 1 };
enum CUmemAllocationGranularity_flags {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1,
};
enum CUmemLocationType { CU_MEM_LOCATION_TYPE_DEVICE = 1 };
enum CUmemAccess_flags { CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3 };
typedef struct CUmemLocation { CUmemLocationType type; int id; } CUmemLocation;
typedef struct CUmemAllocationProp {
    CUmemAllocationType type;
    int requestedAllocationHandleTypes;         // vmm.cpp leaves it 0
    CUmemLocation location;
} CUmemAllocationProp;
typedef struct CUmemAccessDesc { CUmemLocation location; int flags; } CUmemAccessDesc;

#ifndef CUDAAPI
#define CUDAAPI
#endif
inline CUresult CUDAAPI cuDeviceGet(CUdevice*, int) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult CUDAAPI cuGetErrorString(CUresult, const char** str) {
    if (str) *str = "strata-vk: no CUDA driver: the VMM path is off (grow-never arena fallback)";
    return CUDA_SUCCESS; }
inline CUresult CUDAAPI cuGetErrorName(CUresult, const char** str) {
    if (str) *str = "CUDA_ERROR_NOT_SUPPORTED";
    return CUDA_SUCCESS; }
inline CUresult CUDAAPI cuDeviceGetAttribute(int* value, CUdevice_attribute attr, CUdevice) {
    if (value) *value = (attr == CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED) ? 0 : -1;
    return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* ptr, size_t, size_t, CUdeviceptr, unsigned long long) {
    if (ptr) *ptr = 0; return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult CUDAAPI cuMemAddressFree(CUdeviceptr, size_t) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle, size_t, const CUmemAllocationProp*, unsigned long long) {
    if (handle) *handle = 0; return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult CUDAAPI cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) {
    return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult CUDAAPI cuMemUnmap(CUdeviceptr, size_t) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult CUDAAPI cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) {
    return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult CUDAAPI cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) {
    if (granularity) *granularity = 0;
    return CUDA_ERROR_NOT_SUPPORTED; }
