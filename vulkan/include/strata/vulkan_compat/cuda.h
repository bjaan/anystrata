// cuda.h for a Vulkan build: the driver-API types src/core/vmm.cpp spells in its (dlopen-resolved) VMM tier.
// None of these FUNCTIONS exists here - vmm.cpp only names them in `decltype(&cuMemAddressReserve)`-style
// member types and reaches them through cudaGetDriverEntryPoint, which reports "symbol not found" (see
// strata_vk.h), so the VMM tier switches itself off and the arena fallback runs.  The declarations below are
// never linked against: they exist so the TU parses.
#pragma once

#include <cstddef>
#include <cstdint>

typedef int CUresult;
#define CUDA_SUCCESS 0
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

extern "C" {
#ifndef CUDAAPI
#define CUDAAPI
#endif
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuGetErrorString(CUresult res, const char** str);
CUresult cuGetErrorName(CUresult res, const char** str);
CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags);
CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);
CUresult cuDeviceGetAttribute(int* value, CUdevice_attribute attr, CUdevice dev);
CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option);
}
