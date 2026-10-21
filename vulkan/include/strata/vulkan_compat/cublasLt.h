// cublasLt.h for a Vulkan build: the tree's greps name no cublasLt call site (rocBLAS/hipBLASLt tuning is
// behind __HIPCC__ guards in src/prefill).  This header exists so an accidental `#include <cublasLt.h>`
// resolves to an empty surface with named-unsupported entry points instead of the absent NVIDIA header.
#pragma once
#include "cublas_v2.h"
