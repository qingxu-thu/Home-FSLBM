#pragma  once
#ifndef _MLCURUNTIME_
#define _MLCURUNTIME_
#include "mlCoreWin.h"
//#include "builtin_types.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

cudaError_t _MLCuMalloc(void** devPtr, size_t size);
cudaError_t _MLCuMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t _MLCuFree(void* devPtr);


 

 

#define MLCuMalloc _MLCuMalloc
#define MLCuMemcpy _MLCuMemcpy
#define MLCuFree _MLCuFree
 

#endif /////////////////////////////////////
