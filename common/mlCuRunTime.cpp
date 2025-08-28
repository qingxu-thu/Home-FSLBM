#include "mlCoreWin.h"
#include "mlCuRunTime.h"

cudaError_t _MLCuMalloc(void ** devPtr, size_t size)
{
	return cudaMalloc(devPtr, size);
}

cudaError_t _MLCuMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
	return cudaMemcpy(dst, src, count, kind);
}

cudaError_t _MLCuFree(void* devPtr)
{
	return cudaFree(devPtr);
}

 