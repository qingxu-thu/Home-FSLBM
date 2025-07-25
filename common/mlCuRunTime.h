#pragma  once
#ifndef _MLCURUNTIME_
#define _MLCURUNTIME_
#include "mlCoreWin.h"
//#include "builtin_types.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"


cudaError_t _MLCuMalloc(void**devPtr, size_t size);
cudaError_t _MLCuMallocHost(void**ptr, size_t size);
cudaError_t _MLCuMallocPitch(void**devPtr, size_t *pitch, size_t width, size_t height);
cudaError_t _MLCuMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height = 0, unsigned int flags = cudaArrayDefault);

cudaError_t _MLCuMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags = cudaArrayDefault);
cudaError_t _MLCuMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, size_t depth, unsigned int flags = cudaArrayDefault);

cudaError_t _MLCuMemset(void * devPtr, int value, size_t count);

cudaError_t _MLCuFree(void *devPtr);
cudaError_t _MLCuFreeHost(void *ptr);
cudaError_t _MLCuFreeArray(struct cudaArray *array);

cudaError_t _MLCuMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

cudaError_t _MLCuMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
cudaError_t _MLCuMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t _MLCuMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
cudaError_t _MLCuMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);
cudaError_t _MLCuMemcpy3D(const struct cudaMemcpy3DParms *p);
cudaError_t _MLCuMemcpyToArray3D(struct cudaArray *dst, const void *src, size_t size, size_t width, size_t height, size_t depth, enum cudaMemcpyKind kind);

cudaChannelFormatDesc _MLCuCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
cudaChannelFormatDesc _MLCuCreateChannelDescFloat1();
cudaChannelFormatDesc _MLCuCreateChannelDescFloat2();
cudaChannelFormatDesc _MLCuCreateChannelDescFloat3();
cudaChannelFormatDesc _MLCuCreateChannelDescFloat4();


#ifdef MLCUDA_DEVICE

inline cudaError_t __MLCuMemcpyToArray3D(struct cudaArray *dst, const void *src, size_t size, size_t width, size_t height, size_t depth, enum cudaMemcpyKind kind)
{
	cudaExtent extent = make_cudaExtent(width, height, depth);

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void *)src, width*size, width, height);
	copyParams.dstArray = dst;
	copyParams.extent = extent;
	copyParams.kind = kind;

	return cudaMemcpy3D(&copyParams);
}

#define MLCuMalloc cudaMalloc
#define MLCuMallocHost cudaMallocHost
#define MLCuMallocPitch cudaMallocPitch
#define MLCuMallocArray cudaMallocArray

#define MLCuMalloc3DArray cudaMalloc3DArray

#define MLCuMemset cudaMemset

#define MLCuFree cudaFree
#define MLCuFreeHost cudaFreeHost
#define MLCuFreeArray cudaFreeArray

#define MLCuMemcpy cudaMemcpy

#define MLCuMemcpyPeer cudaMemcpyPeer
#define MLCuMemcpyToArray cudaMemcpyToArray
#define MLCuMemcpyFromArray cudaMemcpyFromArray
#define MLCuMemcpyArrayToArray cudaMemcpyArrayToArray
#define MLCuMemcpy3D cudaMemcpy3D
#define MLCuMemcpyToArray3D __MLCuMemcpyToArray3D

#define MLCuCreateChannelDesc cudaCreateChannelDesc
#define MLCuCreateChannelDescFloat1 cudaCreateChannelDesc<float>
#define MLCuCreateChannelDescFloat2 cudaCreateChannelDesc<float2>
#define MLCuCreateChannelDescFloat3 cudaCreateChannelDesc<float3>
#define MLCuCreateChannelDescFloat4 cudaCreateChannelDesc<float4>


#else /////////////////////////////////////

#define MLCuMalloc _MLCuMalloc
#define MLCuMallocHost _MLCuMallocHost
#define MLCuMallocPitch _MLCuMallocPitch
#define MLCuMallocArray _MLCuMallocArray

#define MLCuMalloc3DArray _MLCuMalloc3DArray

#define MLCuMemset _MLCuMemset

#define MLCuFree _MLCuFree
#define MLCuFreeHost _MLCuFreeHost
#define MLCuFreeArray _MLCuFreeArray

#define MLCuMemcpy _MLCuMemcpy

#define MLCuMemcpyPeer _MLCuMemcpyPeer
#define MLCuMemcpyToArray _MLCuMemcpyToArray
#define MLCuMemcpyFromArray _MLCuMemcpyFromArray
#define MLCuMemcpyArrayToArray _MLCuMemcpyArrayToArray
#define MLCuMemcpy3D _MLCuMemcpy3D
#define MLCuMemcpyToArray3D _MLCuMemcpyToArray3D

#define MLCuCreateChannelDesc _MLCuCreateChannelDesc
#define MLCuCreateChannelDescFloat1 _MLCuCreateChannelDescFloat1
#define MLCuCreateChannelDescFloat2 _MLCuCreateChannelDescFloat2
#define MLCuCreateChannelDescFloat3 _MLCuCreateChannelDescFloat3
#define MLCuCreateChannelDescFloat4 _MLCuCreateChannelDescFloat4

#endif /////////////////////////////////////
#endif // !_MLCURUNTIME_
