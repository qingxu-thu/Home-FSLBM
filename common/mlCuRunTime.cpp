#include "mlCoreWin.h"
#include "mlCuRunTime.h"

cudaError_t _MLCuMalloc(void ** devPtr, size_t size)
{
	return cudaMalloc(devPtr, size);
}

cudaError_t _MLCuMallocHost(void ** ptr, size_t size)
{
	return cudaMallocHost(ptr, size);
}

cudaError_t _MLCuMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height)
{
	return cudaMallocPitch(devPtr, pitch, width, height);
}

cudaError_t _MLCuMallocArray(cudaArray ** array, const cudaChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags)
{
	return cudaMallocArray(array, desc, width, height, flags);
}

cudaError_t _MLCuMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned int flags)
{
	return cudaMalloc3DArray(array, desc, extent, flags);
}

cudaError_t _MLCuMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t depth, unsigned int flags)
{
	cudaExtent extent = make_cudaExtent(width, height, depth);
	return cudaMalloc3DArray(array, desc, extent, flags);
}

cudaError_t _MLCuMemset(void * devPtr, int value, size_t count)
{
	return cudaMemset(devPtr, value, count);
}

cudaError_t _MLCuFree(void * devPtr)
{
	return cudaFree(devPtr);
}

cudaError_t _MLCuFreeHost(void * ptr)
{
	return cudaFreeHost(ptr);
}

cudaError_t _MLCuFreeArray(cudaArray * array)
{
	return cudaFreeArray(array);
}

cudaError_t _MLCuMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind)
{
	return cudaMemcpy(dst, src, count, kind);
}

cudaError_t _MLCuMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count)
{
	return cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);

}

 

 
 

cudaError_t _MLCuMemcpy3D(const cudaMemcpy3DParms * p)
{
	return cudaMemcpy3D(p);
}

cudaError_t _MLCuMemcpyToArray3D(cudaArray * dst, const void * src, size_t size, size_t width, size_t height, size_t depth, cudaMemcpyKind kind)
{
	cudaExtent extent = make_cudaExtent(width, height, depth);

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void *)src, width*size, width, height);
	copyParams.dstArray = dst;
	copyParams.extent = extent;
	copyParams.kind = kind;

	return cudaMemcpy3D(&copyParams);
}

cudaChannelFormatDesc _MLCuCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f)
{
	return cudaCreateChannelDesc(x, y, z, w, f);
}

cudaChannelFormatDesc _MLCuCreateChannelDescFloat1()
{
	return cudaCreateChannelDesc<float>();
}

cudaChannelFormatDesc _MLCuCreateChannelDescFloat2()
{
	return cudaCreateChannelDesc<float2>();
}

cudaChannelFormatDesc _MLCuCreateChannelDescFloat3()
{
	return cudaCreateChannelDesc<float3>();
}

cudaChannelFormatDesc _MLCuCreateChannelDescFloat4()
{
	return cudaCreateChannelDesc<float4>();
}
