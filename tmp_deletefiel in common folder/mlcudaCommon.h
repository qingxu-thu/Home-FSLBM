#pragma once
#ifndef _MLCUDACOMMON_
#define _MLCUDACOMMON_


#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
////#define __CUDA_INTERNAL_COMPILATION__
//#include "math_functions.h"
////#undef __CUDA_INTERNAL_COMPILATION__
//#include "helper_math.h"
#include "../3rdParty/helper_cuda.h"
#include "builtin_types.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#define ML_CUDA_BLOCK_SIZE 8

#define MLCUDA_DEVICE
inline int GVLCuDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

#endif // !_MLCUDACOMMON_