#pragma once
#ifndef _MRCONSTANTPARAMSGPU2DH_
#define _MRCONSTANTPARAMSGPU2DH_
#include "../../../common/mlcudaCommon.h"
#include "mrLbmSolverGpu2D.h"

#define def_c  0.57735027f
#define def_w0 (1.0f/2.25f) // center (0)
#define def_ws (1.0f/9.0f) // straight (1-4)
#define def_we (1.0f/36.0f) // edge (5-8)
#define def_6_sigma 3e-2

#define K_h 1e-3
#define MAX_BUBBLE_SIZE 1024

//                                 0  1  2  3  4  5  6  7  8  
__constant__ float ex2d_gpu[9] = { 0, 1,-1, 0, 0, 1,-1, 1,-1 };
__constant__ float ey2d_gpu[9] = { 0, 0, 0, 1,-1, 1,-1,-1, 1 };
__constant__ float ez2d_gpu[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
__constant__ int index2dInv_gpu[9] = { 0,2,1,4,3,6,5,8,7};
__constant__ float w2d_gpu[9] = { def_w0, def_ws, def_ws, def_ws, def_ws, def_we, def_we, def_we, def_we };
__constant__ float as2 = 3.0f;
__constant__ float cs2 = 1.0f / 3.0f;

__constant__ float d2q5_w[5] = { 1.0 / 3.0, 1.0 / 6.0,1.0 / 6.0,1.0 / 6.0 ,1.0 / 6.0 };


#endif // !_MRCONSTANTPARAMSGPU2DH_
