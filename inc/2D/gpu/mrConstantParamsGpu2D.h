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

#define Search_Value 3
#define MAX_BUBBLE_SIZE 1024



//#define fma(a,b,c) a*b+c
//#define cb(x) x*x*x
//#define sq(x) x*x
//#define cbrt(x) powf(x, 1.0/3.0)
//#define clamp(x, a, b) fmin(fmax(x, a), b)

//                                 0  1  2  3  4  5  6  7  8  
__constant__ float ex2d_gpu[9] = { 0, 1,-1, 0, 0, 1,-1, 1,-1 };
__constant__ float ey2d_gpu[9] = { 0, 0, 0, 1,-1, 1,-1,-1, 1 };
__constant__ float ez2d_gpu[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
__constant__ int index2dInv_gpu[9] = { 0,2,1,4,3,6,5,8,7};
__constant__ float w2d_gpu[9] = { def_w0, def_ws, def_ws, def_ws, def_ws, def_we, def_we, def_we, def_we };
__constant__ float as2 = 3.0f;
__constant__ float cs2 = 1.0f / 3.0f;

__constant__ float d2q5_w[5] = { 1.0 / 3.0, 1.0 / 6.0,1.0 / 6.0,1.0 / 6.0 ,1.0 / 6.0 };
// crt order                           0  1  2  3  4  5  6  7  8  
//__constant__   float ex3d_gpu[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1};
// __constant__  float ey3d_gpu[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1};

// view for the crt order
__constant__ int map2crt[9] = {0,1,3,2,4,5,8,6,7};

__constant__ int inv_map2crt[9] = {0,1,3,2,4,5,7,8,6};

__constant__ int index2dInv_gpu_crt[9] = { 0,3,4,1,2,7,8,5,6};
__constant__ float w3d_gpu_crt[9] = {4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
                                  1.0 / 9.0,  1.0 / 9.0,  1.0 / 36.0,
                                  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };

__constant__ float s_crt[9] = { 0, 2.0, 2.0, 1, 1.9988, 1.9988,  1, 1, 1 };

__constant__ float s_crt_w[9] = { 0, 2.0, 2.0, 1.998, 1.998, 1.998, 1.941, 1.941, 1.941};




// Our stencil for D2Q9

//   8 3 5
//   2 0 1
//   6 4 7


__constant__ enum BinaryDirection
{
	Bin_C = 1 << 0,  //!< Center
	Bin_E = 1 << 1,  //!< East
	Bin_W = 1 << 2,  //!< West
	Bin_N = 1 << 3,  //!< North
	Bin_S = 1 << 4,  //!< South
	Bin_NE = 1 << 5,  //!< North-East
	Bin_SW = 1 << 6,  //!< Sorth-West
	Bin_SE = 1 << 7,  //!< South-East
	Bin_NW = 1 << 8,  //!< Nouth-West
};

//need to fix for the D2Q9
__constant__ BinaryDirection dirToBinary[9] = {
   Bin_C, Bin_E, Bin_W, Bin_N, Bin_S, Bin_NE, Bin_SW, Bin_SE, Bin_NW
};

// record the neighbors of neighbors
__constant__ int dir_neighbors[9][8] = { {1, 2, 3, 4, 5, 6, 7, 8},
										{0, 3, 4, 5, 7,-1,-1,-1},
										{0, 3, 4, 6, 8,-1,-1,-1},
										{0, 1, 2, 5, 8,-1,-1,-1},
										{0, 1, 2, 6, 7,-1,-1,-1},
										{0, 1, 3,-1,-1,-1,-1,-1},
										{0, 2, 4,-1,-1,-1,-1,-1},
										{0, 1, 4,-1,-1,-1,-1,-1},
										{0, 2, 3,-1,-1,-1,-1,-1} };

/**
 * \brief Length of the dir_neighbors array
 * For usage see documentation of dir_neighbors
 */
__constant__ int dir_neighbors_length[9] = { 8, 5, 5, 5, 5, 3, 3, 3, 3 };




#endif // !_MRCONSTANTPARAMSGPU2DH_
