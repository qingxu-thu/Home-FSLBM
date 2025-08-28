#pragma once
#ifndef _MLFLUIDPARAM_
#define _MLFLUIDPARAM_
#include "mlCoreWin.h"
#include <cuda_runtime.h>  // 或 <vector_types.h>



struct MLFluidParam3D
{
	float3 start_pt; //the start position of the simulation domain in 2D world coordinate system
	float3 end_pt; //the end position of the simulation domain in 2D world coordinate system
	long3 samples;     //the sample numbers in each dimensions
	long3 domian_size;     //the domain size
	float3 box_size;     //the domain size
	
	REAL delta_x;
	REAL delta_t;
	int scaleNum;
	REAL vis_shear;
	REAL vis_bulk;
	int validCount;

	size_t NUMBER_GHOST_FACE_XY;
	size_t NUMBER_GHOST_FACE_XZ;
	size_t NUMBER_GHOST_FACE_YZ;
	size_t NUM_BLOCK_X;
	size_t NUM_BLOCK_Y;
	size_t NUM_BLOCK_Z;


	REAL gx;
	REAL gy;
	REAL gz;
};



struct MLFluidParam2D
{
	float3 start_pt; //the start position of the simulation domain in 2D world coordinate system
	float3 end_pt; //the end position of the simulation domain in 2D world coordinate system
	long3 samples;     //the sample numbers in each dimensions
	long3 domian_size;     //the domain size
	float3 box_size;     //the domain size
 
	REAL delta_x;
	REAL delta_t;

	int scaleNum;
	REAL vis_shear;
	REAL vis_bulk;
	int validCount;
	REAL gx;
	REAL gy;
};

 
struct MLMappingParam
{
	REAL lp, tp, xp;
	REAL t0p, l0p;
	REAL N;
	REAL u0p;
	REAL viscosity_p;
	REAL viscosity_k;
	REAL labma;
	REAL roup;
public:
	MLMappingParam()
	{}
	MLMappingParam
	(
		REAL _uop,
		REAL _labma,
		REAL _l0p,
		REAL _N,
		REAL _roup
	)
	{
		u0p = _uop;
		labma = _labma;
		l0p = _l0p;
		N = _N;
		roup = _roup;
	}
};

#endif // !_MLFLUIDPARAM_
