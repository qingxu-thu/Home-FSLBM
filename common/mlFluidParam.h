#pragma once
#ifndef _MLFLUIDPARAM_
#define _MLFLUIDPARAM_
#include "mlCoreWin.h"
#include "mlvertex.h"
#include "mlDataType.h"
struct MLFluidParam3D
{
	mlVertex3f start_pt; //the start position of the simulation domain in 2D world coordinate system
	mlVertex3f end_pt; //the end position of the simulation domain in 2D world coordinate system
	GVLSize3l samples;     //the sample numbers in each dimensions
	GVLSize3l domian_size;     //the domain size
	GVLSize3f box_size;     //the domain size
	
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
	mlVertex2f start_pt; //the start position of the simulation domain in 2D world coordinate system
	mlVertex2f end_pt; //the end position of the simulation domain in 2D world coordinate system
	GVLSize2l samples;     //the sample numbers in each dimensions
	GVLSize2l domian_size;     //the domain size
	GVLSize2f box_size;     //the domain size
 
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
