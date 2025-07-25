#pragma once
#ifndef MRFLOW3DH_
#define MRFLOW3DH_

#include "../../../common/mlFluid.h"
#include "../../../common/mlFluidParam.h"
#include "../../../common/mlLbmCommon.h"
#include "../../../common/mlCuRunTime.h"
#include "../../../3rdParty/helper_cuda.h"
#include "../../../common/mlCommon.h"

#include <fstream>
#include <iostream>
#include <vector>


struct mlBubble3D
{
	double* volume;
	double* init_volume;
	double* rho;
	// REAL* volume_diff;

	double* label_init_volume;

	// float3* center;
	// REAL* disjoint_pressure;
	double* label_volume;
	int label_num;
	// REAL *T;
	unsigned int max_bubble_count = 65536;
	int bubble_count = 0;
	int* freeze;
	float* pure_gas_volume;
	float* pure_label_gas_volume;
};



class mrFlow3D
{
public:
	mrFlow3D();
	~mrFlow3D();
	MLLATTICENODE_SURFACE_FLAG* flag;// domain flag
	MLLATTICENODE_SURFACE_FLAG* postflag;

	/*
	* rhoVar 0
	* uxVar  1
	* uyVar  2
	* pixx   3
	* pixy   4
	* piyy   5
	*/
	REAL* fMom;
	REAL* fMomPost;
	REAL* fMomViewer;
	MLFluidParam3D* param;
	int* cutcell;
	int* interp;
	float3* u_interp;
	REAL* forcex;
	REAL* forcey;
	REAL* forcez;
	REAL* rho;
	REAL* mass;
	REAL* massex;
	REAL* phi;
	float3* u;

	REAL vis_shear;
	cudaStream_t stream;
	long count = 0;

	// the property for bubble
	int* tag_matrix;
	int* previous_tag;
	int* previous_merge_tag;
	unsigned char* input_matrix;
	int* label_matrix;


	bool* merge_detector;
	int merge_flag;
	int split_flag;



	REAL* delta_phi;
	mlBubble3D bubble;

	float* disjoin_force;
	int * islet;
	float* gMom;
	float* gMomPost;
	float* delta_g;

	float* c_value;
	float* src;

	void Create
	(
		REAL x0, REAL y0, REAL z0,
		long width, long height, long depth,
		REAL delta_x,
		REAL box_w, REAL box_h, REAL box_d,
		REAL vis, REAL gy
	);

	void BubbleBufferInit(int max_bubble_num);

private:

};

inline mrFlow3D::mrFlow3D()
{
}

inline mrFlow3D::~mrFlow3D()
{
}

inline void mrFlow3D::BubbleBufferInit(int max_bubble_num)
{
	this->bubble.bubble_count = 0;
	this->bubble.max_bubble_count = max_bubble_num;
	this->bubble.volume = new double[max_bubble_num]{ 0.f };
	this->bubble.init_volume = new double[max_bubble_num]{ 0.f };
	this->bubble.rho = new double[max_bubble_num]{ 0.f };
	this->bubble.label_init_volume = new double[max_bubble_num]{ 0.f };

	//this->bubble.center = new float3[max_bubble_num];
	//this->bubble.disjoint_pressure = new REAL[max_bubble_num * 9];
	this->bubble.label_volume = new double[max_bubble_num]{ 0.f };
	this->bubble.label_num = -1;
	this->bubble.freeze = new int[max_bubble_num] { 0 };
	this->bubble.pure_gas_volume = new float[max_bubble_num] { 0.f };
	this->bubble.pure_label_gas_volume = new float[max_bubble_num] { 0.f };
}

inline void mrFlow3D::Create(
	REAL x0, REAL y0, REAL z0,
	long width, long height, long depth,
	REAL delta_x,
	REAL box_w, REAL box_h, REAL box_d,
	REAL vis, REAL gy
)
{
	this->vis_shear = vis;
	param = new MLFluidParam3D[1];
	long sample_x_count = 0; long sample_y_count = 0; long sample_z_count = 0;
	REAL endx = 0; REAL endy = 0; REAL endz = 0;
	REAL i = 0;
	for (i = x0; i < box_w + x0; i += delta_x)
	{
		sample_x_count++;
	}
	endx = i - delta_x;
	for (i = y0; i < box_h + y0; i += delta_x)
	{
		sample_y_count++;
	}
	endy = i - delta_x;
	for (i = z0; i < box_d + z0; i += delta_x)
	{
		sample_z_count++;
	}
	endz = i - delta_x;

	count = sample_x_count * sample_y_count * sample_z_count;
	param->start_pt.x = x0;		param->start_pt.y = y0;		param->start_pt.z = z0;
	param->end_pt.x = endx;		param->end_pt.y = endy;		param->end_pt.z = endz;
	param->delta_x = delta_x;	param->delta_t = delta_x;
	param->validCount = count;	param->box_size.x = box_w; 	param->box_size.y = box_h;	param->box_size.z = box_d;
	param->domian_size.x = width;		param->domian_size.y = height;		param->domian_size.z = depth;
	param->samples.x = sample_x_count;	param->samples.y = sample_y_count;	param->samples.z = sample_z_count;
	param->gx = 0;
	param->gy = gy;
	param->gz = 0;

	fMom = new REAL[10 * count]{};
	fMomPost = new REAL[10 * count]{};
	fMomViewer = new REAL[10 * count];

	flag = new MLLATTICENODE_SURFACE_FLAG[count];
	postflag = new MLLATTICENODE_SURFACE_FLAG[count];
	forcex = new REAL[count];
	forcey = new REAL[count];
	forcez = new REAL[count];
	rho = new REAL[count];
	mass = new REAL[count];
	massex = new REAL[count];
	phi = new REAL[count];
	islet = new	int[count];
	u = new float3[count];

	cutcell = new int[27 * count] {};
	interp = new int[27 * count] {};
	u_interp = new float3[count]{ {-1.f,-1.f,-1.f} };

	// initialization for the bubble
	tag_matrix = new int[count];
	delta_phi = new float[count];
	previous_tag = new int[count];
	previous_merge_tag = new int[count];
	// tmp variable
	input_matrix = new unsigned char[count];
	label_matrix = new int[count];
	// view_label_matrix = new int[count];
	merge_detector = new bool[count];
	merge_flag = false;
	split_flag = false;

	BubbleBufferInit(65536);

	disjoin_force = new float[count];
	gMom = new float[7 * count] {};
	gMomPost = new float[7 * count] {};
	delta_g = new float[count] {};
	c_value = new float[count] {};
	src = new float[count] {};

	// count number; xyz; other initialization
	for (long z = 0; z < sample_z_count; z++)
	{
		for (long y = 0; y < sample_y_count; y++)
		{
			for (long x = 0; x < sample_x_count; x++)
			{
				long num = z * sample_y_count * sample_x_count + y * sample_x_count + x;
				flag[num] = TYPE_G;
				postflag[num] = TYPE_G;
				forcex[num] = 0.f;
				forcey[num] = 0.f;
				forcez[num] = 0.f;
				rho[num] = 1.f;
				mass[num] = 0.f;
				massex[num] = 0.f;
				phi[num] = 0.f;
				u[num].x = 0.f;
				u[num].y = 0.f;
				u[num].z = 0.f;
				islet[num] = 0;
				tag_matrix[num] = -1;
				previous_tag[num] = -1;
				previous_merge_tag[num] = -1;
				input_matrix[num] = 0;
				label_matrix[num] = 0;
				// view_label_matrix[num] = 0;
				merge_detector[num] = false;
				for (int j = 0; j < 7; j++)
				{
					gMom[num * 7 + j] = 0.f;
					gMomPost[num * 7 + j] = 0.f;
				}
				delta_phi[num] = 0.f;
				delta_g[num] = 0.f;
				c_value[num] = 0.f;
				src[num] = 0.f;
				disjoin_force[num] = 0.f;
			}
		}
	}
}


#endif // !MRFLOW3DH_
