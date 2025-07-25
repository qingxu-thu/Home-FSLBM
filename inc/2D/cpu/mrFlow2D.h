#pragma once
#ifndef MRFLOW2DH_
#define MRFLOW2DH_

#include "../../../common/mlFluid.h"
#include "../../../common/mlFluidParam.h"
#include "../../../common/mlLbmCommon.h"
#include "../../../common/mlCuRunTime.h"
//#include "../../../3rdParty/helper_cuda.h"
#include "../../../3rdParty/helper_cuda.h"
#include "../../../common/mlCommon.h"

#include <fstream>
#include <iostream>
#include <vector>

struct mlBubble2D
{
	REAL* volume;
	REAL* init_volume;
	REAL* rho;
	// REAL* volume_diff;

	// float3* center;
	// REAL* disjoint_pressure;
	REAL* label_volume;
	int label_num;
	// REAL *T;
	unsigned int max_bubble_count = 1024;
	int bubble_count = 0;
	int *freeze;
	float* pure_gas_volume;
	float* pure_label_gas_volume;
};


class mrFlow2D
{
public:
	mrFlow2D();
	~mrFlow2D();
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

	REAL* gMom;
	REAL* gMomPost;

	REAL* src;
	REAL* c_value;
	REAL* delta_g;

	REAL* fMomViewer;
	MLFluidParam2D* param;

	REAL* forcex;
	REAL* forcey;
	// REAL* forcez;
	REAL* rho;
	// REAL* postrho;
	REAL* mass;
	REAL* mass_surplus;
	REAL* mass_deficit;

	REAL* massex;
	REAL* phi;
	
	float3* u;

	REAL vis_shear;
	cudaStream_t stream;
	long count = 0;
	REAL* vis_velocity;

	// the property for bubble
	int* tag_matrix;

	int* previous_tag;
	int* previous_merge_tag;

	unsigned char* input_matrix;
	unsigned int* label_matrix;
	int* view_label_matrix;

	bool* split_detector;
	int2* split_record;
	int* split_tag_record;
	int split_record_length;

	bool* merge_detector;
	int2* merge_record;
	int merge_record_length;
	bool merge_flag;
	bool split_flag;
	REAL* delta_phi;
	mlBubble2D bubble;



	void Create
	(
		REAL x0, REAL y0,
		long width, long height,
		REAL delta_x,
		REAL box_w, REAL box_h,
		REAL vis, REAL gy, int max_bubble_num = 1024
	);
	void BubbleBufferInit(int max_bubble_num);
private:

};

inline mrFlow2D::mrFlow2D()
{
}

inline mrFlow2D::~mrFlow2D()
{
}

inline void mrFlow2D::BubbleBufferInit(int max_bubble_num)
{
	this->bubble.bubble_count = 0;
	this->bubble.max_bubble_count = max_bubble_num;
	this->bubble.volume = new REAL[max_bubble_num]{0.f};
	this->bubble.init_volume = new REAL[max_bubble_num]{0.f};
	this->bubble.rho = new REAL[max_bubble_num]{ 0.f };
	//this->bubble.center = new float3[max_bubble_num];
	//this->bubble.disjoint_pressure = new REAL[max_bubble_num * 9];
	this->bubble.label_volume = new REAL[max_bubble_num]{0.f};
	this->bubble.label_num = -1;
	this->bubble.freeze = new int[max_bubble_num]{ 0 };
	this->bubble.pure_gas_volume = new float[max_bubble_num]{ 0.f };
	this->bubble.pure_label_gas_volume = new float[max_bubble_num] { 0.f };
}


inline void mrFlow2D::Create(
	REAL x0, REAL y0,
	long width, long height,
	REAL delta_x,
	REAL box_w, REAL box_h,
	REAL vis, REAL gy, int max_bubble_num
)
{
	this->vis_shear = vis;
	param = new MLFluidParam2D[1];
	long sample_x_count = 0; long sample_y_count = 0;
	REAL endx = 0; REAL endy = 0;
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
	count = sample_x_count * sample_y_count;
	param->start_pt.x = x0;		param->start_pt.y = y0;
	param->end_pt.x = endx;		param->end_pt.y = endy;
	param->delta_x = delta_x;	param->delta_t = delta_x;
	param->validCount = count;	param->box_size.x = box_w; 	param->box_size.y = box_h;
	param->domian_size.x = width;		param->domian_size.y = height;
	param->samples.x = sample_x_count;	param->samples.y = sample_y_count;
	param->gx = 0;
	param->gy = gy;
	fMom = new REAL[6 * count];
	fMomPost = new REAL[6 * count];
	fMomViewer = new REAL[6 * count];

	gMom = new REAL[5 * count];
	gMomPost = new REAL[5 * count];
	src = new REAL[count];
	c_value = new REAL[count];
	delta_g = new REAL[count];
	flag = new MLLATTICENODE_SURFACE_FLAG[count];
	postflag = new MLLATTICENODE_SURFACE_FLAG[count];
	forcex = new REAL[count];
	forcey = new REAL[count];
	// forcez = new REAL[count];
	rho = new REAL[count];
	mass = new REAL[count];
	mass_surplus = new REAL[count];
	mass_deficit = new REAL[count];
	massex = new REAL[count];
	phi = new REAL[count];


	delta_phi = new REAL[count];
	// initialization for the bubble
	tag_matrix = new int[count];
	previous_tag = new int[count];

	previous_merge_tag = new int[count];
	// tmp variable
	input_matrix = new unsigned char[count];
	label_matrix = new unsigned int[count];
	view_label_matrix = new int[count];
	split_detector = new bool[count];
	split_record = new int2[count];
	split_tag_record = new int[count];
	merge_detector = new bool[count];
	merge_record = new int2[count * 8];
	merge_flag = false;
	split_flag = false;
	merge_record_length = 0;
	split_record_length = 0;

	BubbleBufferInit(max_bubble_num);

	u = new float3[count];

	for (long y = 0; y < sample_y_count; y++)
	{
		for (long x = 0; x < sample_x_count; x++)
		{
			int num = y * sample_x_count + x;
			flag[num] = TYPE_G;
			postflag[num] = TYPE_G;
			forcex[num] = 0.f;
			forcey[num] = 0.f;
			// forcez[num] = 0.f;
			rho[num] = 1.f;
			src[num] = 0.f;
			c_value[num] = 0.f;
			delta_g[num] = 0.f;

			mass[num] = 0.f;
			mass_surplus[num] = 0.f;
			mass_deficit[num] = 0.f;

			massex[num] = 0.f;
			phi[num] = 0.f;
			u[num].x = 0.f;
			u[num].y = 0.f;
			u[num].z = 0.f;

			tag_matrix[num] = -1;
			previous_tag[num] = -1;
			previous_merge_tag[num] = -1;
			input_matrix[num] = 0;
			label_matrix[num] = 0;
			view_label_matrix[num] = 0;

			split_detector[num] = false;
			split_record[num].x = -1;
			split_record[num].y = -1;
			split_tag_record[num] = -1;
			merge_detector[num] = false;
			for (int z = 0; z < 8; z++)
			{
				merge_record[num*8+z].x = -1;
				merge_record[num*8+z].y = -1;
			}
			delta_phi[num] = 0.f;
		}
	}
}


#endif // !MRFLOW2DH_
