#pragma once
#ifndef MRFLOW2DH_
#define MRFLOW2DH_

#include "../../../common/mlFluid.h"
#include "../../../common/mlFluidParam.h"
#include "../../../common/mlLbmCommon.h"
#include "../../../common/mlCuRunTime.h"
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
	REAL* label_volume;
	REAL* label_init_volume;
	int label_num;
	unsigned int max_bubble_count = 1024;
	int bubble_count = 0;
};


class mrFlow2D
{
public:
	mrFlow2D();
	~mrFlow2D();
	MLLATTICENODE_SURFACE_FLAG* flag;// domain flag

	REAL* fMom;
	REAL* fMomPost;

	REAL* gMom;
	REAL* gMomPost;

	REAL* src;
	REAL* c_value;
	REAL* delta_g;

	MLFluidParam2D* param;

	REAL* forcex;
	REAL* forcey;
	REAL* mass;
	REAL* massex;
	REAL* phi;
	REAL* disjoin_force;

	bool merge_flag;
	bool split_flag;
	REAL* delta_phi;

	REAL vis_shear;
	cudaStream_t stream;
	long count = 0;

	// the property for bubble
	int* tag_matrix;

	int* previous_tag;
	int* previous_merge_tag;
	
	bool* merge_detector;
	unsigned char* input_matrix;
	unsigned int* label_matrix;

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
	this->bubble.label_volume = new REAL[max_bubble_num]{0.f};
	this->bubble.label_init_volume = new REAL[max_bubble_num]{0.f};
	this->bubble.label_num = -1;
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
	gMom = new REAL[5 * count];
	gMomPost = new REAL[5 * count];
	src = new REAL[count];
	c_value = new REAL[count];
	delta_g = new REAL[count];
	flag = new MLLATTICENODE_SURFACE_FLAG[count];
	forcex = new REAL[count];
	forcey = new REAL[count];
	mass = new REAL[count];
	massex = new REAL[count];
	phi = new REAL[count];
	disjoin_force = new REAL[count];	
	delta_phi = new REAL[count];
	tag_matrix = new int[count];
	previous_tag = new int[count];
	previous_merge_tag = new int[count];

	input_matrix = new unsigned char[count];
	label_matrix = new unsigned int[count];
	merge_detector = new bool[count];

	BubbleBufferInit(max_bubble_num);

	for (long y = 0; y < sample_y_count; y++)
	{
		for (long x = 0; x < sample_x_count; x++)
		{
			int num = y * sample_x_count + x;
			flag[num] = TYPE_G;
			forcex[num] = 0.f;
			forcey[num] = 0.f;
			src[num] = 0.f;
			c_value[num] = 0.f;
			delta_g[num] = 0.f;
			disjoin_force[num] = 0.f;
			mass[num] = 0.f;
			massex[num] = 0.f;
			phi[num] = 0.f;
			tag_matrix[num] = -1;
			previous_tag[num] = -1;
			previous_merge_tag[num] = -1;
			input_matrix[num] = 0;
			label_matrix[num] = 0;
			merge_detector[num] = false;
			delta_phi[num] = 0.f;
		}
	}
}


#endif // !MRFLOW2DH_
