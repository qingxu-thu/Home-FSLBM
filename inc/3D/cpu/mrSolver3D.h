#pragma once
#ifndef MRSOLVER3DH_
#define MRSOLVER3DH_

#include "../gpu/mrLbmSolverGpu3D.h"

#include "../../../common/colorramp.h"
#include "mlInit3D.h"


#include <fstream>
#include <filesystem>

using namespace Mfree;
using namespace std;

class mrSolver3D
{
public:
	mrSolver3D();
	~mrSolver3D();
	void AttachLbmHost(mrFlow3D* lbmvec);
	void AttachLbmDevice(mrFlow3D* lbmvec_dev);
	void AttachMapping(MLMappingParam& mapping);
	void mlInit();
	void mlTransData2Host();
	void mlTransData2Gpu();
	void mlInitGpu();
	void mlDeepCopy(mrFlow3D* mllbm_host, mrFlow3D* mllbm_dev);
	void mlVisVelocitySlice(long upw, long uph, int frame);
	void mlSavePPM(const char* filename, float* data, int mWidth, int mHeight);
	void mlTransResultantFandT2Host();
	void mlIterateCouplingGpu(int timestep);
	void mlVisForceSlice(long upw, long uph, int frame);
	void mlSavePhi(long upw, long uph, int frame);
public:

	mrFlow3D* lbmvec;
	mrFlow3D* lbm_dev_gpu;
	MLMappingParam mparam;
	int gpuId = 0;


private:
	mrInitHandler3D mlinithandler3d;

};

mrSolver3D::mrSolver3D()
{
}

mrSolver3D::~mrSolver3D()
{
}

inline void mrSolver3D::AttachLbmHost(mrFlow3D* lbmvec)
{
	this->lbmvec = lbmvec;
}

inline void mrSolver3D::AttachLbmDevice(mrFlow3D* lbmvec_dev)
{
	this->lbm_dev_gpu = lbmvec_dev;
}

inline void mrSolver3D::AttachMapping(MLMappingParam& mapping)
{
	this->mparam.l0p = mapping.l0p;
	this->mparam.N = mapping.N;

	this->mparam.u0p = mapping.u0p;
	this->mparam.labma = mapping.labma;

	this->mparam.tp = mapping.tp;
	this->mparam.roup = mapping.roup;

}


inline void mrSolver3D::mlInit()
{
	checkCudaErrors(cudaSetDevice(gpuId));
	mlinithandler3d.mlInitBoundaryCpu(lbmvec);
	mlinithandler3d.mlInitInlet(lbmvec);
	mlinithandler3d.mlInitFlowVarCpu(lbmvec);
}

inline void mrSolver3D::mlInitGpu()
{
	checkCudaErrors(cudaSetDevice(gpuId));
	mrInit3DGpu(lbm_dev_gpu, lbmvec->param);
}


inline void mrSolver3D::mlTransData2Host()
{
	mrFlow3D* mllbm_host = new mrFlow3D();
	checkCudaErrors(cudaSetDevice(gpuId));
	checkCudaErrors(_MLCuMemcpy(mllbm_host, lbm_dev_gpu[i], 1 * sizeof(mrFlow3D), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec->fMom, (mllbm_host->fMom), lbmvec->count * 10 * sizeof(REAL), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec->flag, (mllbm_host->flag), lbmvec->count * sizeof(MLLATTICENODE_SURFACE_FLAG), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec->mass, (mllbm_host->mass), lbmvec->count * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec->phi, (mllbm_host->phi), lbmvec->count * sizeof(float), cudaMemcpyDeviceToHost));
}


inline void mrSolver3D::mlTransData2Gpu()
{
	checkCudaErrors(cudaSetDevice(gpuId));
	if (lbm_dev_gpu != NULL)
	{
		checkCudaErrors(cudaFree(lbm_dev_gpu));
	}
	_MLCuMalloc((void**)&lbm_dev_gpu, sizeof(mrFlow3D));
	mlDeepCopy(lbmvec, lbm_dev_gpu);
}


inline void mrSolver3D::mlIterateCouplingGpu(int timestep)
{
	checkCudaErrors(cudaSetDevice(gpuId));
	coupling(lbm_dev_gpu, lbmvec->param, mparam.N, mparam.l0p, mparam.roup, mparam.labma, mparam.u0p, timestep);
	mrSolver3DGpu(lbm_dev_gpu, lbmvec->param, mparam.N, mparam.l0p,
		mparam.roup, mparam.labma, mparam.u0p, timestep);
}

inline void mrSolver3D::mlDeepCopy(mrFlow3D* mllbm_host, mrFlow3D* mllbm_dev)
{

	float* fMom_dev;
	float* fMomPost_dev;
	MLLATTICENODE_SURFACE_FLAG* flag_dev;
	MLFluidParam3D* param_dev;
	float* forcex_dev;
	float* forcey_dev;
	float* forcez_dev;

	float* mass_dev;
	float* massex_dev;
	float* phi_dev;


	//bubble field
	int* tag_matrix_dev;
	int* previous_tag_dev;
	int* previous_merge_tag_dev;
	unsigned char* input_matrix_dev;
	int* label_matrix_dev;

	bool* merge_detector_dev;

	int* merge_flag_dev;
	int* split_flag_dev;
	float* delta_phi_dev;
	mlBubble3D* bubble_dev;

	double* volume_dev;
	double* init_volume_dev;
	double* rhob_dev;

	double* label_volume_dev;
	double* label_init_volume_dev;
	int* label_num_dev;
	int* bubble_count_dev;

	float* disjoin_force_dev;

	float* gMom_dev;
	float* gMomPost_dev;
	float* delta_g_dev;

	float* c_value_dev;
	float* src_dev;

	int* islet_dev;
#pragma region MallocData

	checkCudaErrors(cudaMalloc(&fMom_dev, 10 * mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&fMomPost_dev, 10 * mllbm_host->count * sizeof(REAL)));

	checkCudaErrors(cudaMalloc(&flag_dev, mllbm_host->count * sizeof(MLLATTICENODE_SURFACE_FLAG)));
	checkCudaErrors(cudaMalloc(&param_dev, 1 * sizeof(MLFluidParam3D)));
	checkCudaErrors(cudaMalloc(&forcex_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&forcey_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&forcez_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&mass_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&massex_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&phi_dev, mllbm_host->count * sizeof(REAL)));


	checkCudaErrors(cudaMalloc(&tag_matrix_dev, mllbm_host->count * sizeof(int)));
	checkCudaErrors(cudaMalloc(&previous_tag_dev, mllbm_host->count * sizeof(int)));
	checkCudaErrors(cudaMalloc(&previous_merge_tag_dev, mllbm_host->count * sizeof(int)));
	checkCudaErrors(cudaMalloc(&input_matrix_dev, mllbm_host->count * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc(&label_matrix_dev, mllbm_host->count * sizeof(int)));


	checkCudaErrors(cudaMalloc((void**)&split_flag_dev, sizeof(int)));
	checkCudaErrors(cudaMalloc(&merge_detector_dev, mllbm_host->count * sizeof(bool)));
	checkCudaErrors(cudaMalloc((void**)&merge_flag_dev, sizeof(int)));
	checkCudaErrors(cudaMalloc(&delta_phi_dev, mllbm_host->count * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&bubble_count_dev, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&bubble_dev, sizeof(mlBubble3D)));
	checkCudaErrors(cudaMalloc(&volume_dev, mllbm_host->bubble.max_bubble_count * sizeof(double)));
	checkCudaErrors(cudaMalloc(&init_volume_dev, mllbm_host->bubble.max_bubble_count * sizeof(double)));
	checkCudaErrors(cudaMalloc(&rhob_dev, mllbm_host->bubble.max_bubble_count * sizeof(double)));
	checkCudaErrors(cudaMalloc(&label_volume_dev, mllbm_host->bubble.max_bubble_count * sizeof(double)));
	checkCudaErrors(cudaMalloc(&label_init_volume_dev, mllbm_host->bubble.max_bubble_count * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&label_num_dev, sizeof(int)));

	checkCudaErrors(cudaMalloc(&disjoin_force_dev, mllbm_host->count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&gMom_dev, 7 * mllbm_host->count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&gMomPost_dev, 7 * mllbm_host->count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&delta_g_dev, mllbm_host->count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&c_value_dev, mllbm_host->count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&src_dev, mllbm_host->count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&islet_dev, mllbm_host->count * sizeof(int)));


#pragma endregion

#pragma region MEMCPY

	checkCudaErrors(_MLCuMemcpy(fMom_dev, mllbm_host->fMom, 10 * mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(fMomPost_dev, mllbm_host->fMomPost, 10 * mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(flag_dev, mllbm_host->flag, mllbm_host->count * sizeof(MLLATTICENODE_SURFACE_FLAG), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(param_dev, mllbm_host->param, 1 * sizeof(MLFluidParam3D), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(forcex_dev, mllbm_host->forcex, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(forcey_dev, mllbm_host->forcey, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(forcez_dev, mllbm_host->forcez, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(mass_dev, mllbm_host->mass, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(massex_dev, mllbm_host->massex, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(phi_dev, mllbm_host->phi, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));

	checkCudaErrors(_MLCuMemcpy(tag_matrix_dev, mllbm_host->tag_matrix, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(previous_tag_dev, mllbm_host->previous_tag, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(previous_merge_tag_dev, mllbm_host->previous_merge_tag, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(input_matrix_dev, mllbm_host->input_matrix, mllbm_host->count * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(label_matrix_dev, mllbm_host->label_matrix, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(split_flag_dev, &(mllbm_host->split_flag), sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(_MLCuMemcpy(merge_detector_dev, mllbm_host->merge_detector, mllbm_host->count * sizeof(bool), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(merge_flag_dev, &(mllbm_host->merge_flag), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(delta_phi_dev, mllbm_host->delta_phi, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(bubble_dev, &(mllbm_host->bubble), sizeof(mlBubble3D), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(volume_dev, mllbm_host->bubble.volume, mllbm_host->bubble.max_bubble_count * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(init_volume_dev, mllbm_host->bubble.init_volume, mllbm_host->bubble.max_bubble_count * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(rhob_dev, mllbm_host->bubble.rho, mllbm_host->bubble.max_bubble_count * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(label_volume_dev, mllbm_host->bubble.label_volume, mllbm_host->bubble.max_bubble_count * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(label_init_volume_dev, mllbm_host->bubble.label_init_volume, mllbm_host->bubble.max_bubble_count * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(label_num_dev, &(mllbm_host->bubble.label_num), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(bubble_count_dev, &(mllbm_host->bubble.bubble_count), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(disjoin_force_dev, mllbm_host->disjoin_force, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(gMom_dev, mllbm_host->gMom, 7 * mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(gMomPost_dev, mllbm_host->gMomPost, 7 * mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(delta_g_dev, mllbm_host->delta_g, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(c_value_dev, mllbm_host->c_value, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(src_dev, mllbm_host->src, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(islet_dev, mllbm_host->islet, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));


#pragma endregion
	checkCudaErrors(_MLCuMemcpy(mllbm_dev, mllbm_host, 1 * sizeof(mrFlow3D), cudaMemcpyHostToDevice));

#pragma region DeepCOPY

	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->fMom), &fMom_dev, sizeof(fMom_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->fMomPost), &fMomPost_dev, sizeof(fMomPost_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->flag), &flag_dev, sizeof(flag_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->param), &param_dev, sizeof(param_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->forcex), &forcex_dev, sizeof(forcex_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->forcey), &forcey_dev, sizeof(forcey_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->forcez), &forcez_dev, sizeof(forcez_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->mass), &mass_dev, sizeof(mass_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->massex), &massex_dev, sizeof(massex_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->phi), &phi_dev, sizeof(phi_dev), cudaMemcpyHostToDevice));

	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->tag_matrix), &tag_matrix_dev, sizeof(tag_matrix_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->previous_tag), &previous_tag_dev, sizeof(previous_tag_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->previous_merge_tag), &previous_merge_tag_dev, sizeof(previous_merge_tag_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->input_matrix), &input_matrix_dev, sizeof(input_matrix_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->label_matrix), &label_matrix_dev, sizeof(label_matrix_dev), cudaMemcpyHostToDevice));

	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->split_flag), split_flag_dev, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->merge_detector), &merge_detector_dev, sizeof(merge_detector_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->merge_flag), merge_flag_dev, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->delta_phi), &delta_phi_dev, sizeof(delta_phi_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble), bubble_dev, sizeof(mlBubble3D), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.volume), &volume_dev, sizeof(volume_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.init_volume), &init_volume_dev, sizeof(init_volume_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.rho), &rhob_dev, sizeof(rhob_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.label_volume), &label_volume_dev, sizeof(label_volume_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.label_init_volume), &label_init_volume_dev, sizeof(label_init_volume_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.label_num), label_num_dev, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.bubble_count), bubble_count_dev, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->disjoin_force), &disjoin_force_dev, sizeof(disjoin_force_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->gMom), &gMom_dev, sizeof(gMom_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->gMomPost), &gMomPost_dev, sizeof(gMomPost_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->delta_g), &delta_g_dev, sizeof(delta_g_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->c_value), &c_value_dev, sizeof(c_value_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->src), &src_dev, sizeof(src_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->islet), &islet_dev, sizeof(islet_dev), cudaMemcpyHostToDevice));
#pragma endregion
}

inline void mrSolver3D::mlVisVelocitySlice(long upw, long uph, int frame)
{

	float* cutslice_ve = new float[1 * lbmvec->param->samples.x * lbmvec->param->samples.z];
	int total_num = lbmvec->param->samples.x * lbmvec->param->samples.y * lbmvec->param->samples.z;

	int stx = 0;
	int sty = 0;
	int stz = 0;
	int edx = lbmvec->param->samples.x;
	int edy = lbmvec->param->samples.y;
	int edz = lbmvec->param->samples.z;
	//int y = 344;
	int y = edy / 2;
	//int y = 37;
	for (int z = stz; z < edz; z++)
		for (int x = stx; x < edx; x++)
		{
			int curind = z * lbmvec->param->samples.y * lbmvec->param->samples.x + y * lbmvec->param->samples.x + x;
			float ux = 0, uy = 0, uz = 0, rho = 0;

			ux = lbmvec->fMom[1 * total_num + curind];
			uy = lbmvec->fMom[2 * total_num + curind];
			uz = lbmvec->fMom[3 * total_num + curind];
			rho = lbmvec->fMom[0 * total_num + curind];

			auto flag = lbmvec->flag[curind];
			auto mass = lbmvec->mass[curind];
			auto phi = lbmvec->phi[curind];

			if (flag == TYPE_F || flag == TYPE_I)
				cutslice_ve[num] = sqrt(ux*ux+uy*uy+uz*uz);
			else
				cutslice_ve[num] = 0.f;
			num++;
		}


	float* vertices = new float[upw * uph * 3];
	num = 0;

	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec->param->samples.z;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec->param->samples.x) x01 = lbmvec->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec->param->samples.z) y01 = lbmvec->param->samples.z - 1;

			int ind0 = x00 + y00 * lbmvec->param->samples.x;
			int ind1 = x01 + y00 * lbmvec->param->samples.x;
			int ind2 = x00 + y01 * lbmvec->param->samples.x;
			int ind3 = x01 + y01 * lbmvec->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[k * exz + ind0] + rateX * cutslice_ve[k * exz + ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[k * exz + ind2] + rateX * cutslice_ve[k * exz + ind3]);
			vv = vv / 0.07;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
		

		char filename[2048];
		sprintf(filename, "../dataMR3D/ppm_ve_home_test_f2_sphere/im%05d.ppm", frame);

		mlSavePPM(filename, vertices, upw, uph);

		delete[] vertices;
	}
	delete[] cutslice_ve;

	// estimate the mass and max velocity
	float mass_tot = 0.f;
	float max_mass = 0.f;
	float max_vel = 0.f;
	int pivot = -1;
	int pivot_vel = -1;

	stx = 0;
	sty = 0;
	stz = 0;
	edx = lbmvec->param->samples.x;
	edy = lbmvec->param->samples.y;
	edz = lbmvec->param->samples.z;


	for (int z = stz; z < edz; z++)
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = z * lbmvec->param->samples.y * lbmvec->param->samples.x + y * lbmvec->param->samples.x + x;
				float mass = 0.f;
				float ux = 0, uy = 0, uz = 0, rho = 0;
				ux = lbmvec->fMom[1 * total_num + curind];
				uy = lbmvec->fMom[2 * total_num + curind];
				uz = lbmvec->fMom[3 * total_num + curind];
				rho = lbmvec->fMom[0 * total_num + curind];
				auto flag = lbmvec->flag[curind];
				float vel = 0.f;
				if (flag == TYPE_F || flag == TYPE_I)
					vel = sqrt(ux * ux + uy * uy + uz * uz);
				else
					vel = 0.f;
				pivot_vel = vel > max_vel ? y : pivot_vel;
				max_vel = max(vel, max_vel);
				if (flag == TYPE_F || flag == TYPE_I)
					mass = lbmvec->mass[curind];
				else
					mass = 0.f;
				mass_tot += mass;
				pivot = mass > max_mass ? y : pivot;
				max_mass = max(max_mass, mass);
				num++;
			}
	
	cout << "mass_tot:" << mass_tot << " max mass: " << max_mass << " y: " << pivot << endl;
	cout << "max vel:" << max_vel << " y: " << pivot_vel << endl;


}

inline void mrSolver3D::mlVisMassSlice(long upw, long uph, int frame)
{
	float* cutslice_ve = new float[1 * lbmvec->param->samples.x * lbmvec->param->samples.z];
	int total_num = lbmvec->param->samples.x * lbmvec->param->samples.y * lbmvec->param->samples.z;
	int stx = 0;
	int sty = 0;
	int stz = 0;
	int edx = lbmvec->param->samples.x;
	int edy = lbmvec->param->samples.y;
	int edz = lbmvec->param->samples.z;
	int y = edy / 2;
	for (int z = stz; z < edz; z++)
		for (int x = stx; x < edx; x++)
		{
			int curind = z * lbmvec->param->samples.y * lbmvec->param->samples.x + y * lbmvec->param->samples.x + x;
			float ux = 0, uy = 0, uz = 0, rho = 0;

			ux = lbmvec->fMom[1 * total_num + curind];
			uy = lbmvec->fMom[2 * total_num + curind];
			uz = lbmvec->fMom[3 * total_num + curind];
			rho = lbmvec->fMom[0 * total_num + curind];


			auto flag = lbmvec->flag[curind];
			auto mass = lbmvec->mass[curind];
			auto phi = lbmvec->phi[curind];

			if (flag == TYPE_F || flag == TYPE_I)
				cutslice_ve[num] = mass;
			else
				cutslice_ve[num] = 0.f;

			num++;
		}

	float* vertices = new float[upw * uph * 3];
	num = 0;
	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec->param->samples.z;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec->param->samples.x) x01 = lbmvec->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec->param->samples.z) y01 = lbmvec->param->samples.z - 1;

			int ind0 = x00 + y00 * lbmvec->param->samples.x;
			int ind1 = x01 + y00 * lbmvec->param->samples.x;
			int ind2 = x00 + y01 * lbmvec->param->samples.x;
			int ind3 = x01 + y01 * lbmvec->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[k * exz + ind0] + rateX * cutslice_ve[k * exz + ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[k * exz + ind2] + rateX * cutslice_ve[k * exz + ind3]);
			vv = vv / 1.0;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}
	char filename[2048];
	sprintf(filename, "../dataMR3D/ppm_ve_home_test_mass_f/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] vertices;
	delete[] cutslice_ve;
}



inline void mrSolver3D::mlSavePhi(long upw, long uph, int frame)
{
	char filename[2048];
	sprintf(filename, "../dataMR3D/ppm_ve_home_test_phi/phi%05d.bin", frame);
	std::filesystem::path filePath(filename);
	std::filesystem::path directory = filePath.parent_path();


	if (!directory.empty() && !std::filesystem::exists(directory)) {
		std::filesystem::create_directories(directory);
	}
	FILE* fp = fopen(filename, "wb");
	int Nx = lbmvec->param->samples.x;
	int Ny = lbmvec->param->samples.y;
	int Nz = lbmvec->param->samples.z;
	fwrite(&Nx, sizeof(float), 1, fp);
	fwrite(&Ny, sizeof(float), 1, fp);
	fwrite(&Nz, sizeof(float), 1, fp);
	fwrite(lbmvec->phi, sizeof(float), lbmvec->count, fp);
	fclose(fp);
}



inline void mrSolver3D::mlSavePPM(const char* filename, float* data, int mWidth, int mHeight)
{

	std::filesystem::path filePath(filename);
	std::filesystem::path directory = filePath.parent_path();


	if (!directory.empty() && !std::filesystem::exists(directory)) {
		std::filesystem::create_directories(directory);
	}

	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs << "P6\n" << mWidth << " " << mHeight << "\n255\n";
	for (unsigned i = 0; i < mWidth * mHeight * 3; ++i) {
		ofs << (unsigned char)(data[i] * 255);
	}
	ofs.close();
}


#endif // !MRSOLVER3DH_
