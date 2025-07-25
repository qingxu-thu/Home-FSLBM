#pragma once
#ifndef MRSOLVER2DH_
#define MRSOLVER2DH_

#include "../gpu/mrLbmSolverGpu2D.h"

#include "../../../common/colorramp.h"
#include "mlInit2D.h"

#include <fstream>
#include <filesystem>

using namespace Mfree;
using namespace std;

class mrSolver2D
{
public:
	mrSolver2D();
	~mrSolver2D();
	void AttachLbmHost(std::vector<mrFlow2D*> lbmvec);
	void AttachLbmDevice(std::vector<mrFlow2D*> lbmvec_dev);
	void AttachMapping(MLMappingParam& mapping);
	void mlInit();
	void mlIterateGpu(int time_step);
	void mlTransData2Host(int i);
	void mlTransData2Gpu();
	void mlInitGpu();
	void mlDeepCopy(mrFlow2D* mllbm_host, mrFlow2D* mllbm_dev, int i);
	void mlVisVelocitySlice(long upw, long uph, int scaleNum, int frame);
	void mlVisTypeSlice(long upw, long uph, int scaleNum, int frame);
	void mlVisTagSlice(long upw, long uph, int scaleNum, int frame);
	void mlSavePPM(const char* filename, float* data, int mWidth, int mHeight);
	void mlVisMassSlice(long upw, long uph, int scaleNum, int frame);
	void mlVisDelatMassSlice(long upw, long uph, int scaleNum, int frame);
	void mlVisRhoSlice(long upw, long uph, int scaleNum, int frame);
	void mlVisCurSlice(long upw, long uph, int scaleNum, int frame);
	void mlVisDisjoinSlice(long upw, long uph, int scaleNum, int frame);
	void mlSavePhi(long upw, long uph, int scaleNum, int frame);
public:

	std::vector<mrFlow2D*> lbmvec;
	std::vector<mrFlow2D*> lbm_dev_gpu;
	MLMappingParam mparam;
	float L;
	int gpuId = 0;
private:
	mrInitHandler2D mlinithandler2d;

};

mrSolver2D::mrSolver2D()
{
}

mrSolver2D::~mrSolver2D()
{
}

inline void mrSolver2D::AttachLbmHost(std::vector<mrFlow2D*> lbmvec)
{
	this->lbmvec = lbmvec;
}

inline void mrSolver2D::AttachLbmDevice(std::vector<mrFlow2D*> lbmvec_dev)
{
	this->lbm_dev_gpu = lbmvec_dev;
}



inline void mrSolver2D::AttachMapping(MLMappingParam& mapping)
{
	this->mparam.l0p = mapping.l0p;
	this->mparam.N = mapping.N;

	this->mparam.u0p = mapping.u0p;
	this->mparam.labma = mapping.labma;

	this->mparam.tp = mapping.tp;
	this->mparam.roup = mapping.roup;

}



inline void mrSolver2D::mlInit()
{
	checkCudaErrors(cudaSetDevice(gpuId));
	for (int i = 0; i < lbmvec.size(); i++)
	{
		mlinithandler2d.mlInitBoundaryCpu(lbmvec, i, L);
		mlinithandler2d.mlInitFlowVarCpu(lbmvec, i, L);
		mlinithandler2d.mlInitBubbleCpu(lbmvec, 5, 0.03, i, L);
	}
}

inline void mrSolver2D::mlInitGpu()
{
	checkCudaErrors(cudaSetDevice(gpuId));
	for (int i = 0; i < lbmvec.size(); i++)
	{
		mrInit2DGpu(lbm_dev_gpu[i], lbmvec[i]->param);
	}
}

inline void mrSolver2D::mlIterateGpu(int time_step)
{
	checkCudaErrors(cudaSetDevice(gpuId));
	for (int i = 0; i < lbmvec.size(); i++)
	{
		mrSolver2DGpu(lbm_dev_gpu[i], lbmvec[i]->param, time_step);
	}
}

inline void mrSolver2D::mlTransData2Host(int i)
{
	mrFlow2D* mllbm_host = new mrFlow2D();
	checkCudaErrors(cudaSetDevice(gpuId));
	checkCudaErrors(_MLCuMemcpy(mllbm_host, lbm_dev_gpu[i], 1 * sizeof(mrFlow2D), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->fMom, (mllbm_host->fMom), lbmvec[i]->count * 6 * sizeof(REAL), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->fMomViewer, (mllbm_host->fMomViewer), lbmvec[i]->count * 6 * sizeof(REAL), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->flag, (mllbm_host->flag), lbmvec[i]->count * sizeof(MLLATTICENODE_SURFACE_FLAG), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->mass, (mllbm_host->mass), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->mass_surplus, (mllbm_host->mass_surplus), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->mass_deficit, (mllbm_host->mass_deficit), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->massex, (mllbm_host->massex), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->delta_g, (mllbm_host->delta_g), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->phi, (mllbm_host->phi), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->tag_matrix, (mllbm_host->tag_matrix), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->view_label_matrix, (mllbm_host->view_label_matrix), lbmvec[i]->count * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->bubble.rho, mllbm_host->bubble.rho, mllbm_host->bubble.max_bubble_count * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->c_value, mllbm_host->c_value, mllbm_host->count * sizeof(float), cudaMemcpyDeviceToHost));
}

inline void mrSolver2D::mlTransData2Gpu()
{
	checkCudaErrors(cudaSetDevice(gpuId));
	for (int i = 0; i < lbmvec.size(); i++)
	{
		if (lbm_dev_gpu[i] != NULL)
		{
			checkCudaErrors(cudaFree(lbm_dev_gpu[i]));
		}
		_MLCuMalloc((void**)&lbm_dev_gpu[i], sizeof(mrFlow2D));
		mlDeepCopy(lbmvec[i], lbm_dev_gpu[i], i);
	}
}

inline void mrSolver2D::mlDeepCopy(mrFlow2D* mllbm_host, mrFlow2D* mllbm_dev, int i)
{

	float* fMom_dev;
	float* fMomPost_dev;
	float* fMomViewer_dev;

	float* gMom_dev;
	float* gMomPost_dev;

	float* src_dev;
	float* c_value_dev;
	float* delta_g_dev;

	MLLATTICENODE_SURFACE_FLAG* flag_dev;
	MLLATTICENODE_SURFACE_FLAG* postflag_dev;

	MLFluidParam2D* param_dev;
	float* forcex_dev;
	float* forcey_dev;

	float3* u_dev;
	float* rho_dev;
	float* mass_dev;
	float* mass_surplus_dev;
	float* mass_deficit_dev;
	float* massex_dev;
	float* phi_dev;

	int* tag_matrix_dev;
	int* previous_tag_dev;
	int* previous_merge_tag_dev;
	unsigned char* input_matrix_dev;
	unsigned int* label_matrix_dev;
	int* view_label_matrix_dev;

	bool* split_detector_dev;
	int2* split_record_dev;
	int* split_tag_record_dev;
	int* split_record_length_dev;

	bool* merge_detector_dev;
	int2* merge_record_dev;
	int* merge_record_length_dev;

	bool* merge_flag_dev;
	bool* split_flag_dev;
	float* delta_phi_dev;
	mlBubble2D* bubble_dev;

	REAL* volume_dev;
	REAL* init_volume_dev;
	REAL* rhob_dev;
	int* freeze_dev;

	float* pure_gas_volume_dev;
	float* pure_label_gas_volume_dev;
	// REAL* volume_diff;

	REAL* label_volume_dev;
	int* label_num_dev;
	int* bubble_count_dev;




#pragma region MallocData

	checkCudaErrors(cudaMalloc(&fMom_dev, 6 * mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&fMomPost_dev, 6 * mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&fMomViewer_dev, 6 * mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&gMom_dev, 5 * mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&gMomPost_dev, 5 * mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&src_dev, mllbm_host->count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&c_value_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&delta_g_dev, mllbm_host->count * sizeof(REAL)));

	checkCudaErrors(cudaMalloc(&flag_dev, mllbm_host->count * sizeof(MLLATTICENODE_SURFACE_FLAG)));
	checkCudaErrors(cudaMalloc(&postflag_dev, mllbm_host->count * sizeof(MLLATTICENODE_SURFACE_FLAG)));


	checkCudaErrors(cudaMalloc(&param_dev, 1 * sizeof(MLFluidParam2D)));
	checkCudaErrors(cudaMalloc(&forcex_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&forcey_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&rho_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&mass_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&mass_surplus_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&mass_deficit_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&massex_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&phi_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&u_dev, mllbm_host->count * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&tag_matrix_dev, mllbm_host->count * sizeof(int)));
	checkCudaErrors(cudaMalloc(&previous_tag_dev, mllbm_host->count * sizeof(int)));
	checkCudaErrors(cudaMalloc(&previous_merge_tag_dev, mllbm_host->count * sizeof(int)));
	checkCudaErrors(cudaMalloc(&input_matrix_dev, mllbm_host->count * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc(&label_matrix_dev, mllbm_host->count * sizeof(unsigned int)));

	checkCudaErrors(cudaMalloc(&view_label_matrix_dev, mllbm_host->count * sizeof(int)));

	checkCudaErrors(cudaMalloc(&split_detector_dev, mllbm_host->count * sizeof(bool)));
	checkCudaErrors(cudaMalloc(&split_record_dev, mllbm_host->count * sizeof(int2)));
	checkCudaErrors(cudaMalloc(&split_tag_record_dev, mllbm_host->count * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&split_record_length_dev, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&split_flag_dev, sizeof(bool)));
	checkCudaErrors(cudaMalloc(&merge_detector_dev, mllbm_host->count * sizeof(bool)));
	checkCudaErrors(cudaMalloc(&merge_record_dev, 8 * mllbm_host->count * sizeof(int2)));
	checkCudaErrors(cudaMalloc((void**)&merge_record_length_dev, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&merge_flag_dev, sizeof(bool)));
	checkCudaErrors(cudaMalloc(&delta_phi_dev, mllbm_host->count * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&bubble_count_dev, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&bubble_dev, sizeof(mlBubble2D)));
	checkCudaErrors(cudaMalloc(&volume_dev, mllbm_host->bubble.max_bubble_count*sizeof(float)));
	checkCudaErrors(cudaMalloc(&init_volume_dev, mllbm_host->bubble.max_bubble_count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&rhob_dev, mllbm_host->bubble.max_bubble_count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&freeze_dev, mllbm_host->bubble.max_bubble_count * sizeof(int)));
	checkCudaErrors(cudaMalloc(&pure_gas_volume_dev, mllbm_host->bubble.max_bubble_count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&pure_label_gas_volume_dev, mllbm_host->bubble.max_bubble_count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&label_volume_dev, mllbm_host->bubble.max_bubble_count * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&label_num_dev, sizeof(int)));


#pragma endregion

#pragma region MEMCPY

	checkCudaErrors(_MLCuMemcpy(fMom_dev, mllbm_host->fMom, 6 * mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(fMomPost_dev, mllbm_host->fMomPost, 6 * mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(fMomViewer_dev, mllbm_host->fMomViewer, 6 * mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(gMom_dev, mllbm_host->gMom, 5 * mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(gMomPost_dev, mllbm_host->gMomPost, 5 * mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(src_dev, mllbm_host->src, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(c_value_dev, mllbm_host->c_value, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(delta_g_dev, mllbm_host->delta_g, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));


	checkCudaErrors(_MLCuMemcpy(flag_dev, mllbm_host->flag, mllbm_host->count * sizeof(MLLATTICENODE_SURFACE_FLAG), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(postflag_dev, mllbm_host->postflag, mllbm_host->count * sizeof(MLLATTICENODE_SURFACE_FLAG), cudaMemcpyHostToDevice));
	
	checkCudaErrors(_MLCuMemcpy(param_dev, mllbm_host->param, 1 * sizeof(MLFluidParam2D), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(forcex_dev, mllbm_host->forcex, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(forcey_dev, mllbm_host->forcey, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(rho_dev, mllbm_host->rho, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(mass_dev, mllbm_host->mass, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(mass_surplus_dev, mllbm_host->mass_surplus, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(mass_deficit_dev, mllbm_host->mass_deficit, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));

	checkCudaErrors(_MLCuMemcpy(massex_dev, mllbm_host->massex, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(phi_dev, mllbm_host->phi, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(u_dev, mllbm_host->u, mllbm_host->count * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(tag_matrix_dev, mllbm_host->tag_matrix, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(previous_tag_dev, mllbm_host->previous_tag, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(previous_merge_tag_dev, mllbm_host->previous_merge_tag, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(input_matrix_dev, mllbm_host->input_matrix, mllbm_host->count * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(label_matrix_dev, mllbm_host->label_matrix, mllbm_host->count * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(view_label_matrix_dev, mllbm_host->view_label_matrix, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(split_detector_dev, mllbm_host->split_detector, mllbm_host->count * sizeof(bool), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(split_record_dev, mllbm_host->split_record, mllbm_host->count * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(split_tag_record_dev, mllbm_host->split_tag_record, mllbm_host->count * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(_MLCuMemcpy(split_record_length_dev, &(mllbm_host->split_record_length), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(split_flag_dev, &(mllbm_host->split_flag), sizeof(bool), cudaMemcpyHostToDevice));

	checkCudaErrors(_MLCuMemcpy(merge_detector_dev, mllbm_host->merge_detector, mllbm_host->count * sizeof(bool), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(merge_record_dev, mllbm_host->merge_record, 8 * mllbm_host->count * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(merge_record_length_dev, &(mllbm_host->merge_record_length), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(merge_flag_dev, &(mllbm_host->merge_flag), sizeof(bool), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(delta_phi_dev, mllbm_host->delta_phi, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(bubble_dev, &(mllbm_host->bubble), sizeof(mlBubble2D), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(volume_dev, mllbm_host->bubble.volume, mllbm_host->bubble.max_bubble_count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(init_volume_dev, mllbm_host->bubble.init_volume, mllbm_host->bubble.max_bubble_count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(rhob_dev, mllbm_host->bubble.rho, mllbm_host->bubble.max_bubble_count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(freeze_dev, mllbm_host->bubble.freeze, mllbm_host->bubble.max_bubble_count * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(pure_gas_volume_dev, mllbm_host->bubble.pure_gas_volume, mllbm_host->bubble.max_bubble_count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(pure_label_gas_volume_dev, mllbm_host->bubble.pure_label_gas_volume, mllbm_host->bubble.max_bubble_count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(label_volume_dev, mllbm_host->bubble.label_volume, mllbm_host->bubble.max_bubble_count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(label_num_dev, &(mllbm_host->bubble.label_num), sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(bubble_count_dev, &(mllbm_host->bubble.bubble_count), sizeof(int), cudaMemcpyHostToDevice));



#pragma endregion
	checkCudaErrors(_MLCuMemcpy(mllbm_dev, mllbm_host, 1 * sizeof(mrFlow2D), cudaMemcpyHostToDevice));

#pragma region DeepCOPY

	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->fMom), &fMom_dev, sizeof(fMom_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->fMomPost), &fMomPost_dev, sizeof(fMomPost_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->fMomViewer), &fMomViewer_dev, sizeof(fMomViewer_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->gMom), &gMom_dev, sizeof(gMom_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->gMomPost), &gMomPost_dev, sizeof(gMomPost_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->src), &src_dev, sizeof(src_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->c_value), &c_value_dev, sizeof(c_value_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->delta_g), &delta_g_dev, sizeof(delta_g_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->flag), &flag_dev, sizeof(flag_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->postflag), &postflag_dev, sizeof(postflag_dev), cudaMemcpyHostToDevice));
	

	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->param), &param_dev, sizeof(param_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->forcex), &forcex_dev, sizeof(forcex_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->forcey), &forcey_dev, sizeof(forcey_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->rho), &rho_dev, sizeof(rho_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->mass), &mass_dev, sizeof(mass_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->massex), &massex_dev, sizeof(massex_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->mass_surplus), &mass_surplus_dev, sizeof(mass_surplus_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->mass_deficit), &mass_deficit_dev, sizeof(mass_deficit_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->phi), &phi_dev, sizeof(phi_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->u), &u_dev, sizeof(u_dev), cudaMemcpyHostToDevice));

	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->tag_matrix), &tag_matrix_dev, sizeof(tag_matrix_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->previous_tag), &previous_tag_dev, sizeof(previous_tag_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->previous_merge_tag), &previous_merge_tag_dev, sizeof(previous_merge_tag_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->input_matrix), &input_matrix_dev, sizeof(input_matrix_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->label_matrix), &label_matrix_dev, sizeof(label_matrix_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->view_label_matrix), &view_label_matrix_dev, sizeof(view_label_matrix_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->split_detector), &split_detector_dev, sizeof(split_detector_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->split_record), &split_record_dev, sizeof(split_record_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->split_tag_record), &split_tag_record_dev, sizeof(split_tag_record_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->split_record_length), split_record_length_dev, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->split_flag), split_flag_dev, sizeof(bool), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->merge_detector), &merge_detector_dev, sizeof(merge_detector_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->merge_record), &merge_record_dev, sizeof(merge_record_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->merge_record_length), merge_record_length_dev, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->merge_flag), merge_flag_dev, sizeof(bool), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->delta_phi), &delta_phi_dev, sizeof(delta_phi_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble), bubble_dev, sizeof(mlBubble2D), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.volume), &volume_dev, sizeof(volume_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.init_volume), &init_volume_dev, sizeof(init_volume_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.rho), &rhob_dev, sizeof(rhob_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.freeze), &freeze_dev, sizeof(freeze_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.pure_gas_volume), &pure_gas_volume_dev, sizeof(pure_gas_volume_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.pure_label_gas_volume), &pure_label_gas_volume_dev, sizeof(pure_label_gas_volume_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.label_volume), &label_volume_dev, sizeof(label_volume_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.label_num), label_num_dev, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->bubble.bubble_count), bubble_count_dev, sizeof(int), cudaMemcpyHostToDevice));


#pragma endregion
}

inline void mrSolver2D::mlVisVelocitySlice(long upw, long uph, int scaleNum, int frame)
{
	int upnum = 0;
	int baseScale = 0;
	float* cutslice_ve = new float[lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y];
	int num = 0;
	int total_num = lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y;
	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				float ux = 0, uy = 0, rho = 0;
				
				//for (int j = 0; j < 9; j++)
				//{
				//	ux += lbmvec[0]->fMom[j * total_num + curind] * ex2d_cpu[j];
				//	uy += lbmvec[0]->fMom[j * total_num + curind] * ey2d_cpu[j];

				//	rho += lbmvec[0]->fMom[j * total_num + curind];

				//}

				rho = lbmvec[0]->fMom[curind * 6 + 0];
				ux = lbmvec[0]->fMom[curind * 6 + 1];
				uy = lbmvec[0]->fMom[curind * 6 + 2];

				auto flag = lbmvec[0]->flag[curind];
				auto mass = lbmvec[0]->mass[curind];
				auto massex = lbmvec[0]->massex[curind];
				auto phi = lbmvec[0]->phi[curind];
				auto tag = lbmvec[0]->tag_matrix[curind];
				auto label = lbmvec[0]->view_label_matrix[curind];
				// cutslice_ve[num] = (float)(tag+1)/70 * 0.5;
				/*if (num >= (399-97) * 400 + 2 && num <= (399 -97) * 400 + 7)
					std::cout << "tag: " << tag << std::endl;*/
					//cutslice_ve[num] = (float) (1+tag)/90 * 0.5;
					if ((flag!=TYPE_S&& flag != TYPE_G) && !(std::isnan(rho)))
						cutslice_ve[num] = sqrt(ux * ux + uy * uy)* (float)(flag == TYPE_F | flag == TYPE_I);
					else
						cutslice_ve[num] = 0;
					//
				//cutslice_ve[num] = fmax(-massex,0.f);
				//cutslice_ve[num] = (float)(flag==TYPE_I) / 30 * 0.5;
				//cutslice_ve[num] = (float)(1 + label) / 90 * 0.5;
				// std::cout << cutslice_ve[num] << std::endl;
				if (std::isnan(cutslice_ve[num])) {
					std::cout << "cutslice_ve[" << num << "] is NaN\n"<< "rho "<< rho <<"u  "<< ux * ux + uy * uy << std::endl;
					std::cout << "cutslice_ve[" << num << "] is NaN\n" << "mass " << mass << "phi  " << phi << std::endl;
					std::cout << "cutslice_ve[" << num << "] is NaN\n" << "flag " << (int)(flag == TYPE_F | flag == TYPE_I) << std::endl;
				}
				num++;
			}
	}
	float* vertices = new float[upw * uph * 3];
	num = 0;


	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec[0]->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec[0]->param->samples.y;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec[0]->param->samples.x) x01 = lbmvec[0]->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec[0]->param->samples.y) y01 = lbmvec[0]->param->samples.y - 1;

			int ind0 = x00 + y00 * lbmvec[0]->param->samples.x;
			int ind1 = x01 + y00 * lbmvec[0]->param->samples.x;
			int ind2 = x00 + y01 * lbmvec[0]->param->samples.x;
			int ind3 = x01 + y01 * lbmvec[0]->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[ind0] + rateX * cutslice_ve[ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[ind2] + rateX * cutslice_ve[ind3]);
			vv = vv / 0.1;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}

	char filename[2048];
	sprintf(filename, "../dataMR2D_bubble_home/ppm_ve/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] cutslice_ve;
	delete[] vertices;

	float mass_tot = 0.f;
	float max_mass = 0.f;
	float max_vel = 0.f;
	int pivot = -1;
	int pivot_vel = -1;

	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;

		//int y = edy / 2;

		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				//auto mass = lbmvec[0]->mass[curind];

				float ux = 0, uy = 0, rho = 0;
				//for (int j = 0; j < 9; j++)
				//{
				//	ux += lbmvec[0]->fMom[j * total_num + curind] * ex2d_cpu[j];
				//	uy += lbmvec[0]->fMom[j * total_num + curind] * ey2d_cpu[j];

				//	rho += lbmvec[0]->fMom[j * total_num + curind];

				//}
				rho = lbmvec[0]->fMom[curind * 6 + 0];
				ux = lbmvec[0]->fMom[curind * 6 + 1];
				uy = lbmvec[0]->fMom[curind * 6 + 2];
				auto flag = lbmvec[0]->flag[curind];
				float vel = 0.f;
				float mass = 0.f;
				if (flag != TYPE_S && flag != TYPE_G)
					vel = sqrt(ux * ux + uy * uy) / (1.f + rho) * (float)(flag == TYPE_F | flag == TYPE_I);
				mass = lbmvec[0]->mass[curind];
				//auto vel = sqrt(ux * ux + uy * uy) / (1.f + rho) * (float)(flag == TYPE_F | flag == TYPE_I);
				// mass = (float)(flag == TYPE_F | flag == TYPE_I);
				pivot_vel = vel > max_vel ? y : pivot_vel;
				max_vel = max(vel, max_vel);

				mass_tot += mass;
				pivot = mass > max_mass ? y : pivot;
				max_mass = max(max_mass, mass);
				num++;
			}
	}
	cout << "mass_tot:" << mass_tot << " max mass: " << max_mass << " y: " << pivot << endl;
	cout << "max vel:" << max_vel << " y: " << pivot_vel << endl;
}

inline void mrSolver2D::mlVisMassSlice(long upw, long uph, int scaleNum, int frame)
{
	int upnum = 0;
	int baseScale = 0;
	float* cutslice_ve = new float[lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y];
	int num = 0;
	int total_num = lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y;
	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				float ux = 0, uy = 0, rho = 0;
				//for (int j = 0; j < 9; j++)
				//{
				//	ux += lbmvec[0]->fMom[j * total_num + curind] * ex2d_cpu[j];
				//	uy += lbmvec[0]->fMom[j * total_num + curind] * ey2d_cpu[j];

				//	rho += lbmvec[0]->fMom[j * total_num + curind];

				//}
				rho = lbmvec[0]->fMom[curind * 6 + 0];
				ux = lbmvec[0]->fMom[curind * 6 + 1];
				uy = lbmvec[0]->fMom[curind * 6 + 2];

				auto flag = lbmvec[0]->flag[curind];
				auto mass = lbmvec[0]->mass[curind];
				auto phi = lbmvec[0]->phi[curind];
				auto tag = lbmvec[0]->tag_matrix[curind];
				auto label = lbmvec[0]->view_label_matrix[curind];
				// cutslice_ve[num] = (float)(tag+1)/70 * 0.5;
				/*if (num >= (399-97) * 400 + 2 && num <= (399 -97) * 400 + 7)
					std::cout << "tag: " << tag << std::endl;*/
					//cutslice_ve[num] = (float) (1+tag)/90 * 0.5;
				// cutslice_ve[num] = sqrt(ux * ux + uy * uy) / (1.f + rho) * (float)(flag == TYPE_F | flag == TYPE_I);
				//cutslice_ve[num] = (float)(flag==TYPE_S) / 30 * 0.5;
				if (flag == TYPE_F || flag == TYPE_I)
					cutslice_ve[num] = phi;
				else if (flag == TYPE_S)
					cutslice_ve[num] = 0.25;
				else
					cutslice_ve[num] = 0.f;
				//std::cout << cutslice_ve[num] << std::endl;
				num++;
			}
	}
	float* vertices = new float[upw * uph * 3];
	num = 0;


	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec[0]->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec[0]->param->samples.y;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec[0]->param->samples.x) x01 = lbmvec[0]->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec[0]->param->samples.y) y01 = lbmvec[0]->param->samples.y - 1;

			int ind0 = x00 + y00 * lbmvec[0]->param->samples.x;
			int ind1 = x01 + y00 * lbmvec[0]->param->samples.x;
			int ind2 = x00 + y01 * lbmvec[0]->param->samples.x;
			int ind3 = x01 + y01 * lbmvec[0]->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[ind0] + rateX * cutslice_ve[ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[ind2] + rateX * cutslice_ve[ind3]);
			vv = vv / 0.5;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}

	char filename[2048];
	sprintf(filename, "../dataMR2D_bubble_home/ppm_ve_mass/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] cutslice_ve;
	delete[] vertices;


}

inline void mrSolver2D::mlVisTypeSlice(long upw, long uph, int scaleNum, int frame)
{
	int upnum = 0;
	int baseScale = 0;
	float* cutslice_ve = new float[lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y];
	int num = 0;
	int total_num = lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y;
	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				float ux = 0, uy = 0, rho = 0;
				//for (int j = 0; j < 9; j++)
				//{
				//	ux += lbmvec[0]->fMom[j * total_num + curind] * ex2d_cpu[j];
				//	uy += lbmvec[0]->fMom[j * total_num + curind] * ey2d_cpu[j];

				//	rho += lbmvec[0]->fMom[j * total_num + curind];

				//}
				rho = lbmvec[0]->fMom[curind * 6 + 0];
				ux = lbmvec[0]->fMom[curind * 6 + 1];
				uy = lbmvec[0]->fMom[curind * 6 + 2];
				auto flag = lbmvec[0]->flag[curind];
				auto mass = lbmvec[0]->mass[curind];
				auto phi = lbmvec[0]->phi[curind];
				auto tag = lbmvec[0]->tag_matrix[curind];
				auto label = lbmvec[0]->view_label_matrix[curind];
				auto mass_deficit = lbmvec[0]->mass_deficit[curind];
				// cutslice_ve[num] = (float)(tag+1)/70 * 0.5;
				/*if (num >= (399-97) * 400 + 2 && num <= (399 -97) * 400 + 7)
					std::cout << "tag: " << tag << std::endl;*/
					//cutslice_ve[num] = (float) (1+tag)/90 * 0.5;

					//cutslice_ve[num] = (float)(flag==TYPE_I) / 30 * 0.5;
				//if (flag == TYPE_I)
				//	cutslice_ve[num] = std::abs(lbmvec[0]->bubble.rho[tag - 1] - 1.f);
				//else
				//	cutslice_ve[num] = 0.f;
				/*else
					cutslice_ve[num] = (float)(flag == TYPE_I) / 10 * 0.2;*/
					//std::cout << cutslice_ve[num] << std::endl;

				cutslice_ve[num] = mass_deficit;
				num++;
			}
	}
	float* vertices = new float[upw * uph * 3];
	num = 0;


	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec[0]->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec[0]->param->samples.y;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec[0]->param->samples.x) x01 = lbmvec[0]->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec[0]->param->samples.y) y01 = lbmvec[0]->param->samples.y - 1;

			int ind0 = x00 + y00 * lbmvec[0]->param->samples.x;
			int ind1 = x01 + y00 * lbmvec[0]->param->samples.x;
			int ind2 = x00 + y01 * lbmvec[0]->param->samples.x;
			int ind3 = x01 + y01 * lbmvec[0]->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[ind0] + rateX * cutslice_ve[ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[ind2] + rateX * cutslice_ve[ind3]);
			vv = vv / 0.1;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}

	char filename[2048];
	sprintf(filename, "../dataMR2D_bubble_home/ppm_ve_type/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] cutslice_ve;
	delete[] vertices;
}



inline void mrSolver2D::mlVisTagSlice(long upw, long uph, int scaleNum, int frame)
{
	int upnum = 0;
	int baseScale = 0;
	float* cutslice_ve = new float[lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y];
	int num = 0;
	int total_num = lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y;
	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				float ux = 0, uy = 0, rho = 0;
				//for (int j = 0; j < 9; j++)
				//{
				//	ux += lbmvec[0]->fMom[j * total_num + curind] * ex2d_cpu[j];
				//	uy += lbmvec[0]->fMom[j * total_num + curind] * ey2d_cpu[j];

				//	rho += lbmvec[0]->fMom[j * total_num + curind];

				//}
				rho = lbmvec[0]->fMom[curind * 6 + 0];
				ux = lbmvec[0]->fMom[curind * 6 + 1];
				uy = lbmvec[0]->fMom[curind * 6 + 2];
				auto flag = lbmvec[0]->flag[curind];
				auto mass = lbmvec[0]->mass[curind];
				auto phi = lbmvec[0]->phi[curind];
				auto tag = lbmvec[0]->tag_matrix[curind];
				auto label = lbmvec[0]->view_label_matrix[curind];
				// cutslice_ve[num] = (float)(tag+1)/70 * 0.5;
				/*if (num >= (399-97) * 400 + 2 && num <= (399 -97) * 400 + 7)
					std::cout << "tag: " << tag << std::endl;*/
					//cutslice_ve[num] = (float) (1+tag)/90 * 0.5;
				auto mass_surplus = lbmvec[0]->mass_surplus[curind];
				//cutslice_ve[num] = (float)(flag==TYPE_I) / 30 * 0.5;
			//if (flag == TYPE_G)
			//	cutslice_ve[num] = (float)(flag == TYPE_G) / 10 * 0.5;
			//else
			//	cutslice_ve[num] = (float)(flag == TYPE_I) / 10 * 0.2;
			//std::cout << cutslice_ve[num] << std::endl;
				cutslice_ve[num] = mass_surplus;
				num++;
			}
	}
	float* vertices = new float[upw * uph * 3];
	num = 0;


	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec[0]->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec[0]->param->samples.y;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec[0]->param->samples.x) x01 = lbmvec[0]->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec[0]->param->samples.y) y01 = lbmvec[0]->param->samples.y - 1;

			int ind0 = x00 + y00 * lbmvec[0]->param->samples.x;
			int ind1 = x01 + y00 * lbmvec[0]->param->samples.x;
			int ind2 = x00 + y01 * lbmvec[0]->param->samples.x;
			int ind3 = x01 + y01 * lbmvec[0]->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[ind0] + rateX * cutslice_ve[ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[ind2] + rateX * cutslice_ve[ind3]);
			vv = vv / 0.1;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}

	char filename[2048];
	sprintf(filename, "../dataMR2D_bubble_home/ppm_ve_tag/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] cutslice_ve;
	delete[] vertices;
}


inline void mrSolver2D::mlVisDelatMassSlice(long upw, long uph, int scaleNum, int frame)
{
	int upnum = 0;
	int baseScale = 0;
	float* cutslice_ve = new float[lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y];
	int num = 0;
	int total_num = lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y;
	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				float ux = 0, uy = 0, rho = 0;
				//for (int j = 0; j < 9; j++)
				//{
				//	ux += lbmvec[0]->fMom[j * total_num + curind] * ex2d_cpu[j];
				//	uy += lbmvec[0]->fMom[j * total_num + curind] * ey2d_cpu[j];

				//	rho += lbmvec[0]->fMom[j * total_num + curind];

				//}
				rho = lbmvec[0]->fMom[curind * 6 + 0];
				ux = lbmvec[0]->fMom[curind * 6 + 1];
				uy = lbmvec[0]->fMom[curind * 6 + 2];
				auto flag = lbmvec[0]->flag[curind];
				auto mass = lbmvec[0]->mass[curind];
				auto phi = lbmvec[0]->phi[curind];
				auto tag = lbmvec[0]->tag_matrix[curind];
				auto label = lbmvec[0]->view_label_matrix[curind];
				auto delta_mass = lbmvec[0]->delta_g[curind] ;
				// cutslice_ve[num] = (float)(tag+1)/70 * 0.5;
				/*if (num >= (399-97) * 400 + 2 && num <= (399 -97) * 400 + 7)
					std::cout << "tag: " << tag << std::endl;*/
					//cutslice_ve[num] = (float) (1+tag)/90 * 0.5;
				auto mass_surplus = lbmvec[0]->mass_surplus[curind];
				//cutslice_ve[num] = (float)(flag==TYPE_I) / 30 * 0.5;
			//if (flag == TYPE_G)
			//	cutslice_ve[num] = (float)(flag == TYPE_G) / 10 * 0.5;
			//else
			//	cutslice_ve[num] = (float)(flag == TYPE_I) / 10 * 0.2;
			//std::cout << cutslice_ve[num] << std::endl;
				cutslice_ve[num] = fmax(delta_mass,0.f);
				num++;
			}
	}
	float* vertices = new float[upw * uph * 3];
	num = 0;


	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec[0]->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec[0]->param->samples.y;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec[0]->param->samples.x) x01 = lbmvec[0]->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec[0]->param->samples.y) y01 = lbmvec[0]->param->samples.y - 1;

			int ind0 = x00 + y00 * lbmvec[0]->param->samples.x;
			int ind1 = x01 + y00 * lbmvec[0]->param->samples.x;
			int ind2 = x00 + y01 * lbmvec[0]->param->samples.x;
			int ind3 = x01 + y01 * lbmvec[0]->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[ind0] + rateX * cutslice_ve[ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[ind2] + rateX * cutslice_ve[ind3]);
			vv = vv / 0.1;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}

	char filename[2048];
	sprintf(filename, "../dataMR2D_bubble_home/ppm_ve_delta_mass/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] cutslice_ve;
	delete[] vertices;
}



inline void mrSolver2D::mlSavePhi(long upw, long uph, int scaleNum, int frame)
{
	char filename[2048];
	sprintf(filename, "../dataMR3D/ppm_ve_home_test_phi/phi%05d.bin", frame);
	std::filesystem::path filePath(filename);
	std::filesystem::path directory = filePath.parent_path();

	if (!directory.empty() && !std::filesystem::exists(directory)) {
		std::filesystem::create_directories(directory);
	}
	FILE* fp = fopen(filename, "wb");
	int Nx = lbmvec[0]->param->samples.x;
	int Ny = lbmvec[0]->param->samples.y;
	fwrite(&Nx, sizeof(float), 1, fp);
	fwrite(&Ny, sizeof(float), 1, fp);
	fwrite(lbmvec[0]->phi, sizeof(float), lbmvec[0]->count, fp);
	fclose(fp);
}




inline void mrSolver2D::mlVisRhoSlice(long upw, long uph, int scaleNum, int frame)
{
	int upnum = 0;
	int baseScale = 0;
	float* cutslice_ve = new float[lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y];
	int num = 0;
	int total_num = lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y;
	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				float ux = 0, uy = 0, rho = 0;
				//for (int j = 0; j < 9; j++)
				//{
				//	ux += lbmvec[0]->fMom[j * total_num + curind] * ex2d_cpu[j];
				//	uy += lbmvec[0]->fMom[j * total_num + curind] * ey2d_cpu[j];

				//	rho += lbmvec[0]->fMom[j * total_num + curind];

				//}
				rho = lbmvec[0]->fMom[curind * 6 + 0];
				ux = lbmvec[0]->fMom[curind * 6 + 1];
				uy = lbmvec[0]->fMom[curind * 6 + 2];
				auto flag = lbmvec[0]->flag[curind];
				auto mass = lbmvec[0]->mass[curind];
				auto phi = lbmvec[0]->phi[curind];
				auto tag = lbmvec[0]->tag_matrix[curind];
				auto label = lbmvec[0]->view_label_matrix[curind];
				auto delta_mass = lbmvec[0]->delta_g[curind];
				// cutslice_ve[num] = (float)(tag+1)/70 * 0.5;
				/*if (num >= (399-97) * 400 + 2 && num <= (399 -97) * 400 + 7)
					std::cout << "tag: " << tag << std::endl;*/
					//cutslice_ve[num] = (float) (1+tag)/90 * 0.5;
				auto mass_surplus = lbmvec[0]->mass_surplus[curind];
				//cutslice_ve[num] = (float)(flag==TYPE_I) / 30 * 0.5;
			//if (flag == TYPE_G)
			//	cutslice_ve[num] = (float)(flag == TYPE_G) / 10 * 0.5;
			//else
			//	cutslice_ve[num] = (float)(flag == TYPE_I) / 10 * 0.2;
			//std::cout << cutslice_ve[num] << std::endl;

				cutslice_ve[num] = phi;
				num++;
			}
	}




	float* vertices = new float[upw * uph * 3];
	num = 0;


	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec[0]->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec[0]->param->samples.y;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec[0]->param->samples.x) x01 = lbmvec[0]->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec[0]->param->samples.y) y01 = lbmvec[0]->param->samples.y - 1;

			int ind0 = x00 + y00 * lbmvec[0]->param->samples.x;
			int ind1 = x01 + y00 * lbmvec[0]->param->samples.x;
			int ind2 = x00 + y01 * lbmvec[0]->param->samples.x;
			int ind3 = x01 + y01 * lbmvec[0]->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[ind0] + rateX * cutslice_ve[ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[ind2] + rateX * cutslice_ve[ind3]);
			vv = vv / 1.0;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}

	char filename[2048];
	sprintf(filename, "../dataMR2D_bubble_home/ppm_ve_phi/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] cutslice_ve;
	delete[] vertices;
}


inline void mrSolver2D::mlVisCurSlice(long upw, long uph, int scaleNum, int frame)
{
	int upnum = 0;
	int baseScale = 0;
	float* cutslice_ve = new float[lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y];
	int num = 0;
	int total_num = lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y;
	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				float ux = 0, uy = 0, rho = 0;

				rho = lbmvec[0]->fMom[curind * 6 + 0];
				ux = lbmvec[0]->fMom[curind * 6 + 1];
				uy = lbmvec[0]->fMom[curind * 6 + 2];
				auto flag = lbmvec[0]->flag[curind];
				auto mass = lbmvec[0]->mass[curind];
				auto phi = lbmvec[0]->phi[curind];
				auto tag = lbmvec[0]->tag_matrix[curind];
				auto label = lbmvec[0]->view_label_matrix[curind];
				auto delta_mass = lbmvec[0]->delta_g[curind];
				auto curvature = lbmvec[0]->fMomViewer[curind];
				// cutslice_ve[num] = (float)(tag+1)/70 * 0.5;
				/*if (num >= (399-97) * 400 + 2 && num <= (399 -97) * 400 + 7)
					std::cout << "tag: " << tag << std::endl;*/
					//cutslice_ve[num] = (float) (1+tag)/90 * 0.5;
				auto mass_surplus = lbmvec[0]->mass_surplus[curind];
				auto c_value = lbmvec[0]->c_value[curind];
				//cutslice_ve[num] = (float)(flag==TYPE_I) / 30 * 0.5;
			//if (flag == TYPE_G)
			//	cutslice_ve[num] = (float)(flag == TYPE_G) / 10 * 0.5;
			//else
			//	cutslice_ve[num] = (float)(flag == TYPE_I) / 10 * 0.2;
			//std::cout << cutslice_ve[num] << std::endl;
				cutslice_ve[num] = curvature;
				num++;
			}
	}
	float* vertices = new float[upw * uph * 3];
	num = 0;


	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec[0]->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec[0]->param->samples.y;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec[0]->param->samples.x) x01 = lbmvec[0]->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec[0]->param->samples.y) y01 = lbmvec[0]->param->samples.y - 1;

			int ind0 = x00 + y00 * lbmvec[0]->param->samples.x;
			int ind1 = x01 + y00 * lbmvec[0]->param->samples.x;
			int ind2 = x00 + y01 * lbmvec[0]->param->samples.x;
			int ind3 = x01 + y01 * lbmvec[0]->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[ind0] + rateX * cutslice_ve[ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[ind2] + rateX * cutslice_ve[ind3]);
			vv = vv / 0.7f;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}

	char filename[2048];
	sprintf(filename, "../dataMR2D_bubble_home/ppm_ve_cvalue/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] cutslice_ve;
	delete[] vertices;

	float max_curvature = -1.f;

	for (int i = 0; i < scaleNum; i++)
	{
		int edx = 0;
		int edy = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = 0; y < edy; y++)
			for (int x = 0; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				auto curvature = lbmvec[0]->fMomViewer[curind];
				max_curvature = max(curvature, max_curvature);
				num++;
			}
	}
	cout << "max curvature: " << max_curvature  << endl;
}


inline void mrSolver2D::mlVisDisjoinSlice(long upw, long uph, int scaleNum, int frame)
{
	int upnum = 0;
	int baseScale = 0;
	float* cutslice_ve = new float[lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y];
	int num = 0;
	int total_num = lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y;
	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				float ux = 0, uy = 0, rho = 0;

				rho = lbmvec[0]->fMom[curind * 6 + 0];
				ux = lbmvec[0]->fMom[curind * 6 + 1];
				uy = lbmvec[0]->fMom[curind * 6 + 2];
				auto flag = lbmvec[0]->flag[curind];
				auto mass = lbmvec[0]->mass[curind];
				auto phi = lbmvec[0]->phi[curind];
				auto tag = lbmvec[0]->tag_matrix[curind];
				auto label = lbmvec[0]->view_label_matrix[curind];
				auto delta_mass = lbmvec[0]->delta_g[curind];
				auto curvature = lbmvec[0]->fMomViewer[curind + 2*edx*edy];
				// cutslice_ve[num] = (float)(tag+1)/70 * 0.5;
				/*if (num >= (399-97) * 400 + 2 && num <= (399 -97) * 400 + 7)
					std::cout << "tag: " << tag << std::endl;*/
					//cutslice_ve[num] = (float) (1+tag)/90 * 0.5;
				auto mass_surplus = lbmvec[0]->mass_surplus[curind];
				auto c_value = lbmvec[0]->c_value[curind];
				//cutslice_ve[num] = (float)(flag==TYPE_I) / 30 * 0.5;
			//if (flag == TYPE_G)
			//	cutslice_ve[num] = (float)(flag == TYPE_G) / 10 * 0.5;
			//else
			//	cutslice_ve[num] = (float)(flag == TYPE_I) / 10 * 0.2;
			//std::cout << cutslice_ve[num] << std::endl;
				cutslice_ve[num] = curvature;
				num++;
			}
	}
	float* vertices = new float[upw * uph * 3];
	num = 0;


	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec[0]->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec[0]->param->samples.y;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec[0]->param->samples.x) x01 = lbmvec[0]->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec[0]->param->samples.y) y01 = lbmvec[0]->param->samples.y - 1;

			int ind0 = x00 + y00 * lbmvec[0]->param->samples.x;
			int ind1 = x01 + y00 * lbmvec[0]->param->samples.x;
			int ind2 = x00 + y01 * lbmvec[0]->param->samples.x;
			int ind3 = x01 + y01 * lbmvec[0]->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[ind0] + rateX * cutslice_ve[ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[ind2] + rateX * cutslice_ve[ind3]);
			vv = vv / 0.7f;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}

	char filename[2048];
	sprintf(filename, "../dataMR2D_bubble_home/ppm_ve_disjoint/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] cutslice_ve;
	delete[] vertices;

	float max_curvature = -1.f;

	for (int i = 0; i < scaleNum; i++)
	{
		int edx = 0;
		int edy = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = 0; y < edy; y++)
			for (int x = 0; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;
				auto curvature = lbmvec[0]->fMomViewer[curind + 2*edx*edy];
				max_curvature = max(curvature, max_curvature);
				num++;
			}
	}
	cout << "max disjoint: " << max_curvature  << endl;
}

inline void mrSolver2D::mlSavePPM(const char* filename, float* data, int mWidth, int mHeight)
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

#endif // !MRSOLVER2DH_
