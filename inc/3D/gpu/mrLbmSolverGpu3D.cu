
#include "../../../common/mlcudaCommon.h"
#include "mrConstantParamsGpu3D.h"
#include "mrUtilFuncGpu3D.h"
#include "mrLbmSolverGpu3D.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/host_vector.h>
#include "tDCCL.cuh"


__host__ __device__
void MomSwap(REAL*& pt1, REAL*& pt2) {
	REAL* temp = pt1;
	pt1 = pt2;
	pt2 = temp;
}


__host__ __device__
void MomSwap(double*& pt1, double*& pt2) {
	double* temp = pt1;
	pt1 = pt2;
	pt2 = temp;
}

__host__ __device__ inline void swap(int& a, int& b) {
	int temp = a;
	a = b;
	b = temp;
}

__device__ inline void swap(float& a, float& b) {
	int temp = a;
	a = b;
	b = temp;
}

__device__ inline void swap(double& a, double& b) {
	int temp = a;
	a = b;
	b = temp;
}

__device__ inline void swap(float3& a, float3& b) {
	float3 temp = a;
	a = b;
	b = temp;
}


__device__ void report_split(mrFlow3D* mlflow, int x, int y, int z, int sample_x, int sample_y, int sample_z)
{
	int curind = z * sample_y * sample_x + y * sample_x + x;
	// use previous tag to record for the bubble volume change computation
	mlflow[0].previous_tag[curind] = mlflow[0].tag_matrix[curind];
	mlflow[0].tag_matrix[curind] = -1;
	if (mlflow[0].previous_tag[curind]>0)
		atomicExch(&mlflow[0].split_flag, 1);
}


__global__ void clear_detector(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		mlflow[0].merge_detector[curind] = 0;
	}
	if (curind == 1)
	{
		mlflow[0].split_flag = 0;
		mlflow[0].merge_flag = 0;
	}
}

void ClearDectector(mrFlow3D* mlflow, MLFluidParam3D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_z = param->samples.z;
	int sample_num = sample_x * sample_y;
	int total_num = sample_num * sample_z;
	dim3 threads1(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		ceil(REAL(sample_z) / threads1.z)
	);

	clear_detector << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}


__global__ void clear_inlet(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int total_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		if (mlflow[0].islet[curind] == 1)
		{
			mlflow[0].islet[curind] = 0;
			mlflow[0].flag[curind] = TYPE_G;
			mlflow[0].phi[curind] = 0.f;
			mlflow[0].mass[curind] = 0.f; // update mass
			mlflow[0].massex[curind] = 0.f; // update mass
			mlflow[0].fMom[curind + 0 * total_num] = 1.f;
			mlflow[0].fMom[curind + 1 * total_num] = 0.f;
			mlflow[0].fMom[curind + 2 * total_num] = 0.f;
			mlflow[0].fMom[curind + 3 * total_num] = 0.f;
			mlflow[0].fMom[curind + 4 * total_num] = 0.f;
			mlflow[0].fMom[curind + 5 * total_num] = 0.f;
			mlflow[0].fMom[curind + 6 * total_num] = 0.f;
			mlflow[0].fMom[curind + 7 * total_num] = 0.f;
			mlflow[0].fMom[curind + 8 * total_num] = 0.f;
			mlflow[0].fMom[curind + 9 * total_num] = 0.f;	
		}
	}
}

__global__ void ResetDisjoinForce(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;

	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		mlflow[0].massex[curind] = 0.0f;
		mlflow[0].disjoin_force[curind] = 0.f;
	}
}

__global__ void calculate_disjoint(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int total_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int curind = z * sample_x * sample_y + y * sample_x + x;
	mrUtilFuncGpu3D mrutilfunc;

	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_su == TYPE_I)
		{
			float massn = mlflow[0].mass[curind];
			float phij[27]; // cache fill level of neighbor lattice points
			for (int i = 1; i < 27; i++)
			{
				int dx = int(ex3d_gpu[i]);
				int dy = int(ey3d_gpu[i]);
				int dz = int(ez3d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int z1 = z - dz;

				int ind_back = z1 * sample_x * sample_y + y1 * sample_x + x1;
				massn += mlflow[0].massex[ind_back]; // get mass to derive the current phij[0]
			}
			for (int i = 1; i < 27; i++)
			{
				int dx = int(ex3d_gpu[i]);
				int dy = int(ey3d_gpu[i]);
				int dz = int(ez3d_gpu[i]);

				int x1 = x - dx;
				int y1 = y - dy;
				int z1 = z - dz;

				int ind_back = z1 * sample_x * sample_y + y1 * sample_x + x1;
				phij[i] = mlflow[0].phi[ind_back]; 

				if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
				{
					for (int ijk = 0; ijk < 6; ijk++)
					{
						int x2 = x1 - int(ex3d_gpu[ijk + 1]);
						int y2 = y1 - int(ey3d_gpu[ijk + 1]);
						int z2 = z1 - int(ez3d_gpu[ijk + 1]);
						if (x2 >= 0 && x2 < sample_x && y2 >= 0 && y2 < sample_y && z2 >= 0 && z2 < sample_z)
						{
							int ind_k = z2 * sample_y * sample_x + y2 * sample_x + x2;
							if ((mlflow[0].flag[ind_k] & TYPE_BO) != TYPE_S)
							{
								phij[i] = mlflow[0].phi[ind_k];
								break;
							}
						}
					}
				}
			}
			float rhon = 0.0f;
			rhon = mlflow[0].fMom[total_num * 0 + curind];
			phij[0] = mrutilfunc.calculate_phi(rhon, massn, flagsn); 
			int tag_curind = mlflow[0].tag_matrix[curind] - 1;
			float3 normal = mrutilfunc.calculate_normal(phij);
			float disjoint = 0.f;
			int max_ids = -1;
		
			for (int jk = 1; jk < 20; jk++)
			{
				// start a ray along the normal direction for max(jk) * 0.2 = 4.0 length and check the interface
				int x12 = round((float)x - (float)jk * 0.2f * normal.x);
				int y12 = round((float)y - (float)jk * 0.2f * normal.y);
				int z12 = round((float)z - (float)jk * 0.2f * normal.z);

				if (x12 >= 0 && x12 < sample_x && y12 >= 0 && y12 < sample_y && z12 >= 0 && z12 < sample_z)
				{
					int ind_back = z12 * sample_x * sample_y + y12 * sample_x + x12;
					if (mlflow[0].tag_matrix[ind_back] > 0)
					{
						int tag_neighbor = mlflow[0].tag_matrix[ind_back] - 1;
						// refine the distance (consider the center offset for the current node and the opacity for the neighbor node)
						if (tag_curind != tag_neighbor && mlflow[0].flag[ind_back] == TYPE_I)
						{
							float center_offset = mrutilfunc.plic_cube(phij[0], normal);
							float alpha = mlflow[0].phi[ind_back];

							float dis = abs((float)jk * 0.2f * normal.x) - (1 - alpha);
							float d = abs(dis / (normal.x + 1e-8)) - center_offset;

							if (disjoint < 1.f - d / 4.f)
							{
								disjoint = 1.f - d / 4.f;
								max_ids = tag_neighbor;
							}
						}
					}
				}
			}
			if (disjoint > 0)
			{
				atomicAdd(&mlflow[0].disjoin_force[curind], disjoint);
			}
		}
	}
}

__global__ void Init3D(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num, int total_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_num + y * sample_x + x;
	mrUtilFuncGpu3D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{

		unsigned char flagsn = mlflow[0].flag[curind];
		const unsigned char flagsn_bo = flagsn & TYPE_BO; // extract boundary flags

		unsigned char flagsj[27]{};
		for (int i = 0; i < 27; i++)
		{
			int dx = int(ex3d_gpu[i]);
			int dy = int(ey3d_gpu[i]);
			int dz = int(ez3d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int z1 = z - dz;

			if (x1 >= 0 && x1 < sample_x && y1 >= 0 && y1 < sample_y && z1 >= 0 && z1 < sample_z)
			{
				int ind_back = z1 * sample_num + y1 * sample_x + x1;
				flagsj[i] = mlflow[0].flag[ind_back];
			}
		}

		if (flagsn_bo == TYPE_S) { // cell is solid
			bool TYPE_ONLY_S = true; // has only solid neighbors
			for (int i = 1; i < 27; i++) TYPE_ONLY_S = TYPE_ONLY_S && (flagsj[i] & TYPE_BO) == TYPE_S;
			if (TYPE_ONLY_S) {
				mlflow[0].fMomPost[curind + 1 * total_num] = mlflow[0].fMom[curind + 1 * total_num] = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
				mlflow[0].fMomPost[curind + 2 * total_num] = mlflow[0].fMom[curind + 2 * total_num] = 0.0f;
				mlflow[0].fMomPost[curind + 3 * total_num] = mlflow[0].fMom[curind + 3 * total_num] = 0.0f;
			}
		}
		if (flagsn_bo == TYPE_S) {
			mlflow[0].fMomPost[curind + 1 * total_num] = mlflow[0].fMom[curind + 1 * total_num] = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
			mlflow[0].fMomPost[curind + 2 * total_num] = mlflow[0].fMom[curind + 2 * total_num] = 0.0f;
			mlflow[0].fMomPost[curind + 3 * total_num] = mlflow[0].fMom[curind + 3 * total_num] = 0.0f;
		}

		// calculate the equilibrium distribution function
		float feq[27]{}; 
		mrutilfunc.calculate_f_eq(mlflow[0].fMom[curind + 0 * total_num], 
			mlflow[0].fMom[curind + 1 * total_num], mlflow[0].fMom[curind + 2 * total_num], 
			mlflow[0].fMom[curind + 3 * total_num], feq);

		float geq[7]{};
		mrutilfunc.calculate_g_eq(mlflow[0].c_value[curind], 
			mlflow[0].fMom[curind + 1 * total_num], mlflow[0].fMom[curind + 2 * total_num], 
			mlflow[0].fMom[curind + 3 * total_num], geq);


		// reset the flag for the interface between fluid and gas
		float phin = mlflow[0].phi[curind];
		if (!(flagsn & (TYPE_S | TYPE_E | TYPE_T | TYPE_F | TYPE_I)))
			flagsn = (flagsn & ~TYPE_SU) | TYPE_G; // change all non-fluid and non-interface flags to gas
		if ((flagsn & TYPE_SU) == TYPE_G)
		{ // cell with updated flags is gas
			bool change = false; // check if cell has to be changed to interface
			for (int i = 1; i < 27; i++)
				change = change || (flagsj[i] & TYPE_SU) == TYPE_F; // if neighbor flag fluid is set, the cell must be interface
			if (change)
			{ // create interface automatically if phi has not explicitely defined for the interface layer
				flagsn = (flagsn & ~TYPE_SU) | TYPE_I; // cell must be interface
				phin = 0.5f;
				float rhon, uxn, uyn, uzn; // initialize interface cells with average density/velocity of fluid neighbors
				 // average over all fluid/interface neighbors
				float rhot = 0.0f, uxt = 0.0f, uyt = 0.0f, uzt = 0.0f, counter = 0.0f; // average over all fluid/interface neighbors
				float rhon_g = 0.0f ,rhogt = 0.0f, c_k = 0.0f; // average over all fluid/interface neighbors
				for (int i = 1; i < 27; i++)
				{
					int dx = int(ex3d_gpu[i]);
					int dy = int(ey3d_gpu[i]);
					int dz = int(ez3d_gpu[i]);
					int x1 = x - dx;
					int y1 = y - dy;
					int z1 = z - dz;
					int ind_back = z1 * sample_num + y1 * sample_x + x1;
					const unsigned char flagsji_su = mlflow[0].flag[ind_back] & TYPE_SU;
					if (flagsji_su == TYPE_F) { // fluid or interface or (interface->fluid) neighbor
						counter += 1.0f;
						rhot += mlflow[0].fMom[ind_back + 0 * total_num];
						uxt += mlflow[0].fMom[ind_back + 1 * total_num];
						uyt += mlflow[0].fMom[ind_back + 2 * total_num];
						uzt += mlflow[0].fMom[ind_back + 3 * total_num];
						if (i < 7)
						{
							c_k += 1.0f;
							rhogt += mlflow[0].c_value[ind_back];
						}
					}
				}
				rhon = counter > 0.0f ? rhot / counter : 1.0f;
				uxn = counter > 0.0f ? uxt / counter : 0.0f;
				uyn = counter > 0.0f ? uyt / counter : 0.0f;
				uzn = counter > 0.0f ? uzt / counter : 0.0f;
				rhon_g = c_k > 0.0f ? rhogt / c_k : 0.0f;

				mrutilfunc.calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs
				mrutilfunc.calculate_g_eq(rhon_g, uxn, uyn, uzn, geq);
				mlflow[0].c_value[curind] = rhon_g;
			}
		}
		if ((flagsn & TYPE_SU) == TYPE_G) { // cell with updated flags is still gas
			mlflow[0].fMom[curind + 1 * total_num] = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
			mlflow[0].fMom[curind + 2 * total_num] = 0.0f;
			mlflow[0].fMom[curind + 3 * total_num] = 0.0f;
			phin = 0.0f;
		}
		else if ((flagsn & TYPE_SU) == TYPE_I && (phin < 0.0f || phin>1.0f)) {
			phin = 0.5f; // cell should be interface, but phi was invalid
		}
		else if ((flagsn & TYPE_SU) == TYPE_F) {
			phin = 1.0f;
		}
		mlflow[0].phi[curind] = phin;
		mlflow[0].mass[curind] = phin * mlflow[0].fMom[curind + 0 * total_num];
		mlflow[0].massex[curind] = 0.0f; // reset excess mass
		mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)flagsn;

		// deal with the high order moment
		for (int i = 0; i < 27; i++)
		{
			feq[i] += w3d_gpu[i];
		}
		float invRho = 1.0 / mlflow[0].fMom[curind + 0 * total_num];
		float pixx = ((feq[1] + feq[2] + feq[7] + feq[8] + feq[9] + feq[10] + feq[13] + feq[14] + feq[15] + feq[16] + feq[19] + feq[20] + feq[21] + feq[22] + feq[23] + feq[24] + feq[25] + feq[26]));
		float pixy = (((feq[7] + feq[8] + feq[19] + feq[20] + feq[21] + feq[22]) - (feq[13] + feq[14] + feq[23] + feq[24] + feq[25] + feq[26])));
		float pixz = (((feq[9] + feq[10] + feq[19] + feq[20] + feq[23] + feq[24]) - (feq[15] + feq[16] + feq[21] + feq[22] + feq[25] + feq[26])));
		float piyy = ((feq[3] + feq[4] + feq[7] + feq[8] + feq[11] + feq[12] + feq[13] + feq[14] + feq[17] + feq[18] + feq[19] + feq[20] + feq[21] + feq[22] + feq[23] + feq[24] + feq[25] + feq[26]));
		float piyz = (((feq[11] + feq[12] + feq[19] + feq[20] + feq[25] + feq[26]) - (feq[17] + feq[18] + feq[21] + feq[22] + feq[23] + feq[24])));
		float pizz = ((feq[5] + feq[6] + feq[9] + feq[10] + feq[11] + feq[12] + feq[15] + feq[16] + feq[17] + feq[18] + feq[19] + feq[20] + feq[21] + feq[22] + feq[23] + feq[24] + feq[25] + feq[26]));

		pixx = pixx * invRho - cs2;
		pixy = pixy * invRho;
		pixz = pixz * invRho;
		piyy = piyy * invRho - cs2;
		piyz = piyz * invRho;
		pizz = pizz * invRho - cs2;
		
		mlflow[0].fMomPost[curind + 0 * total_num] = mlflow[0].fMom[curind + 0 * total_num];
		mlflow[0].fMomPost[curind + 1 * total_num] = mlflow[0].fMom[curind + 1 * total_num];
		mlflow[0].fMomPost[curind + 2 * total_num] = mlflow[0].fMom[curind + 2 * total_num];
		mlflow[0].fMomPost[curind + 3 * total_num] = mlflow[0].fMom[curind + 3 * total_num];
		mlflow[0].fMomPost[curind + 4 * total_num] = mlflow[0].fMom[curind + 4 * total_num] = pixx;
		mlflow[0].fMomPost[curind + 5 * total_num] = mlflow[0].fMom[curind + 5 * total_num] = pixy;
		mlflow[0].fMomPost[curind + 6 * total_num] = mlflow[0].fMom[curind + 6 * total_num] = pixz;
		mlflow[0].fMomPost[curind + 7 * total_num] = mlflow[0].fMom[curind + 7 * total_num] = piyy;
		mlflow[0].fMomPost[curind + 8 * total_num] = mlflow[0].fMom[curind + 8 * total_num] = piyz;
		mlflow[0].fMomPost[curind + 9 * total_num] = mlflow[0].fMom[curind + 9 * total_num] = pizz;

		for (int i = 0; i < 7; i++)
		{
			mlflow[0].gMom[curind + i * total_num] = geq[i];
			mlflow[0].gMomPost[curind + i * total_num] = geq[i];
		}
	}

}

__global__ void surface_1(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num, int total_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_num + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S); // extract SURFACE flags
		if (flagsn_sus == TYPE_IF)
		{
			for (int i = 1; i < 27; i++)
			{
				int dx = int(ex3d_gpu[i]);
				int dy = int(ey3d_gpu[i]);
				int dz = int(ez3d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int z1 = z - dz;
				int ind_back = z1 * sample_num + y1 * sample_x + x1;
				const unsigned char flagsji = mlflow[0].flag[ind_back];
				const unsigned char flagsji_su = flagsji & (TYPE_SU | TYPE_S); // extract SURFACE flags
				const unsigned char flagsji_r = flagsji & ~TYPE_SU; // extract all non-SURFACE flags
				if (flagsji_su == TYPE_IG) mlflow[0].flag[ind_back] = (MLLATTICENODE_SURFACE_FLAG)(flagsji_r | TYPE_I); // prevent interface neighbor cells from becoming gas
				else if (flagsji_su == TYPE_G) mlflow[0].flag[ind_back] = (MLLATTICENODE_SURFACE_FLAG)(flagsji_r | TYPE_GI); // neighbor cell was gas and must change to interface
			}
		}
	}
}

__global__ void surface_2(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num, int total_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	mrUtilFuncGpu3D mrutilfunc;
	int curind = z * sample_num + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S); // extract SURFACE flags
		if (flagsn_sus == TYPE_GI) { // initialize the fi of gas cells that should become interface
			float rhon, uxn, uyn, uzn; // average over all fluid/interface neighbors
			float rhot = 0.0f, uxt = 0.0f, uyt = 0.0f, uzt = 0.0f, counter = 0.0f; // average over all fluid/interface neighbors
			float rho_gt = 0.f, c_k = 0.f;
			
			for (int i = 1; i < 27; i++)
			{
				int dx = int(ex3d_gpu[i]);
				int dy = int(ey3d_gpu[i]);
				int dz = int(ez3d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int z1 = z - dz;
				int ind_back = z1 * sample_num + y1 * sample_x + x1;
				const unsigned char flagsji_sus = mlflow[0].flag[ind_back] & (TYPE_SU | TYPE_S); // extract SURFACE flags
				if (flagsji_sus == TYPE_F || flagsji_sus == TYPE_I || flagsji_sus == TYPE_IF) { // fluid or interface or (interface->fluid) neighbor
					counter += 1.0f;
					rhot += mlflow[0].fMomPost[ind_back + 0 * total_num];
					uxt += mlflow[0].fMomPost[ind_back + 1 * total_num];
					uyt += mlflow[0].fMomPost[ind_back + 2 * total_num];
					uzt += mlflow[0].fMomPost[ind_back + 3 * total_num];
					if (i < 7)
					{
						rho_gt += mlflow[0].c_value[ind_back];
						c_k += 1.0f;
					}
				}
			}
			rhon = counter > 0.0f ? rhot / counter : 1.0f;
			uxn = counter > 0.0f ? uxt / counter : 0.0f;
			uyn = counter > 0.0f ? uyt / counter : 0.0f;
			uzn = counter > 0.0f ? uzt / counter : 0.0f;

			rho_gt = c_k > 0.0f ? rho_gt / c_k : 0.0f;
			
			float feq[27];
			mrutilfunc.calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs
			for (int i = 0; i < 27; i++)
			{
				feq[i] += w3d_gpu[i];
			}
			float invRho = 1.0 / rhon;
			float pixx = ((feq[1] + feq[2] + feq[7] + feq[8] + feq[9] + feq[10] + feq[13] + feq[14] + feq[15] + feq[16] + feq[19] + feq[20] + feq[21] + feq[22] + feq[23] + feq[24] + feq[25] + feq[26]));
			float pixy = (((feq[7] + feq[8] + feq[19] + feq[20] + feq[21] + feq[22]) - (feq[13] + feq[14] + feq[23] + feq[24] + feq[25] + feq[26])));
			float pixz = (((feq[9] + feq[10] + feq[19] + feq[20] + feq[23] + feq[24]) - (feq[15] + feq[16] + feq[21] + feq[22] + feq[25] + feq[26])));
			float piyy = ((feq[3] + feq[4] + feq[7] + feq[8] + feq[11] + feq[12] + feq[13] + feq[14] + feq[17] + feq[18] + feq[19] + feq[20] + feq[21] + feq[22] + feq[23] + feq[24] + feq[25] + feq[26]));
			float piyz = (((feq[11] + feq[12] + feq[19] + feq[20] + feq[25] + feq[26]) - (feq[17] + feq[18] + feq[21] + feq[22] + feq[23] + feq[24])));
			float pizz = ((feq[5] + feq[6] + feq[9] + feq[10] + feq[11] + feq[12] + feq[15] + feq[16] + feq[17] + feq[18] + feq[19] + feq[20] + feq[21] + feq[22] + feq[23] + feq[24] + feq[25] + feq[26]));

			pixx = pixx * invRho - cs2;
			pixy = pixy * invRho;
			pixz = pixz * invRho;
			piyy = piyy * invRho - cs2;
			piyz = piyz * invRho;
			pizz = pizz * invRho - cs2;

			mlflow[0].fMomPost[curind + 0 * total_num] = rhon;
			mlflow[0].fMomPost[curind + 1 * total_num] = uxn;
			mlflow[0].fMomPost[curind + 2 * total_num] = uyn;
			mlflow[0].fMomPost[curind + 3 * total_num] = uzn;
			mlflow[0].fMomPost[curind + 4 * total_num] = pixx;
			mlflow[0].fMomPost[curind + 5 * total_num] = pixy;
			mlflow[0].fMomPost[curind + 6 * total_num] = pixz;
			mlflow[0].fMomPost[curind + 7 * total_num] = piyy;
			mlflow[0].fMomPost[curind + 8 * total_num] = piyz;
			mlflow[0].fMomPost[curind + 9 * total_num] = pizz;

			//recontruction for g
			mlflow[0].c_value[curind] = rho_gt;
			float geq[7];
			mrutilfunc.calculate_g_eq(rho_gt, uxn, uyn, uzn, geq); 
			for (int i = 0; i < 7; i++)
				mlflow[0].gMom[curind + i * total_num] = geq[i];

		}
		else if (flagsn_sus == TYPE_IG) { // flag interface->gas is set
			for (int i = 1; i < 27; i++)
			{
				int dx = int(ex3d_gpu[i]);
				int dy = int(ey3d_gpu[i]);
				int dz = int(ez3d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int z1 = z - dz;
				int ind_back = z1 * sample_num + y1 * sample_x + x1;

				const unsigned char flagsji = mlflow[0].flag[ind_back];
				const unsigned char flagsji_su = flagsji & (TYPE_SU | TYPE_S); // extract SURFACE flags
				const unsigned char flagsji_r = flagsji & (~TYPE_SU); // extract all non-SURFACE flags

				if (flagsji_su == TYPE_F || flagsji_su == TYPE_IF) {
					if (mlflow[0].islet[ind_back] == 0)
					{
						mlflow[0].flag[ind_back] = (MLLATTICENODE_SURFACE_FLAG)(flagsji_r | TYPE_I); // prevent fluid or interface neighbors that turn to fluid from being/becoming fluid
						
						// if the node is changed to interface, we need to detect the merge.
						mlflow[0].merge_detector[ind_back] = 1;
					}
					else
					{
						mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)(flagsji_r | TYPE_I);
					}
				}

			}

		}
	}
}

__global__ void surface_3(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num, int total_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	mrUtilFuncGpu3D mrutilfunc;
	int curind = z * sample_num + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S); // extract SURFACE flags
		if (flagsn_sus & TYPE_S) return;
		if (mlflow[0].islet[curind] == 1) 
		{	
			// use previous tag to record for the bubble volume change computation
			mlflow[0].previous_tag[curind] = mlflow[0].tag_matrix[curind];
			mlflow[0].tag_matrix[curind] = -1;
			return;
		}
		const float rhon = mlflow[0].fMomPost[curind + 0 * total_num]; // density of cell n
		float massn = mlflow[0].mass[curind]; // mass of cell n
		float massexn = 0.0f; // excess mass of cell n
		float phin = 0.0f;
		if (flagsn_sus == TYPE_F) { // regular fluid cell
			massexn = massn - rhon; // dump mass-rho difference into excess mass
			massn = rhon; // fluid cell mass has to equal rho
			phin = 1.0f;
			// use previous tag to record for the bubble volume change computation
			mlflow[0].previous_tag[curind] = mlflow[0].tag_matrix[curind];
			mlflow[0].tag_matrix[curind] = -1;
		}
		else if (flagsn_sus == TYPE_I) { // regular interface cell
			massexn = massn > rhon ? massn - rhon : massn < 0.0f ? massn : 0.0f; // allow interface cells with mass>rho or mass<0
			massn = clamp(massn, 0.0f, rhon);
			phin = mrutilfunc.calculate_phi(rhon, massn, TYPE_I); // calculate fill level for next step (only necessary for interface cells)
		}
		else if (flagsn_sus == TYPE_G) { // regular gas cell
			massexn = massn; // dump remaining mass into excess mass
			massn = 0.0f;
			phin = 0.0f;
		}
		else if (flagsn_sus == TYPE_IF) { // flag interface->fluid is set
			mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((mlflow[0].flag[curind] & ~TYPE_SU) | TYPE_F); // cell becomes fluid
			// if the node is changed to fluid, we need to report the split
			report_split(mlflow, x, y, z, sample_x, sample_y, sample_z); // report interface->fluid conversion
			massexn = massn - rhon; // dump mass-rho difference into excess mass		
			massn = rhon; // fluid cell mass has to equal rho
			phin = 1.0f; // set phi[n] to 1.0f for fluid cells
		}
		else if (flagsn_sus == TYPE_IG) { // flag interface->gas is set
			mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((mlflow[0].flag[curind] & ~TYPE_SU) | TYPE_G); // cell becomes gas
			massexn = massn; // dump remaining mass into excess mass
			massn = 0.0f; // gas mass has to be zero
			phin = 0.0f; // set phi[n] to 0.0f for gas cells
		}
		else if (flagsn_sus == TYPE_GI) { // flag gas->interface is set
			mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((mlflow[0].flag[curind] & ~TYPE_SU) | TYPE_I); // cell becomes interface
			massexn = massn > rhon ? massn - rhon : massn < 0.0f ? massn : 0.0f; // allow interface cells with mass>rho or mass<0
			massn = clamp(massn, 0.0f, rhon);
			phin = mrutilfunc.calculate_phi(rhon, massn, TYPE_I); // calculate fill level for next step (only necessary for interface cells)
		}
		int counter = 0; // count (fluid|interface) neighbors
		for (int i = 1; i < 27; i++)
		{
			int dx = int(ex3d_gpu[i]);
			int dy = int(ey3d_gpu[i]);
			int dz = int(ez3d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int z1 = z - dz;

			int ind_back = z1 * sample_num + y1 * sample_x + x1;
			const unsigned char flagsji_su = mlflow[0].flag[ind_back] & (TYPE_SU | TYPE_S); // extract SURFACE flags
			counter += (int)(flagsji_su == TYPE_F || flagsji_su == TYPE_I || flagsji_su == TYPE_IF || flagsji_su == TYPE_GI); // avoid branching

		}
		massn += counter > 0 ? 0.0f : massexn; // if excess mass can't be distributed to neighboring interface or fluid cells, add it to local mass (ensure mass conservation)
		massexn = counter > 0 ? massexn / (float)counter : 0.0f; // divide excess mass up for all interface or fluid neighbors
		mlflow[0].mass[curind] = massn; // update mass
		mlflow[0].massex[curind] = massexn; // update excess mass
		mlflow[0].delta_phi[curind] = phin - mlflow[0].phi[curind];

		if ((mlflow[0].flag[curind] & (TYPE_SU | TYPE_S)) == TYPE_I)
		{
			float rhon_g = 0.f;
			for (int i = 0; i < 7; i++)
				rhon_g += mlflow[0].gMom[curind + i * total_num];
			mlflow[0].delta_g[curind] -= rhon_g * (mlflow[0].delta_phi[curind]);
		}

		mlflow[0].phi[curind] = phin; // update phi

	}
}

__global__ void stream_collide_bvh(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num, int total_num, float N, float l0p, float roup, float labma,
	float u0p, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_num + y * sample_x + x;
	mrUtilFuncGpu3D mrutilfunc;
	float Omega = 1 / ((1e-4) * 3.0f + 0.5f);
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_bo == TYPE_S || flagsn_su == TYPE_G) return;

		if (mlflow[0].islet[curind] == 1) 
			{
				mlflow[0].flag[curind] = TYPE_F;
				return;
			}

		float fhn[27]{};
		float fon[27]{};

		for (int i = 0; i < 27; i++)
		{

			int dx = int(ex3d_gpu[i]);
			int dy = int(ey3d_gpu[i]);
			int dz = int(ez3d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int z1 = z - dz;
			int ind_back = z1 * sample_x * sample_y + y1 * sample_x + x1;

			if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
			{
				float feq[27]{}; // f_equilibrium
				mrutilfunc.calculate_f_eq(mlflow[0].fMom[curind + total_num * 0], 0.f, 0.f, 0.f, feq);
				fhn[i] = feq[i];
			}
			else
			{
				float rhoVar = mlflow[0].fMom[ind_back + total_num * 0];
				float ux_t30 = mlflow[0].fMom[ind_back + total_num * 1];
				float uy_t30 = mlflow[0].fMom[ind_back + total_num * 2];
				float uz_t30 = mlflow[0].fMom[ind_back + total_num * 3];
				float pixx_t45 = mlflow[0].fMom[ind_back + total_num * 4];
				float pixy_t90 = mlflow[0].fMom[ind_back + total_num * 5];
				float pixz_t90 = mlflow[0].fMom[ind_back + total_num * 6];
				float piyy_t45 = mlflow[0].fMom[ind_back + total_num * 7];
				float piyz_t90 = mlflow[0].fMom[ind_back + total_num * 8];
				float pizz_t45 = mlflow[0].fMom[ind_back + total_num * 9];


				mrutilfunc.mlCalDistributionFourthOrderD3Q27AtIndex(
					rhoVar,
					ux_t30,
					uy_t30,
					uz_t30,
					pixx_t45,
					pixy_t90,
					pixz_t90,
					piyy_t45,
					piyz_t90,
					pizz_t45,
					i, fhn[i]
				);
				fhn[i] -= w3d_gpu[i];
			}

			float rhoVar = mlflow[0].fMom[curind + total_num * 0];
			float ux_t30 = mlflow[0].fMom[curind + total_num * 1];
			float uy_t30 = mlflow[0].fMom[curind + total_num * 2];
			float uz_t30 = mlflow[0].fMom[curind + total_num * 3];
			float pixx_t45 = mlflow[0].fMom[curind + total_num * 4];
			float pixy_t90 = mlflow[0].fMom[curind + total_num * 5];
			float pixz_t90 = mlflow[0].fMom[curind + total_num * 6];
			float piyy_t45 = mlflow[0].fMom[curind + total_num * 7];
			float piyz_t90 = mlflow[0].fMom[curind + total_num * 8];
			float pizz_t45 = mlflow[0].fMom[curind + total_num * 9];

			mrutilfunc.mlCalDistributionFourthOrderD3Q27AtIndex(
				rhoVar,
				ux_t30,
				uy_t30,
				uz_t30,
				pixx_t45,
				pixy_t90,
				pixz_t90,
				piyy_t45,
				piyz_t90,
				pizz_t45,
				i, fon[i]
			);
			fon[i] -= w3d_gpu[i];
		}

		float massn = mlflow[0].mass[curind];

		for (int i = 1; i < 27; i++)
		{
			int dx = int(ex3d_gpu[i]);
			int dy = int(ey3d_gpu[i]);
			int dz = int(ez3d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int z1 = z - dz;

			int ind_back = z1 * sample_num + y1 * sample_x + x1;
			massn += mlflow[0].massex[ind_back]; // distribute excess mass from last step which is stored in neighbors
		}

		if (flagsn_su == TYPE_F) {
			for (int i = 1; i < 27; i++)
			{
				massn += fhn[i] - fon[i]; // neighbor is fluid or interface cell
			}
		}
		else if (flagsn_su == TYPE_I)
		{ // cell is interface
			float phij[27]; // cache fill level of neighbor lattice points
			for (int i = 1; i < 27; i++)
			{
				int dx = int(ex3d_gpu[i]);
				int dy = int(ey3d_gpu[i]);
				int dz = int(ez3d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int z1 = z - dz;

				int ind_back = z1 * sample_num + y1 * sample_x + x1;
				phij[i] = mlflow[0].phi[ind_back]; // cache fill level of neighbor lattice points

				// deal with the boundary phi without any assumption about the wetting property
				if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
				{
					for (int ijk = 0; ijk < 6; ijk++)
					{
						int x2 = x1 - int(ex3d_gpu[ijk + 1]);
						int y2 = y1 - int(ey3d_gpu[ijk + 1]);
						int z2 = z1 - int(ez3d_gpu[ijk + 1]);
						if (x2 >= 0 && x2 < sample_x && y2 >= 0 && y2 < sample_y && z2 >= 0 && z2 < sample_z)
						{
							int ind_k = z2 * sample_num + y2 * sample_x + x2;
							if ((mlflow[0].flag[ind_k] & TYPE_BO) != TYPE_S)
							{
								phij[i] = mlflow[0].phi[ind_k];
								break;
							}
						}
					}
				}
			}
			float rhon = 0.0f, uxn = 0.0f, uyn = 0.0f, uzn = 0.0f, rho_laplace = 0.0f; // no surface tension if rho_laplace is not overwritten later
			rhon = mlflow[0].fMom[curind + total_num * 0];
			uxn = mlflow[0].fMom[curind + total_num * 1];
			uyn = mlflow[0].fMom[curind + total_num * 2];
			uzn = mlflow[0].fMom[curind + total_num * 3];
			phij[0] = mrutilfunc.calculate_phi(rhon, massn, flagsn); // don't load phi[n] from memory, instead recalculate it with mass corrected by excess mass
			float curv = mrutilfunc.calculate_curvature(phij);
			
			float disjoint = mlflow[0].disjoin_force[curind];
			float disjoint_factor = 0.032;
			float rho_k = 1.f;
			// for bubble pressure
			if (mlflow[0].tag_matrix[curind] > 0)
			{
				rho_k = mlflow[0].bubble.rho[mlflow[0].tag_matrix[curind] - 1];
			}
			// for air layer surface tension  
			float def_6_sigma_k = def_6_sigma;
			if (mlflow[0].bubble.init_volume[mlflow[0].tag_matrix[curind] - 1] > 5000000.0)
			{
				def_6_sigma_k = 1e-6f;
			}
			// for small bubble surface tension
			if ((!(disjoint > 0)) && (def_6_sigma_k>1e-3) && (mlflow[0].bubble.volume[mlflow[0].tag_matrix[curind] - 1] < 64.0))
			{
				def_6_sigma_k = 2e-4;
			}
			// for bubble disappearance visual effect
			if (time > 320 * 180-5)
			{
				def_6_sigma_k = 0e-5;
				disjoint_factor = 0.0;
			}

			rho_laplace = def_6_sigma_k == 0.0f ? 0.0f : def_6_sigma_k * curv;
			
			float feg[27]; // reconstruct f from neighbor gas lattice points

			const float rho2tmp = 0.5f / rhon; // apply external volume force (Guo forcing, Krueger p.233f)
			float uxntmp = fma(mlflow[0].forcex[curind] * rhon, rho2tmp, uxn);// limit velocity (for stability purposes)
			float uyntmp = fma(mlflow[0].forcey[curind] * rhon, rho2tmp, uyn);// force term: F*dt/(2*rho)
			float uzntmp = fma(mlflow[0].forcez[curind] * rhon, rho2tmp, uzn);
			float3 u_2{ uxntmp,uyntmp,uzntmp };
			u_2 = normalizing_clamp(u_2, 0.4);
			uxntmp = u_2.x;
			uyntmp = u_2.y;
			uzntmp = u_2.z;

			mrutilfunc.calculate_f_eq(rho_k - rho_laplace - disjoint_factor * disjoint, uxntmp, uyntmp, uzntmp, feg); // calculate gas equilibrium DDFs with constant ambient pressure

			unsigned char flagsj_su[27]; // cache neighbor flags for multiple readings
			unsigned char flagsj_bo[27];
			for (int i = 1; i < 27; i++)
			{
				int dx = int(ex3d_gpu[i]);
				int dy = int(ey3d_gpu[i]);
				int dz = int(ez3d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int z1 = z - dz;
				int ind_back = z1 * sample_num + y1 * sample_x + x1;
				flagsj_su[i] = mlflow[0].flag[ind_back] & TYPE_SU;
				flagsj_bo[i] = mlflow[0].flag[ind_back] & TYPE_BO;
			}
			for (int i = 1; i < 27; i++)
			{
				massn += flagsj_su[i] & (TYPE_F | TYPE_I) ? flagsj_su[i] == TYPE_F ? fhn[i] - fon[index3dInv_gpu[i]] : 0.5f * (phij[i] + phij[0]) * (fhn[i] - fon[index3dInv_gpu[i]]) : 0.0f; // neighbor is fluid or interface cell
			}
			for (int i = 1; i < 27; i++)
			{
				if (flagsj_su[i] == TYPE_G)
					fhn[i] = feg[index3dInv_gpu[i]] - fon[index3dInv_gpu[i]] + feg[i];
			}

		}
		mlflow[0].mass[curind] = massn;
		float pop[27]{};

		for (int i = 0; i < 27; i++)
		{
			pop[i] = fhn[i] + w3d_gpu[i];
		}

		float FX = mlflow[0].forcex[curind];
		float FY = mlflow[0].forcey[curind];
		float FZ = mlflow[0].forcez[curind];
		float rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
		
		//calculate streaming moments
		FX = FX * rhoVar;
		FY = FY * rhoVar;
		FZ = FZ * rhoVar;

		float invRho = 1 / rhoVar;
		float ux_t30 = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) - (pop[2] + pop[8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5f * FX) * invRho;
		float uy_t30 = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[4] + pop[8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5f * FY) * invRho;
		float uz_t30 = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5f * FZ) * invRho;

		float3 u_{ ux_t30,uy_t30,uz_t30 };
		u_ = normalizing_clamp(u_, 0.4);
		ux_t30 = u_.x;
		uy_t30 = u_.y;
		uz_t30 = u_.z;

		float pixx_t45 = ((pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]));
		float pixy_t90 = (((pop[7] + pop[8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])));
		float pixz_t90 = (((pop[9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])));
		float piyy_t45 = ((pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]));
		float piyz_t90 = (((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24])));
		float pizz_t45 = ((pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]));


		if (flagsn_su == TYPE_I)
		{
			bool TYPE_NO_F = true, TYPE_NO_G = true; // temporary flags for no fluid or gas neighbors
			for (int i = 1; i < 27; i++)
			{
				int dx = int(ex3d_gpu[i]);
				int dy = int(ey3d_gpu[i]);
				int dz = int(ez3d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int z1 = z - dz;
				int ind_back = z1 * sample_num + y1 * sample_x + x1;
				const unsigned char flagsji_su = mlflow[0].flag[ind_back] & TYPE_SU; // extract SURFACE flags
				TYPE_NO_F = TYPE_NO_F && flagsji_su != TYPE_F;
				TYPE_NO_G = TYPE_NO_G && flagsji_su != TYPE_G;
			}
			float massn = mlflow[0].mass[curind];

			if (massn > rhoVar || TYPE_NO_G)
				mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((flagsn & ~TYPE_SU) | TYPE_IF); // set flag interface->fluid
			else if (massn < 0.0f || TYPE_NO_F)
			{
				mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((flagsn & ~TYPE_SU) | TYPE_IG); // set flag interface->gas
			}
		}


		for (int ij = -3;ij<3;ij++)
			for(int jk=-3;jk<3;jk++)
				for(int kh=-3;kh<3;kh++)
			{
				int x12 = x + jk;
				int y12 = y + ij;
				int z12 = z + kh;
				if (x12 >= 0 && x12 < sample_x && y12 >= 0 && y12 < sample_y && z12 >= 0 && z12 < sample_z)
				{
					int ind_back = z12 * sample_num + y12 * sample_x + x12;
					if (mlflow[0].tag_matrix[ind_back] > 0)
					{
						if (mlflow[0].bubble.volume[mlflow[0].tag_matrix[ind_back] - 1] < 5000000.0)
						{
							float xx = pixx_t45 * invRho - cs2; //-uxux;
							float yy = piyy_t45 * invRho - cs2;
							float zz = pizz_t45 * invRho - cs2;
							float xy = pixy_t90 * invRho;
							float xz = pixz_t90 * invRho;
							float yz = piyz_t90 * invRho;
							float fact2 = 4.0f;
							float vis = fact2 *sqrtf((xx*xx+2 * xy*xy+  2 * xz*xz+ yy*yy+ 2 *yz*yz +  zz*zz));
							Omega = 1 / ((vis+1e-4) * 3.0f + 0.5f); //1e-1
							break;
						}
					}
			}
		}

		mrutilfunc.mlGetPIAfterCollision(
			rhoVar,
			ux_t30,
			uy_t30,
			uz_t30,
			FX,
			FY,
			FZ,
			Omega,
			pixx_t45,
			pixy_t90,
			pixz_t90,
			piyy_t45,
			piyz_t90,
			pizz_t45
		);
		mlflow[0].fMomPost[curind + total_num * 0] = rhoVar;
		mlflow[0].fMomPost[curind + total_num * 1] = ux_t30 + FX * invRho / 2.0f;
		mlflow[0].fMomPost[curind + total_num * 2] = uy_t30 + FY * invRho / 2.0f;
		mlflow[0].fMomPost[curind + total_num * 3] = uz_t30 + FZ * invRho / 2.0f;
		mlflow[0].fMomPost[curind + total_num * 4] = (pixx_t45 * invRho - cs2);
		mlflow[0].fMomPost[curind + total_num * 5] = (pixy_t90 * invRho);
		mlflow[0].fMomPost[curind + total_num * 6] = (pixz_t90 * invRho);
		mlflow[0].fMomPost[curind + total_num * 7] = (piyy_t45 * invRho - cs2);
		mlflow[0].fMomPost[curind + total_num * 8] = (piyz_t90 * invRho);
		mlflow[0].fMomPost[curind + total_num * 9] = (pizz_t45 * invRho - cs2);
	}
}

__global__ void mrSolver3D_step2Kernel(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_num)
{
	MomSwap(mlflow[0].fMom, mlflow[0].fMomPost);
}


__global__ void parse_label(mrFlow3D* mlflow, MLFluidParam3D* param, int sample_x, int sample_y, int sample_z)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		if ((int)mlflow[0].label_matrix[curind] > 0)
		{
			int label = (int)mlflow[0].label_matrix[curind];
			atomicMax(&mlflow[0].bubble.label_num, (int)mlflow[0].label_matrix[curind]);
			atomicAdd(&mlflow[0].bubble.label_volume[label - 1], (double) (1. - mlflow[0].phi[curind]));
		}
	}
}


__global__ void InitTag(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_y * sample_x + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		unsigned char flagsn = mlflow[0].flag[curind];
		const unsigned char flagsn_bo = flagsn & TYPE_BO; // extract boundary flags

		if (flagsn_bo == TYPE_S)
			mlflow[0].tag_matrix[curind] = -1;
		if (flagsn == TYPE_F)
			mlflow[0].tag_matrix[curind] = -1;
		mlflow[0].previous_tag[curind] = -1;
	}
}


__global__ void convertIntToUnsignedChar(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{

		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
		const unsigned char flagsn_bo = mlflow[0].flag[curind] & TYPE_BO;
		if ((flagsn_sus == TYPE_G || flagsn_sus == TYPE_I) && (flagsn_bo != TYPE_S))
			mlflow[0].input_matrix[curind] = 255;
		else
			mlflow[0].input_matrix[curind] = 0;
	}

}



__global__ void create_bubble_label(mrFlow3D* mlflow)
{
	for (int i = 0; i < mlflow[0].bubble.label_num; i++)
	{
		mlflow[0].bubble.volume[i] = mlflow[0].bubble.label_volume[i];
		mlflow[0].bubble.init_volume[i] = mlflow[0].bubble.label_volume[i];
		mlflow[0].bubble.rho[i] = 1.0;
	}
	mlflow[0].bubble.bubble_count = mlflow[0].bubble.label_num;
}

__global__ void update_init_tag(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_y * sample_x + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		if ((int)mlflow[0].label_matrix[curind] > 0)
		{
			mlflow[0].tag_matrix[curind] = (int)mlflow[0].label_matrix[curind];
		}

	}

}

__global__ void print_label_num(mrFlow3D* d_mlflow)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("label count %d\n", d_mlflow[0].bubble.label_num);
	}
}
__global__ void print_bubble(mrFlow3D* d_mlflow)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < d_mlflow[0].bubble.bubble_count; i++)
			printf("bubble %d volume %f init volume %f bubble rho %f\n", i, d_mlflow[0].bubble.volume[i], d_mlflow[0].bubble.init_volume[i], d_mlflow[0].bubble.rho[i]);
	}
}

__global__ void ResetLabelVolume(mrFlow3D* mlflow)
{
	for (int i = 0; i < mlflow[0].bubble.label_num; i++)
	{
		mlflow[0].bubble.label_volume[i] = 0.0;
		mlflow[0].bubble.label_init_volume[i] = 0.f;
	}
	mlflow[0].bubble.label_num = 0;
}


void InitBubble(mrFlow3D* mlflow, MLFluidParam3D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_z = param->samples.z;
	int sample_num = sample_x * sample_y;
	int total_num = sample_num * sample_z;
	dim3 threads1(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		ceil(REAL(sample_z) / threads1.z)
	);

	// set the boundary and fluid as -1
	InitTag << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	printf("InitBubble 2\n");
	
	// prepare for the input bool image
	convertIntToUnsignedChar << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_z);
	printf("InitBubble 3\n");
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	// do labeling
	connectedComponentLabeling(mlflow, sample_x, sample_y, sample_z);
	printf("InitBubble 4\n");
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// parse the label
	parse_label << <grid1, threads1 >> > (mlflow, param, sample_x, sample_y, sample_z);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	print_label_num << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaDeviceSynchronize());
	// create the bubble with label
	create_bubble_label << <1, 1 >> > (mlflow);
	cudaDeviceSynchronize();
	// update the tag matrix
	update_init_tag << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_z, sample_num);
	cudaDeviceSynchronize();
	ClearDectector(mlflow, param);
	print_bubble << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaDeviceSynchronize());
}




__device__ static double atomicExch(double *address, double val)
{
  return __longlong_as_double(atomicExch((unsigned long long int *) address, __double_as_longlong(val)));
}

// update the atmosphere for the open tank
__global__ void atmosphere_rho_update_kernel(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num, int total_num, float N, float l0p, float roup, float labma,
	float u0p, int time) {

		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int z = threadIdx.z + blockDim.z * blockIdx.z;
			int curind = z * sample_num + y * sample_x + x;
			if (
				(x >= 0 && x <= sample_x - 1) &&
				(y >= 0 && y <= sample_y - 1) &&
				(z >= 0 && z <= sample_z - 1)
				)
			{
				{
					if (x == sample_x - 2 || z > sample_z - 10)
					{
						if (mlflow[0].tag_matrix[curind] > 0 && mlflow[0].bubble.volume[mlflow[0].tag_matrix[curind] - 1]>1000000)
							float old = atomicExch(&mlflow[0].bubble.rho[mlflow[0].tag_matrix[curind] - 1], 1.0);
					}
				}
			}
		}
}


// update the atmosphere volume for the open tank
__global__ void atmosphere_volme_update_kernel(mrFlow3D* d_mlflow) {

	{
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			for (int i = 0; i < d_mlflow[0].bubble.bubble_count; i++)
			{
				// update the volume for the atmosphere
				if (d_mlflow[0].bubble.rho[i] == 1.0)
					d_mlflow[0].bubble.init_volume[i] = d_mlflow[0].bubble.rho[i] * d_mlflow[0].bubble.volume[i];
			}
		}
	}
}

// update the volume of the bubble with delta_phi
__global__ void bubble_volume_update_kernel(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num, int total_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
		if (mlflow[0].delta_phi[curind] != 0)
		{
			int tag = mlflow[0].tag_matrix[curind];
			if (tag <= 0)
			{
				tag = mlflow[0].previous_tag[curind];
				mlflow[0].previous_tag[curind] = -1;
			}
			atomicAdd(&mlflow[0].bubble.volume[tag - 1], (double)-mlflow[0].delta_phi[curind]);
			mlflow[0].delta_phi[curind] = 0;
		}
	}
}

void bubble_volume_update(mrFlow3D* mlflow, MLFluidParam3D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_z = param->samples.z;

	int sample_num = sample_x * sample_y;
	int total_num = sample_num * sample_z;
	dim3 threads1(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		ceil(REAL(sample_z) / threads1.z)
	);

	bubble_volume_update_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num, total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}



// update the rho of the bubble
__global__ void bubble_rho_update_kernel(mrFlow3D* mlflow)
{
	for (int i = 0; i < mlflow[0].bubble.bubble_count; i++)
	{

		mlflow[0].bubble.rho[i] = mlflow[0].bubble.init_volume[i] / mlflow[0].bubble.volume[i];
	}
}


__global__ void MergeSplitDetectorKernel(mrFlow3D* mlflow, int* merge_flag, int* split_flag) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*merge_flag = mlflow[0].merge_flag;
		*split_flag = mlflow[0].split_flag;
	}
}


//assign neighbor tag to the current node
__global__ void get_tag_kernel(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;

	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1) &&
		mlflow[0].merge_detector[curind] &&
		mlflow[0].tag_matrix[curind] == -1
		)
	{
		int thisCellID = mlflow[0].tag_matrix[curind];
		for (int i = 1; i < 27; i++)
		{

			int dx = int(ex3d_gpu[i]);
			int dy = int(ey3d_gpu[i]);
			int dz = int(ez3d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int z1 = z - dz;

			int ind_back = z1 * sample_x * sample_y + y1 * sample_x + x1;
			if ((x1 >= 0 && x1 <= sample_x - 1) &&
				(y1 >= 0 && y1 <= sample_y - 1) &&
				(z1 >= 0 && z1 <= sample_z - 1)
				)
			{
				if (mlflow[0].tag_matrix[ind_back] > -1)
				{
					thisCellID = mlflow[0].tag_matrix[ind_back];
				}
			}
		}
		// use previous merge tag to avoid the confict
		mlflow[0].previous_merge_tag[curind] = (thisCellID > 0) ? thisCellID : -1;
	}
	else
	{
		mlflow[0].merge_detector[curind] = 0;
	}
}

// update the tag of the current node to avoid the confict
__global__ void assign_tag_kernel(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num, int total_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		// use previous merge tag to avoid the confict
		if (mlflow[0].previous_merge_tag[curind] > 0 && mlflow[0].merge_detector[curind])

		{
			mlflow[0].tag_matrix[curind] = mlflow[0].previous_merge_tag[curind];
			mlflow[0].previous_merge_tag[curind] = -1;
		}
		else
		{
			mlflow[0].previous_merge_tag[curind] = -1;
		}
	}
}

// identify whether the merge happens or just the bubble moves
__global__ void recheck_merge_kernel(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int total_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;

	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1) &&
		mlflow[0].merge_detector[curind]
		)
	{
		mlflow[0].merge_detector[curind] = 0;
		int thisCellID = -1;
		for (int i = 1; i < 27; i++)
		{
			int dx = int(ex3d_gpu[i]);
			int dy = int(ey3d_gpu[i]);
			int dz = int(ez3d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int z1 = z - dz;

			int ind_back = z1 * sample_x * sample_y + y1 * sample_x + x1;
			if ((x1 >= 0 && x1 <= sample_x - 1) &&
				(y1 >= 0 && y1 <= sample_y - 1) &&
				(z1 >= 0 && z1 <= sample_z - 1)
				)
			{
				if (mlflow[0].tag_matrix[ind_back] > -1)
				{
					if (thisCellID < 0)
					{
						thisCellID = mlflow[0].tag_matrix[ind_back];
					}
					else
					{
						if (thisCellID != mlflow[0].tag_matrix[ind_back])
						{
							atomicExch(&mlflow[0].merge_flag, 1);
						}

					}
				}
			}
		}

	}
}


__global__ void reset_label_volume(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	if (curind < mlflow[0].bubble.max_bubble_count)
	{
		mlflow[0].bubble.label_volume[curind] = 0;
		mlflow[0].bubble.label_init_volume[curind] = 0;
	}
}


__global__ void reduce_label_rho(mrFlow3D* mlflow, MLFluidParam3D* param, int sample_x, int sample_y, int sample_z)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		if ((int)mlflow[0].label_matrix[curind] > 0)
		{
			int label = (int)mlflow[0].label_matrix[curind];
			int tag = mlflow[0].tag_matrix[curind] - 1;
			if (tag >= 0)
			{
				atomicMax(&mlflow[0].bubble.label_num, (int)mlflow[0].label_matrix[curind]);
				atomicAdd(&mlflow[0].bubble.label_volume[label - 1], (double)(1.f - mlflow[0].phi[curind]));
				atomicAdd(&mlflow[0].bubble.label_init_volume[label - 1], (double)(1.f - mlflow[0].phi[curind]) * mlflow[0].bubble.rho[tag]);
			}
			else
			{
				atomicMax(&mlflow[0].bubble.label_num, (int)mlflow[0].label_matrix[curind]);
				atomicAdd(&mlflow[0].bubble.label_volume[label - 1], (double)(1.f - mlflow[0].phi[curind]));
				atomicAdd(&mlflow[0].bubble.label_init_volume[label - 1], (double)(1.f - mlflow[0].phi[curind]) * 1.0);
			}
			mlflow[0].tag_matrix[curind] = label;
		}
		else
		{
			mlflow[0].tag_matrix[curind] = -1;
		}
	}
}

__global__ void bubble_list_swap(mrFlow3D* mlflow)
{
	MomSwap(mlflow[0].bubble.volume, mlflow[0].bubble.label_volume);
	MomSwap(mlflow[0].bubble.init_volume, mlflow[0].bubble.label_init_volume);
}


__global__ void num_rho_update_kernel(mrFlow3D* d_mlflow) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_mlflow[0].bubble.bubble_count = d_mlflow[0].bubble.label_num;
		d_mlflow[0].bubble.label_num = 0;
		for (int i = 0; i < d_mlflow[0].bubble.bubble_count; i++)
		{
			d_mlflow[0].bubble.rho[i] = d_mlflow[0].bubble.init_volume[i] / d_mlflow[0].bubble.volume[i];
		}
	}
}


void handle_merge_spilt(mrFlow3D* mlflow, MLFluidParam3D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_z = param->samples.z;
	int sample_num = sample_x * sample_y;
	int total_num = sample_num * sample_z;
	dim3 threads1(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		ceil(REAL(sample_z) / threads1.z)
	);

	// prepare for the input bool image
	convertIntToUnsignedChar << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_z);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	connectedComponentLabeling(mlflow, sample_x, sample_y, sample_z);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// reduce the label volume with the results of CCL
	reset_label_volume << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_z);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	reduce_label_rho << <grid1, threads1 >> > (mlflow, param, sample_x, sample_y, sample_z);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	bubble_list_swap << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	num_rho_update_kernel << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

void update_bubble(mrFlow3D* mlflow, MLFluidParam3D* param)
{
	// update the delta_phi
	bubble_volume_update(mlflow, param);
	// update the new rho with the updated volume
	bubble_rho_update_kernel << <1, 1 >> > (mlflow);
	cudaDeviceSynchronize();

	// detect the merge and split
	int* d_merge_flag;
	int merge_flag;
	int* d_split_flag;
	int split_flag;
	cudaMalloc(&d_merge_flag, sizeof(int));
	cudaMalloc(&d_split_flag, sizeof(int));
	MergeSplitDetectorKernel << <1, 1 >> > (mlflow, d_merge_flag, d_split_flag);
	cudaDeviceSynchronize();
	cudaMemcpy(&merge_flag, d_merge_flag, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&split_flag, d_split_flag, sizeof(int), cudaMemcpyDeviceToHost);

	if (merge_flag > 0 || split_flag > 0)
	{
		handle_merge_spilt(mlflow, param);
		// clear the merge/split detector
		ClearDectector(mlflow, param);
	}

}

__global__ void g_reconstruction(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int total_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	mrUtilFuncGpu3D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_bo == TYPE_S || flagsn_su == TYPE_G) return; // cell processed here is fluid or interface

		// g temporary streaming
		float ghn[7];
		float gon[7];
		float g_eq_k[7];
		float rhon_g = 0.f, uxn_g = 0.f, uyn_g = 0.f, uzn_g = 0.f;

		for (int i = 0; i < 7; i++)
			rhon_g += mlflow[0].gMom[curind + i * total_num];

		mrutilfunc.calculate_g_eq(rhon_g, uxn_g, uyn_g, uzn_g, g_eq_k);

		// g_eq boundary condition
		for (int i = 0; i < 7; i++)
		{
			int dx = int(ex3d_gpu[i]);
			int dy = int(ey3d_gpu[i]);
			int dz = int(ez3d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int z1 = z - dz;

			int ind_back = z1 * sample_x * sample_y + y1 * sample_x + x1;
			gon[i] = mlflow[0].gMom[curind + i * total_num];

			if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
			{
				
				ghn[i] = g_eq_k[i];
			}
			else
			{
				ghn[i] = mlflow[0].gMom[ind_back + i * total_num];
			}

		}
		gon[0] = ghn[0];

		if (flagsn_su == TYPE_I)
		{ // cell is interface
			float  uxn = 0.0f, uyn = 0.0f, uzn = 0.0f, rho_laplace = 0.0f; // no surface tension if rho_laplace is not overwritten later
			uxn = mlflow[0].fMom[curind + total_num * 1];
			uyn = mlflow[0].fMom[curind + total_num * 2];
			uzn = mlflow[0].fMom[curind + total_num * 3];

			REAL rho_k = 1.f;
			if (mlflow[0].tag_matrix[curind] > 0)
			{
				rho_k = mlflow[0].bubble.rho[mlflow[0].tag_matrix[curind] - 1];
			}
			float in_rho = K_h / 4.f * rho_k;

			// henry's law
			float geg[7]{};
			mrutilfunc.calculate_g_eq(in_rho, uxn, uyn, uzn, geg);
			
			unsigned char flagsj_su[27]; // cache neighbor flags for multiple readings
			unsigned char flagsj_bo[27];
			for (int i = 1; i < 27; i++)
			{
				int dx = int(ex3d_gpu[i]);
				int dy = int(ey3d_gpu[i]);
				int dz = int(ez3d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int z1 = z - dz;
				int ind_back = z1 * sample_x * sample_y + y1 * sample_x + x1;
				flagsj_su[i] = mlflow[0].flag[ind_back] & TYPE_SU;
				flagsj_bo[i] = mlflow[0].flag[ind_back] & TYPE_BO;
			}
			// bubble gas from interface
			float g_in = 0.f;
			for (int i = 1; i < 7; i++)
			{
				g_in += flagsj_su[i] == TYPE_F ? ghn[i] - gon[index3dInv_gpu[i]] : 0.f;
			}
			mlflow[0].delta_g[curind] += g_in;

			for (int i = 1; i < 7; i++)
			{
				if (flagsj_su[i] == TYPE_G)
					mlflow[0].gMom[curind + i * total_num] = geg[index3dInv_gpu[i]] - gon[index3dInv_gpu[i]] + geg[i];
			}
		}

	}
}


__global__ void g_stream_collide(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int total_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	mrUtilFuncGpu3D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_bo == TYPE_S || flagsn_su == TYPE_G) return;
		if (mlflow[0].islet[curind] == 1) return;	
		// inject the g's streaming
		REAL pop_g[7]{};
		
		float g_eq_k[7];
		float rhon_g = 0.f, uxn_g = 0.f, uyn_g = 0.f, uzn_g = 0.f;
		for (int i = 0; i < 7; i++)
			rhon_g += mlflow[0].gMom[curind + i * total_num];
		mrutilfunc.calculate_g_eq(rhon_g, uxn_g, uyn_g, uzn_g, g_eq_k);

		// g_eq boundary condition
		for (int i = 0; i < 7; i++)
		{
			int dx = int(ex3d_gpu[i]);
			int dy = int(ey3d_gpu[i]);
			int dz = int(ez3d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int z1 = z - dz;

			int ind_back = z1 * sample_x * sample_y + y1 * sample_x + x1;

			if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
			{
				pop_g[i] = g_eq_k[i];
			}
			else
			{
				pop_g[i] = mlflow[0].gMom[ind_back + i * total_num];
			}
		}

		float FX = mlflow[0].forcex[curind];
		float FY = mlflow[0].forcey[curind];
		float FZ = mlflow[0].forcez[curind];

		float rhon = 0.f, uxn = 0.f, uyn = 0.f, uzn = 0.f;
		float fxn = FX, fyn = FY, fzn = FZ;


		rhon = mlflow[0].fMom[curind + total_num * 0];
		uxn = mlflow[0].fMom[curind + total_num * 1];
		uyn = mlflow[0].fMom[curind + total_num * 2];
		uzn = mlflow[0].fMom[curind + total_num * 3];

		// D3Q7 g
		float g_eq[7];

		float rhon_gt = 0.f;
		for (int i = 0; i < 7; i++)
			rhon_gt += pop_g[i];

		mrutilfunc.calculate_g_eq(rhon_gt, uxn, uyn, uzn, g_eq);
		mlflow[0].c_value[curind] = rhon_gt;

		// cmr
		float w = 1.0f / 0.53f;
		float src = 0.f;
		mrutilfunc.mlConvertCmrMoment_d3q7(uxn,uyn,uzn,pop_g);
		mrutilfunc.mlConvertCmrMoment_d3q7(uxn,uyn,uzn,g_eq);
		float pop_out[7];
		float s[7];
		s[0] = 1.0f;
		s[1] = s[2] = s[3] = 1.0f / (0.1 * 4  + 0.5);
		s[4] = 1.5f;
		s[5] = s[6] = 1.5f;
		for (int i = 0; i < 7; i++)
		{
			src = mlflow[0].src[curind];
			pop_out[i] = fma(1.0f - s[i], pop_g[i], fma(s[i], g_eq[i], src));
		}
		mrutilfunc.mlConvertCmrF_d3q7(uxn,uyn,uzn,pop_out);
		for (int i = 0; i < 7; i++)
			mlflow[0].gMomPost[curind + i * total_num] = pop_out[i];
	}
}


__global__ void bubble_volume_g_update_kernel(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int total_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	int curind = z * sample_x * sample_y + y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		(z >= 0 && z <= sample_z - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
		float factor;
		if (mlflow[0].delta_g[curind] != 0)
		{
			int tag = mlflow[0].tag_matrix[curind];
			factor = 1.f;
			if (mlflow[0].flag[curind] == TYPE_I)
			{
				atomicAdd(&mlflow[0].bubble.init_volume[tag - 1], (double)1.f / 4.f * factor * mlflow[0].delta_g[curind] * mlflow[0].phi[curind]);
				
			}
			mlflow[0].delta_g[curind] = 0;
		}
	}
}

__global__ void mrSolver3D_g_step2Kernel(
	mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int total_num)
{
	MomSwap(mlflow[0].gMom, mlflow[0].gMomPost);
}


void mrInit3DGpu(mrFlow3D* mlflow, MLFluidParam3D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_z = param->samples.z;
	int sample_num = sample_x * sample_y;
	int total_num = sample_num * sample_z;
	dim3 threads1(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		ceil(REAL(sample_z) / threads1.z)
	);
	Init3D << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num, total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	InitBubble(mlflow, param);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	ResetLabelVolume << <1, 1 >> > (mlflow);
	cudaDeviceSynchronize();

}



void g_handle(mrFlow3D* mlflow, MLFluidParam3D* param, int time)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_z = param->samples.z;
	int total_num = sample_x * sample_y * sample_z;
	dim3 threads1(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		ceil(REAL(sample_z) / threads1.z)
	);
	// g reconstruction
	g_reconstruction << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	g_stream_collide << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			total_num, time
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// updated the volume of the bubble caused by g
	bubble_volume_g_update_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			total_num, time
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	mrSolver3D_g_step2Kernel << <1, 1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	// update the new rho with the updated volume
	bubble_rho_update_kernel << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaDeviceSynchronize());
}


void mrSolver3DGpu(mrFlow3D* mlflow, MLFluidParam3D* param, float N, float l0p, float roup, float labma,
	float u0p, int time_step)
{

	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_z = param->samples.z;
	int sample_num = sample_x * sample_y;
	int total_num = sample_num * sample_z;
	dim3 threads1(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		ceil(REAL(sample_z) / threads1.z)
	);
	// calculate the disjoint force
	calculate_disjoint << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// clear the inlet with some time steps
	if (time_step == 180 * 320 - 5)
	{
		clear_inlet << <grid1, threads1 >> >
			(
				mlflow,
				sample_x, sample_y, sample_z,
				total_num
				);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	// update the atmosphere for the open tank
	atmosphere_rho_update_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num, total_num, N, l0p, roup,
			labma, u0p, time_step
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// update the atmosphere volume for the open tank
	atmosphere_volme_update_kernel << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// FS stream collide
	stream_collide_bvh << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num, total_num, N, l0p, roup,
			labma, u0p, time_step
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// clear the disjoint force
	ResetDisjoinForce << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	surface_1 << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num, total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	surface_2 << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num, total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	surface_3 << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num, total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	mrSolver3D_step2Kernel << <1, 1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}


void coupling(mrFlow3D* mlflow, MLFluidParam3D* param, float N, float l0p, float roup, float labma,
	float u0p, int time_step)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_z = param->samples.z;
	int t = time_step;
	int sample_num = sample_x * sample_y;
	int total_num = sample_num * sample_z;
	dim3 threads1(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		ceil(REAL(sample_z) / threads1.z)
	);

	// assign neighbor tag to the current node
	get_tag_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// update the tag of the current node to avoid the confict
	assign_tag_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			sample_num, total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// identify whether the merge happens or just the bubble moves
	recheck_merge_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y, sample_z,
			total_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// update the bubble
	update_bubble(mlflow, param);
	// handle the g update
	g_handle(mlflow, param, time_step);
	// print the bubble
	if (time_step % 320 == 0 && time_step > 0)
		print_bubble << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaDeviceSynchronize());
}




