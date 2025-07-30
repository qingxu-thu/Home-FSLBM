
#include "../../../common/mlcudaCommon.h"
#include "mrConstantParamsGpu2D.h"
#include "mrUtilFuncGpu2D.h"
#include "mrLbmSolverGpu2D.h"
#include "CCL.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/host_vector.h>
#include <math_constants.h>

__host__ __device__
void MomSwap2D(MLLATTICENODE_SURFACE_FLAG*& pt1, MLLATTICENODE_SURFACE_FLAG*& pt2) {
	MLLATTICENODE_SURFACE_FLAG* temp = pt1;
	pt1 = pt2;
	pt2 = temp;
}

__host__ __device__
void MomSwap2D(REAL*& pt1, REAL*& pt2) {
	REAL* temp = pt1;
	pt1 = pt2;
	pt2 = temp;
}



__global__ void surface_1(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S); // extract SURFACE flags
		if (flagsn_sus == TYPE_IF)
		{
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int dz = int(ez2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;
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
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S); // extract SURFACE flags
		mrUtilFuncGpu2D mrutilfunc;
		if (flagsn_sus == TYPE_GI) { // initialize the fi of gas cells that should become interface
			float rhon, uxn, uyn, uzn; // average over all fluid/interface neighbors
			float rhot = 0.0f, uxt = 0.0f, uyt = 0.0f, uzt = 0.0f, counter = 0.0f; // average over all fluid/interface neighbors
			float rho_gt = 0, c_k = 0.f;
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;
				const unsigned char flagsji_sus = mlflow[0].flag[ind_back] & (TYPE_SU | TYPE_S); // extract SURFACE flags

				if (flagsji_sus == TYPE_F || flagsji_sus == TYPE_I || flagsji_sus == TYPE_IF) { // fluid or interface or (interface->fluid) neighbor
					counter += 1.0f;
					rhot += mlflow[0].fMom[ind_back + 0 * sample_num];
					uxt += mlflow[0].fMom[ind_back + 1 * sample_num];
					uyt += mlflow[0].fMom[ind_back + 2 * sample_num];
					uzt += mlflow[0].fMom[ind_back + 3 * sample_num];
					if (i < 5)
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

			float feq[9];
			mrutilfunc.calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs
			for (int i = 0; i < 9; i++)
			{
				feq[i] += w2d_gpu[i];
			}
			REAL invRho = 1 / rhon;
			REAL pixx = feq[1] + feq[2] + feq[5] + feq[6] + feq[7] + feq[8];
			REAL piyy = feq[3] + feq[4] + feq[5] + feq[6] + feq[7] + feq[8];
			REAL pixy = feq[5] + feq[6] - feq[7] - feq[8];
			pixx = pixx * invRho - cs2;
			piyy = piyy * invRho - cs2;
			pixy = pixy * invRho;
			mlflow[0].fMomPost[curind + 0 * sample_num] = rhon;
			mlflow[0].fMomPost[curind + 1 * sample_num] = uxn;
			mlflow[0].fMomPost[curind + 2 * sample_num] = uyn;
			mlflow[0].fMomPost[curind + 3 * sample_num] = pixx;
			mlflow[0].fMomPost[curind + 4 * sample_num] = piyy;
			mlflow[0].fMomPost[curind + 5 * sample_num] = pixy;

			mlflow[0].fMom[curind + 0 * sample_num] = rhon;
			mlflow[0].fMom[curind + 1 * sample_num] = uxn;
			mlflow[0].fMom[curind + 2 * sample_num] = uyn;
			mlflow[0].c_value[curind] = rho_gt;

			float geq[5];
			mrutilfunc.calculate_g_eq(rho_gt, uxn, uyn, uzn, geq); // calculate equilibrium DDFs
			for (int i = 0; i < 5; i++)
				mlflow[0].gMom[curind + i * sample_num] = geq[i];

		}
		else if (flagsn_sus == TYPE_IG) { // flag interface->gas is set
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);

				int x1 = x - dx;
				int y1 = y - dy;


				int ind_back = y1 * sample_x + x1;

				const unsigned char flagsji = mlflow[0].flag[ind_back];

				const unsigned char flagsji_su = flagsji & (TYPE_SU | TYPE_S); // extract SURFACE flags
				const unsigned char flagsji_r = flagsji & (~TYPE_SU); // extract all non-SURFACE flags

				if (flagsji_su == TYPE_F || flagsji_su == TYPE_IF) {

					mlflow[0].flag[ind_back] = (MLLATTICENODE_SURFACE_FLAG)(flagsji_r | TYPE_I); // prevent fluid or interface neighbors that turn to fluid from being/becoming fluid
					mlflow[0].merge_detector[ind_back] = 1;
				}

			}

		}
	}
}

__global__ void surface_3(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	mrUtilFuncGpu2D mrutilfunc;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S); // extract SURFACE flags
		if (flagsn_sus & TYPE_S) return;
		const float rhon = mlflow[0].fMom[curind + 0 * sample_num]; // density of cell n
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
			// printf("phin %f", phin);
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
		for (int i = 1; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;


			int ind_back = y1 * sample_x + x1;
			const unsigned char flagsji_su = mlflow[0].flag[ind_back] & (TYPE_SU | TYPE_S); // extract SURFACE flags
			counter += (int)(flagsji_su == TYPE_F || flagsji_su == TYPE_I || flagsji_su == TYPE_IF || flagsji_su == TYPE_GI); // avoid branching

		}
		massn += counter > 0 ? 0.0f : massexn; // if excess mass can't be distributed to neighboring interface or fluid cells, add it to local mass (ensure mass conservation)
		massexn = counter > 0 ? massexn / (float)counter : 0.0f; // divide excess mass up for all interface or fluid neighbors

		mlflow[0].mass[curind] = massn; // update mass
		mlflow[0].massex[curind] = massexn; // update excess mass

		mlflow[0].delta_phi[curind] = phin - mlflow[0].phi[curind]; // calculate phi difference for next step

		// reconstruct true phi
		if ((mlflow[0].flag[curind] & (TYPE_SU | TYPE_S))==TYPE_I)
		{
			float rhon_g = 0.f;
			for (int i = 0; i < 5; i++)
				rhon_g += mlflow[0].gMom[curind + i * sample_num];
			mlflow[0].delta_g[curind] -= rhon_g * (mlflow[0].delta_phi[curind]);
		}

		mlflow[0].phi[curind] = phin; // update phi

	}
}



__global__ void ResetDisjoinForce(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mlflow[0].disjoin_force[curind] = 0.f;
}


__global__ void calculate_disjoint(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_su == TYPE_I)
		{
			float massn = mlflow[0].mass[curind];
			float phij[9]; // cache fill level of neighbor lattice points
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;
				massn += mlflow[0].massex[ind_back]; // distribute excess mass from last step which is stored in neighbors
			}
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);

				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;
				if (mlflow[0].flag[ind_back] & TYPE_SU == TYPE_G)
					phij[i] = 0.f;
				else
					phij[i] = mlflow[0].phi[ind_back]; // cache fill level of neighbor lattice points

			}
			float rhon = 0.0f; 
			rhon = mlflow[0].fMom[curind + 0 * sample_num];
			phij[0] = mrutilfunc.calculate_phi(rhon, massn, flagsn); // don't load phi[n] from memory, instead recalculate it with mass corrected by excess mass
			int tag_curind = mlflow[0].tag_matrix[curind] - 1;
			float3 normal = mrutilfunc.calculate_normal(phij);
			float disjoint = 0.f;
			int max_ids = -1;

			for (int jk = 1; jk < 16; jk++)
			{
				int x12 = round((float)x - (float) jk * 0.2f * normal.x);
				int y12 = round((float)y - (float) jk * 0.2f * normal.y);

				if (x12 >= 0 && x12 < sample_x && y12 >= 0 && y12 < sample_y)
				{
					int ind_back = y12 * sample_x + x12;
					if (mlflow[0].tag_matrix[ind_back] > 0)
					{
						int tag_neighbor = mlflow[0].tag_matrix[ind_back] - 1;
						if (tag_curind!=tag_neighbor && mlflow[0].flag[ind_back] == TYPE_I)
						{
							float center_offset = mrutilfunc.plic_cube(phij[0], normal);
							float alpha = mlflow[0].phi[ind_back]; 
							float dis = abs((float) jk * 0.2f * normal.x);
							float d = abs(dis/(normal.x+1e-8));
							if (disjoint< 1.f - d/3.f)
							{
								disjoint = 1.f - d/3.f;
								max_ids = tag_neighbor;
							}
						}
					}
				}
			}

			if (disjoint>0)
			{
			atomicAdd(&mlflow[0].disjoin_force[curind], disjoint);
			}
		}
	}
}

__global__ void set_atmosphere(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;

	// Note that we only use one in 2D and multiple in 3D, therefore single kernel for 2D and multiple kernels for 3D
	if (x==2&&y==sample_y-2)
	{
		if (mlflow[0].tag_matrix[curind] > 0)
		{
			mlflow[0].bubble.rho[mlflow[0].tag_matrix[curind] - 1] = 1.f;
			mlflow[0].bubble.init_volume[mlflow[0].tag_matrix[curind] - 1] = mlflow[0].bubble.volume[mlflow[0].tag_matrix[curind] - 1];
		}
	}
}

__global__ void stream_collide(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		
		float Omega = 1 / ((1e-4) * 3.0f + 0.5f);
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_bo == TYPE_S||flagsn_su == TYPE_G) return; // cell processed here is fluid or interface

		float fhn[9]{};
		float fon[9]{};


		for (int i = 0; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			int ind_back = y1 * sample_x + x1;

			if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
			{
				float feq[9]{};
				mrutilfunc.calculate_f_eq(rhoVar, 0.f, 0.f, 0.f, feq); // calculate equilibrium DDFs
				fhn[i] = feq[i];
			}
			else
			{
				REAL rhoVar = mlflow[0].fMom[ind_back + 0 * sample_num];
				REAL ux = mlflow[0].fMom[ind_back + 1 * sample_num];
				REAL uy = mlflow[0].fMom[ind_back + 2 * sample_num];
				REAL pixx = mlflow[0].fMom[ind_back + 3 * sample_num];
				REAL piyy = mlflow[0].fMom[ind_back + 4 * sample_num];
				REAL pixy = mlflow[0].fMom[ind_back + 5 * sample_num];

				mrutilfunc.mlCalDistributionD2Q9AtIndex(
					rhoVar, ux, uy, pixx, pixy, piyy, i, fhn[i]
				);
				fhn[i] -= w2d_gpu[i];

			}

			REAL rhoVar = mlflow[0].fMom[curind + 0 * sample_num];
			REAL ux = mlflow[0].fMom[curind + 1 * sample_num];
			REAL uy = mlflow[0].fMom[curind + 2 * sample_num];
			REAL pixx = mlflow[0].fMom[curind + 3 * sample_num];
			REAL piyy = mlflow[0].fMom[curind + 4 * sample_num];
			REAL pixy = mlflow[0].fMom[curind + 5 * sample_num];

			mrutilfunc.mlCalDistributionD2Q9AtIndex(
				rhoVar, ux, uy, pixx, pixy, piyy, i, fon[i]
			);
			fon[i] -= w2d_gpu[i];
		}

		float massn = mlflow[0].mass[curind];

		for (int i = 1; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			int ind_back = y1 * sample_x + x1;
			massn += mlflow[0].massex[ind_back]; // distribute excess mass from last step which is stored in neighbors
		}

		if (flagsn_su == TYPE_F) {
			for (int i = 1; i < 9; i++)
			{
				massn += fhn[i] - fon[i]; // neighbor is fluid or interface cell
			}
		}
		else if (flagsn_su == TYPE_I)
		{ // cell is interface
			float phij[9]; // cache fill level of neighbor lattice points
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);

				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;

				phij[i] = mlflow[0].phi[ind_back]; // cache fill level of neighbor lattice points

				if ((mlflow[0].flag[ind_back] & TYPE_BO)==TYPE_S)
					{	
						if (x1==0)
						{
							int ind_k = y1 * sample_x + 1;
							phij[i] = mlflow[0].phi[ind_k]; 
						}
						else if (x1==sample_x-1)
						{
							int ind_k = y1 * sample_x + sample_x - 2;
							phij[i] = mlflow[0].phi[ind_k]; 
						}
						else if (y1==0)
						{
							int ind_k = 1 * sample_x + x1;
							phij[i] = mlflow[0].phi[ind_k]; 
						}
						else if (y1==sample_y-1)
						{
							int ind_k = (sample_y - 2) * sample_x + x1;
							phij[i] = mlflow[0].phi[ind_k]; 
						}
					}
			}

			float feg[9]; // reconstruct f from neighbor gas lattice points
			float rhon = 0.0f, uxn = 0.0f, uyn = 0.0f, uzn = 0.0f, rho_laplace = 0.0f; // no surface tension if rho_laplace is not overwritten later
			rhon = mlflow[0].fMom[curind + 0 * sample_num];
			uxn = mlflow[0].fMom[curind + 1 * sample_num];
			uyn = mlflow[0].fMom[curind + 2 * sample_num];
			uzn = 0.f;

			phij[0] = mrutilfunc.calculate_phi(rhon, massn, flagsn); // don't load phi[n] from memory, instead recalculate it with mass corrected by excess mass

			float3 normal = mrutilfunc.calculate_normal(phij);
			float curv = mrutilfunc.calculate_curvature(phij);
			rho_laplace = def_6_sigma == 0.0f ? 0.0f : def_6_sigma * curv; // surface tension least squares fit (PLIC, most accurate)
			int tag_curind = mlflow[0].tag_matrix[curind] - 1;
			float disjoint = mlflow[0].disjoin_force[curind];
			const float rho2tmp = 0.5f / rhon; // apply external volume force (Guo forcing, Krueger p.233f)
			float uxntmp = fma(mlflow[0].forcex[curind] * rhon, rho2tmp, uxn);// limit velocity (for stability purposes)
			float uyntmp = fma(mlflow[0].forcey[curind] * rhon, rho2tmp, uyn);// force term: F*dt/(2*rho)
			float uzntmp = uzn;
			float3 u_2{ uxntmp,uyntmp,uzntmp };
			u_2 = normalizing_clamp(u_2, 0.4);
			uxntmp = u_2.x;
			uyntmp = u_2.y;
			uzntmp = u_2.z;
			float rho_k = 1.f;
			if (mlflow[0].tag_matrix[curind] > 0)
			{
				rho_k = mlflow[0].bubble.rho[mlflow[0].tag_matrix[curind] - 1];			
			}
			rho_laplace = def_6_sigma == 0.0f ? 0.0f : def_6_sigma * curv;
			float in_rho = rho_k -  rho_laplace - 0.05 * disjoint;
			mrutilfunc.calculate_f_eq(in_rho, uxntmp, uyntmp, uzntmp, feg); // calculate gas equilibrium DDFs with constant ambient pressure
			

			unsigned char flagsj_su[9]; // cache neighbor flags for multiple readings
			unsigned char flagsj_bo[9];
			int num;
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int ind_back = y1 * sample_x + x1;
				flagsj_su[i] = mlflow[0].flag[ind_back] & TYPE_SU;
				flagsj_bo[i] = mlflow[0].flag[ind_back] & TYPE_BO;
			}
			
			for (int i = 1; i < 9; i++)
			{ 
				massn += flagsj_su[i] & (TYPE_F | TYPE_I) ? flagsj_su[i] == TYPE_F ? fhn[i] - fon[index2dInv_gpu[i]] : 0.5f * (phij[i] + phij[0]) * (fhn[i] - fon[index2dInv_gpu[i]]) : 0.0f; // neighbor is fluid or interface cell	
			}

			int flag_s;
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int ind_back = y1 * sample_x + x1;

				if (mlflow[0].flag[ind_back]==TYPE_S)
				{
					if (x1==0||x1==sample_x-1)
					{
						if (sign(normal.y)!= -1 *sign((float)dy))
							flag_s = 1;
					}
					if (y1==0||y1==sample_y-1)
					{
						if (sign(normal.x)!= -1 * sign((float)dx))
							flag_s = 1;
					}
				}
				if (flagsj_su[i] == TYPE_G||flag_s == 1)
					fhn[i] = feg[index2dInv_gpu[i]] - fon[index2dInv_gpu[i]] + feg[i];
			}
		}

		mlflow[0].mass[curind] = massn;
		
		REAL pop[9]{};
		for (int i = 0; i < 9; i++)
		{
			pop[i] =  fhn[i] + w2d_gpu[i];
		}

		REAL FX = mlflow[0].forcex[curind];
		REAL FY = mlflow[0].forcey[curind];


		float rhon = 0.f, uxn = 0.f, uyn = 0.f, uzn = 0.f;


		rhon = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8];
		float fxn = FX * rhon, fyn = FY * rhon, fzn = 0.f;
		REAL invRho = 1 / rhon;
		uxn = ((pop[1] - pop[2] + pop[5] - pop[8] - pop[6] + pop[7]) * invRho + 0.5 * FX * invRho);
		uyn = ((pop[3] - pop[4] + pop[5] + pop[8] - pop[6] - pop[7]) * invRho + 0.5 * FY * invRho);
		REAL pixx = pop[1] + pop[2] + pop[5] + pop[8] + pop[6] + pop[7];
		REAL piyy = pop[3] + pop[4] + pop[5] + pop[8] + pop[6] + pop[7];
		REAL pixy = pop[5] - pop[8] + pop[6] - pop[7];

		
		float3 u_{ uxn,uyn,uzn };
		u_ = normalizing_clamp(u_, 0.4);
		uxn = u_.x;
		uyn = u_.y;
		uzn = u_.z;
		if (flagsn_su == TYPE_I)
		{
			bool TYPE_NO_F = true, TYPE_NO_G = true; // temporary flags for no fluid or gas neighbors
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);

				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;
				const unsigned char flagsji_su = mlflow[0].flag[ind_back] & TYPE_SU; // extract SURFACE flags
				TYPE_NO_F = TYPE_NO_F && flagsji_su != TYPE_F;
				TYPE_NO_G = TYPE_NO_G && flagsji_su != TYPE_G;
			}
			REAL massn = mlflow[0].mass[curind];
			//printf("massn : %f, rhon: %f\n", massn, rhon);
			if (massn > rhon || TYPE_NO_G)
			{
				mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((flagsn & ~TYPE_SU) | TYPE_IF); // set flag interface->fluid
			}
			else if (massn < 0.0f || TYPE_NO_F)
			{
				mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((flagsn & ~TYPE_SU) | TYPE_IG); // set flag interface->gas
			}
		}

		// foam ossifying viscosity
		for (int ij = -2;ij<2;ij++)
			for(int jk=-2;jk<2;jk++)
			{
				int x12 = x + jk;
				int y12 = y + ij;
				if (x12 >= 0 && x12 < sample_x && y12 >= 0 && y12 < sample_y)
				{
					int ind_back = (y + ij) * sample_x + x + jk;
					if (mlflow[0].tag_matrix[ind_back] > 0)
					{
						Omega =  1.f / 0.9f;
						break;
					}
					if (mlflow[0].flag[ind_back] == TYPE_S)
					{
						Omega =  1.f / (3.0 * (1e-3) + 0.5f);
						break;
					}
				}
			}

		mrutilfunc.mlGetPIAfterCollision(
			rhon,
			uxn, uyn, FX, FY,
			Omega,
			pixx, piyy, pixy
		);
		pixx = pixx * invRho - 1.0 * cs2;
		piyy = piyy * invRho - 1.0 * cs2;
		pixy = pixy * invRho;

		mlflow[0].fMomPost[curind + 0 * sample_num] = rhon;
		mlflow[0].fMomPost[curind + 1 * sample_num] = uxn + FX * invRho / 2.0f;
		mlflow[0].fMomPost[curind + 2 * sample_num] = uyn + FY * invRho / 2.0f;
		mlflow[0].fMomPost[curind + 3 * sample_num] = pixx;
		mlflow[0].fMomPost[curind + 4 * sample_num] = piyy;
		mlflow[0].fMomPost[curind + 5 * sample_num] = pixy;

	}
}

__global__ void Init2D(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;


	mrUtilFuncGpu2D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{

		unsigned char flagsn = mlflow[0].flag[curind];
		const unsigned char flagsn_bo = flagsn & TYPE_BO; // extract boundary flags

		unsigned char flagsj[9]{};
		for (int i = 0; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			if (x1 >= 0 && x1 < sample_x && y1 >= 0 && y1 < sample_y)
			{
				int ind_back = y1 * sample_x + x1;
				flagsj[i] = mlflow[0].flag[ind_back];
			}
		}

		// need to change for the moving boundary the speed need to change for no zero initial
		if (flagsn_bo == TYPE_S) { // cell is solid
			bool TYPE_ONLY_S = true; // has only solid neighbors
			for (int i = 1; i < 9; i++) TYPE_ONLY_S = TYPE_ONLY_S && (flagsj[i] & TYPE_BO) == TYPE_S;
			if (TYPE_ONLY_S) {
				mlflow[0].fMomPost[curind + 1 * sample_num] = mlflow[0].fMom[curind + 1 * sample_num] = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
				mlflow[0].fMomPost[curind + 2 * sample_num] = mlflow[0].fMom[curind + 2 * sample_num] = 0.0f;
			}
		}
		if (flagsn_bo == TYPE_S) {
			mlflow[0].fMomPost[curind + 1 * sample_num] = mlflow[0].fMom[curind + 1 * sample_num] = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
			mlflow[0].fMomPost[curind + 2 * sample_num] = mlflow[0].fMom[curind + 2 * sample_num] = 0.0f;
		}



		float feq[9]{}; // f_equilibrium

		mrutilfunc.calculate_f_eq(mlflow[0].fMom[curind + 0 * sample_num], 
			mlflow[0].fMom[curind + 1 * sample_num], mlflow[0].fMom[curind + 2 * sample_num], 
			mlflow[0].fMom[curind + 3 * sample_num], feq);

		float geq[5]{};
		mrutilfunc.calculate_g_eq(mlflow[0].c_value[curind], 
			mlflow[0].fMom[curind + 1 * sample_num], mlflow[0].fMom[curind + 2 * sample_num], 
			mlflow[0].fMom[curind + 3 * sample_num], geq);


		float phin = mlflow[0].phi[curind];
		if (!(flagsn & (TYPE_S | TYPE_E | TYPE_T | TYPE_F | TYPE_I)))
			flagsn = (flagsn & ~TYPE_SU) | TYPE_G; // change all non-fluid and non-interface flags to gas
		if ((flagsn & TYPE_SU) == TYPE_G)
		{ // cell with updated flags is gas
			bool change = false; // check if cell has to be changed to interface
			for (int i = 1; i < 9; i++)
				change = change || (flagsj[i] & TYPE_SU) == TYPE_F; // if neighbor flag fluid is set, the cell must be interface
			if (change)
			{ // create interface automatically if phi has not explicitely defined for the interface layer
				flagsn = (flagsn & ~TYPE_SU) | TYPE_I; // cell must be interface
				phin = 0.5f;
				float rhon, uxn, uyn, uzn, rhon_g; // initialize interface cells with average density/velocity of fluid neighbors

				 // average over all fluid/interface neighbors
				float rhot = 0.0f, uxt = 0.0f, uyt = 0.0f, uzt = 0.0f, counter = 0.0f,rhogt = 0.0f, c_k = 0.0f; // average over all fluid/interface neighbors
				for (int i = 1; i < 9; i++)
				{
					int dx = int(ex2d_gpu[i]);
					int dy = int(ey2d_gpu[i]);
					int x1 = x - dx;
					int y1 = y - dy;
					int ind_back = y1 * sample_x + x1;
					const unsigned char flagsji_su = mlflow[0].flag[ind_back] & TYPE_SU;
					if (flagsji_su == TYPE_F) { // fluid or interface or (interface->fluid) neighbor
						counter += 1.0f;
						rhot += mlflow[0].fMom[ind_back + 0 * sample_num];
						uxt += mlflow[0].fMom[ind_back + 1 * sample_num];
						uyt += mlflow[0].fMom[ind_back + 2 * sample_num];
						if (i < 5)
						{
							c_k += 1.0f;
							rhogt += mlflow[0].c_value[ind_back];
						}
					}

				}
				rhon = counter > 0.0f ? rhot / counter : 1.0f;
				uxn = counter > 0.0f ? uxt / counter : 0.0f;
				uyn = counter > 0.0f ? uyt / counter : 0.0f;
				uzn = 0.0f;
				rhon_g = c_k > 0.0f ? rhogt / c_k : 0.0f;
				mrutilfunc.calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs
				mrutilfunc.calculate_g_eq(rhon_g, uxn, uyn, uzn, geq); // calculate equilibrium DDFs
			}
		}
		if ((flagsn & TYPE_SU) == TYPE_G) { // cell with updated flags is still gas
			mlflow[0].fMom[curind + 1 * sample_num] = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
			mlflow[0].fMom[curind + 2 * sample_num] = 0.0f;
			phin = 0.0f;
		}
		else if ((flagsn & TYPE_SU) == TYPE_I && (phin < 0.0f || phin>1.0f)) {
			phin = 0.5f; // cell should be interface, but phi was invalid

		}
		else if ((flagsn & TYPE_SU) == TYPE_F) {
			phin = 1.0f;
		}
		mlflow[0].phi[curind] = phin;
		mlflow[0].mass[curind] = phin * mlflow[0].fMom[curind + 0 * sample_num];
		mlflow[0].massex[curind] = 0.0f; // reset excess mass
		mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)flagsn;

		for (int i = 0; i < 9; i++)
		{
			feq[i] += w2d_gpu[i];
		}
		REAL invRho = 1 / mlflow[0].fMom[curind + 0 * sample_num];
		REAL pixx = feq[1] + feq[2] + feq[5] + feq[8] + feq[6] + feq[7];
		REAL piyy = feq[3] + feq[4] + feq[5] + feq[8] + feq[6] + feq[7];
		REAL pixy = feq[5] - feq[8] + feq[6] - feq[7];
		pixx = pixx * invRho - cs2;
		piyy = piyy * invRho - cs2;
		pixy = pixy * invRho;
		mlflow[0].fMomPost[curind + 0 * sample_num] = mlflow[0].fMom[curind + 0 * sample_num];
		mlflow[0].fMomPost[curind + 1 * sample_num] = mlflow[0].fMom[curind + 1 * sample_num];
		mlflow[0].fMomPost[curind + 2 * sample_num] = mlflow[0].fMom[curind + 2 * sample_num];
		mlflow[0].fMomPost[curind + 3 * sample_num] = mlflow[0].fMom[curind + 3 * sample_num] = pixx;
		mlflow[0].fMomPost[curind + 4 * sample_num] = mlflow[0].fMom[curind + 4 * sample_num] = piyy;
		mlflow[0].fMomPost[curind + 5 * sample_num] = mlflow[0].fMom[curind + 5 * sample_num] = pixy;

		for (int i = 0; i < 5; i++)
		{
			mlflow[0].gMom[curind + i * sample_num] = geq[i];
			mlflow[0].gMomPost[curind + i * sample_num] = geq[i];
		}

	}

}


__global__ void InitTag(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
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




__global__ void create_bubble_label(mrFlow2D* mlflow)
{
	for (int i = 0; i < mlflow[0].bubble.label_num; i++)
	{
		mlflow[0].bubble.volume[i] = mlflow[0].bubble.label_volume[i];
		mlflow[0].bubble.init_volume[i] = mlflow[0].bubble.label_volume[i];
		mlflow[0].bubble.rho[i] = 1.0f;
	}
	mlflow[0].bubble.bubble_count = mlflow[0].bubble.label_num;

}


__global__ void update_init_tag(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{

		if ((int)mlflow[0].label_matrix[curind] > 0)
		{

			mlflow[0].tag_matrix[curind] = (int)mlflow[0].label_matrix[curind];

		}

	}

}

__global__ void mrSolver2D_step2Kernel(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	MomSwap2D(mlflow[0].fMom, mlflow[0].fMomPost);
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


//assign neighbor tag to the current node
__global__ void get_tag_kernel(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z, int sample_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;

	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
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

			int ind_back = y1 * sample_x + x1;
			if ((x1 >= 0 && x1 <= sample_x - 1) &&
				(y1 >= 0 && y1 <= sample_y - 1)
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



__global__ void assign_tag_kernel(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;

	// need to fix the following
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
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


__global__ void recheck_merge_kernel(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;

	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		mlflow[0].merge_detector[curind]
		)
	{
		mlflow[0].merge_detector[curind] = 0;
		int thisCellID = -1;
		for (int i = 1; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			int ind_back = y1 * sample_x + x1;
			if ((x1 >= 0 && x1 <= sample_x - 1) &&
				(y1 >= 0 && y1 <= sample_y - 1)
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


__global__ void convertIntToUnsignedChar(mrFlow2D* mlflow, int sample_x, int sample_y) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
		const unsigned char flagsn_bo = mlflow[0].flag[curind] & TYPE_BO;
		if ((flagsn_sus == TYPE_G || flagsn_sus == TYPE_I)&&(flagsn_bo !=TYPE_S))
			mlflow[0].input_matrix[curind] = 255;
		else
			mlflow[0].input_matrix[curind] = 0;
	}

}

__global__ void parse_label(mrFlow2D* mlflow, MLFluidParam2D* param, int sample_x, int sample_y)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		if ((int)mlflow[0].label_matrix[curind] > 0)
		{
			int label = (int)mlflow[0].label_matrix[curind];
			atomicMax(&mlflow[0].bubble.label_num, (int)mlflow[0].label_matrix[curind]);
			atomicAdd(&mlflow[0].bubble.label_volume[label -1], 1.f - mlflow[0].phi[curind]);
		}
	}
}

__global__ void ResetLabelVolume(mrFlow2D* mlflow)
{
	for (int i = 0; i < mlflow[0].bubble.label_num; i++)
	{
		mlflow[0].bubble.label_volume[i] = 0.f;
		mlflow[0].bubble.label_init_volume[i] = 0.f;
	}
	mlflow[0].bubble.label_num = 0;
}


__global__ void bubble_volume_update_kernel(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
		if (mlflow[0].delta_phi[curind]!= 0)
		{
			int tag = mlflow[0].tag_matrix[curind];
			if (tag <= 0)
			{
				tag = mlflow[0].previous_tag[curind];
				mlflow[0].previous_tag[curind] = -1;

			}
			atomicAdd(&mlflow[0].bubble.volume[tag - 1], -mlflow[0].delta_phi[curind]);
			mlflow[0].delta_phi[curind] = 0;
		}
	}
}



__global__ void clear_detector(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
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


__global__ void PushLabelNumKernel(mrFlow2D* d_mlflow, int *d_bubble_count) {

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_mlflow[0].bubble.label_num = *d_bubble_count;
	}
}

__global__ void print_bubble(mrFlow2D* d_mlflow)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < d_mlflow[0].bubble.bubble_count; i++)
			printf("bubble %d volume %f init volume %f bubble rho %f\n", i, d_mlflow[0].bubble.volume[i], d_mlflow[0].bubble.init_volume[i], d_mlflow[0].bubble.rho[i]);
	}
}


void ClearDectector(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);

	clear_detector << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void bubble_volume_update(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;

	int sample_num = sample_x * sample_y;

	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	bubble_volume_update_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}







__global__ void bubble_rho_update_kernel(mrFlow2D* mlflow)
{
	for (int i = 0; i < mlflow[0].bubble.bubble_count; i++)
	{	
		mlflow[0].bubble.rho[i] = mlflow[0].bubble.init_volume[i] / mlflow[0].bubble.volume[i];
	}

}


__global__ void MergeSplitDetectorKernel(mrFlow2D* mlflow,int * merge_flag, int * split_flag) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*merge_flag = mlflow[0].merge_flag;
		*split_flag = mlflow[0].split_flag;
	}
}




__global__ void reset_label_volume(mrFlow2D* mlflow, int sample_x, int sample_y)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (curind < mlflow[0].bubble.max_bubble_count)
	{
		mlflow[0].bubble.label_volume[curind] = 0;
		mlflow[0].bubble.label_init_volume[curind] = 0;
	}
}


__global__ void reduce_label_rho(mrFlow2D* mlflow, MLFluidParam2D* param, int sample_x, int sample_y)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		if ((int)mlflow[0].label_matrix[curind] > 0)
		{
			int label = (int)mlflow[0].label_matrix[curind];
			int tag = mlflow[0].tag_matrix[curind] - 1;
			if (tag >= 0)
			{
				atomicMax(&mlflow[0].bubble.label_num, (int)mlflow[0].label_matrix[curind]);
				atomicAdd(&mlflow[0].bubble.label_volume[label - 1], (1.f - mlflow[0].phi[curind]));
				atomicAdd(&mlflow[0].bubble.label_init_volume[label - 1], (1.f - mlflow[0].phi[curind]) * mlflow[0].bubble.rho[tag]);
			}
			else
			{
				atomicMax(&mlflow[0].bubble.label_num, (int)mlflow[0].label_matrix[curind]);
				atomicAdd(&mlflow[0].bubble.label_volume[label - 1], (1.f - mlflow[0].phi[curind]));
				atomicAdd(&mlflow[0].bubble.label_init_volume[label - 1], (1.f - mlflow[0].phi[curind]) * 1.0);
			}
			mlflow[0].tag_matrix[curind] = label;
		}
		else
		{
			mlflow[0].tag_matrix[curind] = -1;
		}
	}
}


__global__ void bubble_list_swap(mrFlow2D* mlflow)
{
	MomSwap2D(mlflow[0].bubble.volume, mlflow[0].bubble.label_volume);
	MomSwap2D(mlflow[0].bubble.init_volume, mlflow[0].bubble.label_init_volume);
}

__global__ void num_rho_update_kernel(mrFlow2D* d_mlflow) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_mlflow[0].bubble.bubble_count = d_mlflow[0].bubble.label_num;
		d_mlflow[0].bubble.label_num = 0;
		for (int i = 0; i < d_mlflow[0].bubble.bubble_count; i++)
		{
			d_mlflow[0].bubble.rho[i] = d_mlflow[0].bubble.init_volume[i] / d_mlflow[0].bubble.volume[i];
		}
	}
}

void handle_merge_spilt(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);

	// prepare for the input bool image
	convertIntToUnsignedChar << <grid1, threads1 >> > (mlflow, sample_x, sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	connectedComponentLabeling(mlflow, (size_t)sample_x, (size_t)sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// reduce the label volume with the results of CCL
	reset_label_volume << <grid1, threads1 >> > (mlflow, sample_x, sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	reduce_label_rho << <grid1, threads1 >> > (mlflow, param, sample_x, sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	bubble_list_swap << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	num_rho_update_kernel << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}


void update_bubble(mrFlow2D* mlflow, MLFluidParam2D* param)
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


void InitBubble(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	InitTag << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// prepare for the input bool image
	convertIntToUnsignedChar << <grid1, threads1 >> > (mlflow, sample_x, sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// do labeling
	connectedComponentLabeling(mlflow, (size_t)sample_x, (size_t)sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	parse_label << <grid1, threads1 >> > (mlflow, param, sample_x, sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	create_bubble_label << <1, 1 >> > (mlflow);
	cudaDeviceSynchronize();
	update_init_tag << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_num);
	cudaDeviceSynchronize();
	ClearDectector(mlflow, param);
	print_bubble << <1, 1 >> > (mlflow);
	checkCudaErrors(cudaDeviceSynchronize());
}



void mrInit2DGpu(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	mrUtilFuncGpu2D mrutilfunc;
	Init2D << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	InitBubble(mlflow, param);
	ResetLabelVolume<<<1,1>>>(mlflow);
	cudaDeviceSynchronize();
}


__global__ void g_reconstruction(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_bo == TYPE_S || flagsn_su == TYPE_G) return; // cell processed here is fluid or interface


		// g temporary streaming
		float ghn[5];
		float gon[5];
		float g_eq_k[5];
		float rhon_g = 0.f, uxn_g = 0.f, uyn_g = 0.f, uzn_g = 0.f;
		
		for (int i = 0; i < 5; i++)
			rhon_g += mlflow[0].gMom[curind + i * sample_num];

		mrutilfunc.calculate_g_eq(rhon_g, uxn_g, uyn_g, uzn_g, g_eq_k);


		for (int i = 0; i < 5; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int ind_back = y1 * sample_x + x1;
			gon[i] = mlflow[0].gMom[curind + i * sample_num];
			if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
			{
				
				ghn[i] = g_eq_k[i];
			}
			else
			{
				ghn[i] = mlflow[0].gMom[ind_back + i * sample_num];
			}

		}
		gon[0] = ghn[0];


		if (flagsn_su == TYPE_I)
		{
			float rhon = 0.0f, uxn = 0.0f, uyn = 0.0f, uzn = 0.0f, rho_laplace = 0.0f; // no surface tension if rho_laplace is not overwritten later
			rhon = mlflow[0].fMom[curind * 6 + 0];
			uxn = mlflow[0].fMom[curind * 6 + 1];
			uyn = mlflow[0].fMom[curind * 6 + 2];
			REAL rho_k = 1.f;
			if (mlflow[0].tag_matrix[curind] > 0)
			{
				rho_k = mlflow[0].bubble.rho[mlflow[0].tag_matrix[curind] - 1];
			}
			float in_rho = K_h  /3.f * rho_k;
			// henry's law
			float geg[5]{};
			mrutilfunc.calculate_g_eq(in_rho, uxn, uyn, uzn, geg);
			unsigned char flagsj_su[9]; // cache neighbor flags for multiple readings
			unsigned char flagsj_bo[9];
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int ind_back = y1 * sample_x + x1;
				flagsj_su[i] = mlflow[0].flag[ind_back] & TYPE_SU;
				flagsj_bo[i] = mlflow[0].flag[ind_back] & TYPE_BO;
			}
			float g_delta = 0.f;
			for (int i = 1; i < 5; i++)
			{
				g_delta += flagsj_su[i] == TYPE_F ? ghn[i] - gon[index2dInv_gpu[i]] : 0.f;
			}
			mlflow[0].delta_g[curind] += g_delta;

			for (int i = 1; i < 5; i++)
			{
				if (flagsj_su[i] == TYPE_G)
					mlflow[0].gMom[ind_back + i * sample_num] = geg[index2dInv_gpu[i]] - gon[index2dInv_gpu[i]] + geg[i];
			}
		}

	}
}

__global__ void g_stream_collide(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_bo == TYPE_S || flagsn_su == TYPE_G) return;


		// inject the g's streaming
		REAL pop_g[5]{};
		float g_eq_k[5];
		float rhon_g = 0.f, uxn_g = 0.f, uyn_g = 0.f, uzn_g = 0.f;
		for (int i = 0; i < 5; i++)
			rhon_g += mlflow[0].gMom[curind + i * sample_num];
		mrutilfunc.calculate_g_eq(rhon_g, uxn_g, uyn_g, uzn_g, g_eq_k);
		

		for (int i = 0; i < 5; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);

			int x1 = x - dx;
			int y1 = y - dy;
			int ind_back = y1 * sample_x + x1;
			if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
			{
				pop_g[i] = g_eq_k[i];
			}
			else
			{
				pop_g[i] = mlflow[0].gMom[ind_back + i * sample_num];
			}
		}


		REAL FX = mlflow[0].forcex[curind];
		REAL FY = mlflow[0].forcey[curind];

		float rhon = 0.f, uxn = 0.f, uyn = 0.f, uzn = 0.f;
		float fxn = FX, fyn = FY, fzn = 0.f;

		rhon = mlflow[0].fMom[curind + 0 * sample_num];
		uxn = mlflow[0].fMom[curind + 1 * sample_num];
		uyn = mlflow[0].fMom[curind + 2 * sample_num];
		
		// D2Q5 g
		float g_eq[5];
		float rhon_gt = 0.f;
		for (int i = 0; i < 5; i++)	
			rhon_gt += pop_g[i];
		mrutilfunc.calculate_g_eq(rhon_gt, uxn, uyn, uzn, g_eq);
		mlflow[0].c_value[curind] = rhon_gt;
		REAL w = 1.0f / 0.6f;
		float src = 0.f;
		for (int i = 0; i < 5; i++)
		{
			src = mlflow[0].src[curind];
			REAL pop_out;
			pop_out = fma(1.0f - w, pop_g[i], fma(w, g_eq[i], src));
			mlflow[0].gMomPost[curind + i * sample_num] = pop_out;
		}
	}
}


__global__ void mrSolver2D_g_step2Kernel(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	MomSwap2D(mlflow[0].gMom, mlflow[0].gMomPost);
}


__global__ void bubble_volume_g_update_kernel(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
		
		float factor = 0.85f;
		if (mlflow[0].delta_g[curind] != 0)
		{
			int tag = mlflow[0].tag_matrix[curind];
			if (mlflow[0].flag[curind]  == TYPE_I)
			{
				{
					atomicAdd(&mlflow[0].bubble.init_volume[tag - 1], 1.f/3.f *factor * mlflow[0].delta_g[curind] * mlflow[0].phi[curind]);
				}
			}
			mlflow[0].delta_g[curind] = 0;	
		}
		
	}
}


void g_handle(mrFlow2D* mlflow, MLFluidParam2D* param ,int time)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	// g reconstruction
	g_reconstruction << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	g_stream_collide << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num, time
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// updated the volume of the bubble caused by g
	bubble_volume_g_update_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num, time
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	mrSolver2D_g_step2Kernel << <1, 1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// update the new rho with the updated volume
	bubble_rho_update_kernel << <1, 1 >> > (mlflow);
}




void mrSolver2DGpu(mrFlow2D* mlflow, MLFluidParam2D* param, int time_step)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int t = time_step;
	int sample_num = sample_x * sample_y;

	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	// calculate the disjoint force
	calculate_disjoint << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// set the atmosphere for the open tank
	set_atmosphere << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num,t
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());	
	// FS stream collide	
	stream_collide << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num, t
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// clear the disjoint force
	ResetDisjoinForce << <grid1, threads1 >> >
	(
		mlflow,
		sample_x, sample_y,
		sample_num
		);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	surface_1 << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	surface_2 << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	surface_3 << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	mrSolver2D_step2Kernel << <1, 1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// assign neighbor tag to the current node
	get_tag_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	// update the tag of the current node to avoid the confict
	assign_tag_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// identify whether the merge happens or just the bubble moves
	recheck_merge_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// update the bubble
	update_bubble(mlflow, param);
	// handle the g update
	g_handle(mlflow, param, time_step);

	if (time_step % 100 == 0)
	{
		print_bubble << <1, 1 >> > (mlflow);
	}
}
