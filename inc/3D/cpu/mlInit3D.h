#pragma once
#ifndef MRINIT3DH_
#define MRINIT3DH_
#include "mrFlow3D.h"
#include "mrConstantParamsCpu3D.h"


#define TPH_POISSON_IMPLEMENTATION
#include "../common/tph_poisson.h"
#include <array> 

#include <vector>
#include <cstdint>    // UINT64_C, etc
#include <cstdio>     // std::printf
#include <functional> // std::function
#include <memory>     // std::unique_ptr

// #include "PoissonGenerator.h"
class mrInitHandler3D
{
public:
	mrInitHandler3D();
	~mrInitHandler3D();

	void mlInitBoundaryCpu(std::vector<mrFlow3D*>  mlflowvec, int scale, REAL L);
	void mlInitFlowVarCpu(std::vector<mrFlow3D*>  mlflowvec, int scale, REAL L);
	void mlInitInlet(std::vector<mrFlow3D*> mlflowvec, int scale, REAL L);
private:

};

mrInitHandler3D::mrInitHandler3D()
{
}

mrInitHandler3D::~mrInitHandler3D()
{
}

inline void mrInitHandler3D::mlInitBoundaryCpu(std::vector<mrFlow3D*> mlflowvec, int scale, REAL L)
{
	int Nx = mlflowvec[scale]->param->samples.x;
	int Ny = mlflowvec[scale]->param->samples.y;
	int Nz = mlflowvec[scale]->param->samples.z;
#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	for (long z = 0; z < mlflowvec[scale]->param->samples.z; z++)
	{
		for (long y = 0; y < mlflowvec[scale]->param->samples.y; y++)
		{
			for (long x = 0; x < mlflowvec[scale]->param->samples.x; x++)
			{
				int curind = z * mlflowvec[scale]->param->samples.y * mlflowvec[scale]->param->samples.x + y * mlflowvec[scale]->param->samples.x + x;



				REAL _cellsize = mlflowvec[scale]->param->delta_x;
				REAL _endx;
				REAL _endy;
				REAL _endz;

				REAL i, j;
				for (i = mlflowvec[scale]->param->start_pt.x; i < mlflowvec[scale]->param->domian_size.x; i += mlflowvec[scale]->param->delta_x)
				{
				}
				_endx = i - mlflowvec[scale]->param->delta_x;

				for (i = mlflowvec[scale]->param->start_pt.y; i < mlflowvec[scale]->param->domian_size.y; i += mlflowvec[scale]->param->delta_x)
				{
				}
				_endy = i - mlflowvec[scale]->param->delta_x;

				for (i = mlflowvec[scale]->param->start_pt.z; i < mlflowvec[scale]->param->domian_size.z; i += mlflowvec[scale]->param->delta_x)
				{
				}
				_endz = i - mlflowvec[scale]->param->delta_x;

				//mlflowvec[scale]->flag[curind] = ML_FLUID;
				if (x == 0)
					mlflowvec[scale]->flag[curind] = TYPE_S;
				if (x == Nx - 1)
					mlflowvec[scale]->flag[curind] = TYPE_S;
				if (y == Ny - 1)
					mlflowvec[scale]->flag[curind] = TYPE_S;
				if (y == 0)
					mlflowvec[scale]->flag[curind] = TYPE_S;
				if (z == 0)
					mlflowvec[scale]->flag[curind] = TYPE_S;
				if (z == Nz - 1)
					mlflowvec[scale]->flag[curind] = TYPE_S;

				// if (z>=Nz* 3/5-10 && z<=Nz* 3/5+10)
				// {
				// 	if (!(x>=Nx/2-15&&x<=Nx/2+15&&y>=Ny/2-15&&y<=Ny/2+15))
				// 	{
				// 		mlflowvec[scale]->flag[curind] = TYPE_S;
				// 	}
				// }
				
				if (z > 0 && x > 0 && z <= Nz/3-10  && x < Nx-1&&y>0&&y<Ny-1)
				{
					mlflowvec[scale]->flag[curind] = TYPE_F;
				}

				// if (z > Nz * 3/5+3 && x > 0 && z <= Nz * 3/5+3 + Nz / 8 * 1 && x < Nx-1&&y>0&&y<Ny-1)
				// {
				// 	mlflowvec[scale]->flag[curind] = TYPE_F;
				// }

				// if (x >=Nx/3-1&& x <Nx/3+5 &&z>Nz*2/5 && z <= Nz-1 &&y>0&&y<Ny-1)
				// {
				// 	mlflowvec[scale]->flag[curind] = TYPE_S;
				// }
				// if (z > 0 && x >=Nx/3-1&& x <Nx/3+5 &&z<=Nz*2/5 && z <= Nz-1 &&y>0&&y<Ny-1)
				// {
				// 	if ((y-1)%4>=2)
				// 		mlflowvec[scale]->flag[curind] = TYPE_S;
				// }
				//if (z >= Nz / 2 - 5 && z <= Nz / 2 + 5)
				//{
				//	mlflowvec[scale]->flag[curind] = TYPE_I;
				//}

				//if (
				//	(
				//		powf(x - 0.5 * Nx + 0.5, 2) +
				//		powf(y - 0.4 * Ny + 0.5, 2))
				//	<= 0.03 * Nx * 0.03 * Nx
				//	)
				//{
				//	mlflowvec[scale]->flag[curind] = TYPE_S;
				//}


			}
		}
	}
}



inline void mrInitHandler3D::mlInitInlet(std::vector<mrFlow3D*> mlflowvec, int scale, REAL L)
{
	int total_num = mlflowvec[scale]->param->samples.x * mlflowvec[scale]->param->samples.y * mlflowvec[scale]->param->samples.z;
	int Nx = mlflowvec[scale]->param->samples.x;
	int Ny = mlflowvec[scale]->param->samples.y;
	int Nz = mlflowvec[scale]->param->samples.z;

	//#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	for (long z = 0; z < mlflowvec[scale]->param->samples.z; z++)
	{
	for (long y = 0; y < mlflowvec[scale]->param->samples.y; y++)
	{
		for (long x = 0; x < mlflowvec[scale]->param->samples.x; x++)
		{
			int curind = z * mlflowvec[scale]->param->samples.y * mlflowvec[scale]->param->samples.x + y * mlflowvec[scale]->param->samples.x + x;
			mlflowvec[scale]->islet[curind] = 0;
			REAL rho = 1.0;

			int dis = 2;
			int left = Nx/2 - dis;
			int right = Nx/2 + dis;

			//if (x <= left+sqrt(3) && x>=left-sqrt(3))
			double b0 = 200 - sqrt(3) * (left-sqrt(3));
			double b1 = 200 + sqrt(3) * (right+sqrt(3));
			if (x==1)
			{
				//if (z == (x - left-sqrt(3))/sqrt(3)+200) && ((float)x-left) )
				// if (((float)(z-200)-((float)(x-left)+sqrt(3))/sqrt(3)<=0.5)
				// 	&&(((float)(x-left)+sqrt(3))/sqrt(3)-(float)(z-200)>=-0.5)
				// 	&&(((float)(x-left))*((float)(x-left))+((float)(z-201))*((float)(z-201))+((float)(y-Ny/2))*
				// 	((float)(y-Ny/2))<=4))
				if (y>=10&&y<=Ny-10&&z>=250&&z<=255)
				{
					//printf("insss\n");
				mlflowvec[scale]->flag[curind] = mlflowvec[scale]->postflag[curind] = TYPE_F;
				mlflowvec[scale]->islet[curind] = 1;
				mlflowvec[scale]->rho[curind] = rho;
				mlflowvec[scale]->mass[curind] = 1.0;
				mlflowvec[scale]->massex[curind] = 0.0;
				mlflowvec[scale]->phi[curind] = 1.0;
				mlflowvec[scale]->u[curind].x = 7.5e-2;
				mlflowvec[scale]->u[curind].y = 0.0;
				mlflowvec[scale]->u[curind].z = 0.0;
				mlflowvec[scale]->forcez[curind] = -1e-5;
				mlflowvec[scale]->c_value[curind] = 0e-3f;
				for (int i = 0; i < 10; i++)
				{
					mlflowvec[scale]->fMomPost[total_num * i + curind] = mlflowvec[scale]->fMom[total_num * i + curind]
						= 0;
					mlflowvec[scale]->fMomViewer[total_num * i + curind] = 0;
				}
				for (int i = 0; i < 7; i++)
				{
					mlflowvec[scale]->gMomPost[total_num * i + curind] = mlflowvec[scale]->gMom[total_num * i + curind]
						= 0;
				}
				mlflowvec[scale]->src[curind] = 0e-7;
				}
				
			}
	
		}
	}
	}

}

inline void mrInitHandler3D::mlInitFlowVarCpu(std::vector<mrFlow3D*> mlflowvec, int scale, REAL L)
{
	int total_num = mlflowvec[scale]->param->samples.x * mlflowvec[scale]->param->samples.y * mlflowvec[scale]->param->samples.z;
	//#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	int Nx = mlflowvec[scale]->param->samples.x;
	int Ny = mlflowvec[scale]->param->samples.y;
	int Nz = mlflowvec[scale]->param->samples.z;
	for (long z = 0; z < mlflowvec[scale]->param->samples.z; z++)
	{
		for (long y = 0; y < mlflowvec[scale]->param->samples.y; y++)
		{
			for (long x = 0; x < mlflowvec[scale]->param->samples.x; x++)
			{
				int curind = z * mlflowvec[scale]->param->samples.y * mlflowvec[scale]->param->samples.x + y * mlflowvec[scale]->param->samples.x + x;
				REAL rho = 1.0;
				if (mlflowvec[scale]->islet[curind] == 1) continue;
				if (mlflowvec[scale]->flag[curind] == TYPE_F)
				{
					mlflowvec[scale]->rho[curind] = rho;
					mlflowvec[scale]->mass[curind] = 1.0;
					mlflowvec[scale]->massex[curind] = 0.0;
					mlflowvec[scale]->phi[curind] = 1.0;
					mlflowvec[scale]->forcez[curind] = -1e-5;
					mlflowvec[scale]->c_value[curind] = 0e-7f;
				}
				else
				{
					mlflowvec[scale]->rho[curind] = 1.0;
					mlflowvec[scale]->mass[curind] = 0.0;
					mlflowvec[scale]->massex[curind] = 0.0;
					mlflowvec[scale]->phi[curind] = 0.0;
					mlflowvec[scale]->forcez[curind] = -1e-5;
					mlflowvec[scale]->c_value[curind] = 0e-7f;
				}
				if (x==1||x==2)
				{
					if (y>=10&&y<=Ny-10&&(z==249||z==256))
					{
						mlflowvec[0]->flag[curind] = mlflowvec[0]->postflag[curind] = TYPE_S;
						mlflowvec[0]->rho[curind] = 1.0;
						mlflowvec[0]->mass[curind] = 0.0;
						mlflowvec[0]->massex[curind] = 0.0;
						mlflowvec[0]->phi[curind] = 0.0;
					}
					if ((y==9||y==Ny-9)&(z>=249&&z<=256))
					{
						mlflowvec[0]->flag[curind] = mlflowvec[0]->postflag[curind] = TYPE_S;
						mlflowvec[0]->rho[curind] = 1.0;
						mlflowvec[0]->mass[curind] = 0.0;
						mlflowvec[0]->massex[curind] = 0.0;
						mlflowvec[0]->phi[curind] = 0.0;
					}
				}

				for (int i = 0; i < 10; i++)
				{
					mlflowvec[scale]->fMomPost[total_num * i + curind] = mlflowvec[scale]->fMom[total_num * i + curind]
						= 0;
					mlflowvec[scale]->fMomViewer[total_num * i + curind] = 0;
				}
			for (int i = 0; i < 7; i++)
			{
				mlflowvec[scale]->gMomPost[total_num * i + curind] = mlflowvec[scale]->gMom[total_num * i + curind]
					= 0;
			}
			mlflowvec[scale]->src[curind] = 0e-7;
			}
		}
	}

}





#endif // !MRINIT3DH_
