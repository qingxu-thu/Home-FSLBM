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

	void mlInitBoundaryCpu(mrFlow3D*  mlflowvec);
	void mlInitFlowVarCpu(mrFlow3D*  mlflowvec);
	void mlInitInlet(mrFlow3D* mlflowvec);
private:

};

mrInitHandler3D::mrInitHandler3D()
{
}

mrInitHandler3D::~mrInitHandler3D()
{
}

inline void mrInitHandler3D::mlInitBoundaryCpu(mrFlow3D* mlflowvec)
{
	int Nx = mlflowvec->param->samples.x;
	int Ny = mlflowvec->param->samples.y;
	int Nz = mlflowvec->param->samples.z;
#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	for (long z = 0; z < mlflowvec->param->samples.z; z++)
	{
		for (long y = 0; y < mlflowvec->param->samples.y; y++)
		{
			for (long x = 0; x < mlflowvec->param->samples.x; x++)
			{
				int curind = z * mlflowvec->param->samples.y * mlflowvec->param->samples.x + y * mlflowvec->param->samples.x + x;

				if (x == 0)
					mlflowvec->flag[curind] = TYPE_S;
				if (x == Nx - 1)
					mlflowvec->flag[curind] = TYPE_S;
				if (y == Ny - 1)
					mlflowvec->flag[curind] = TYPE_S;
				if (y == 0)
					mlflowvec->flag[curind] = TYPE_S;
				if (z == 0)
					mlflowvec->flag[curind] = TYPE_S;
				if (z == Nz - 1)
					mlflowvec->flag[curind] = TYPE_S;
				
				// set the tank domain
				if (z > 0 && x > 0 && z <= Nz/3-10  && x < Nx-1&&y>0&&y<Ny-1)
				{
					mlflowvec->flag[curind] = TYPE_F;
				}


			}
		}
	}
}



inline void mrInitHandler3D::mlInitInlet(mrFlow3D* mlflowvec)
{
	int total_num = mlflowvec->param->samples.x * mlflowvec->param->samples.y * mlflowvec->param->samples.z;
	int Nx = mlflowvec->param->samples.x;
	int Ny = mlflowvec->param->samples.y;
	int Nz = mlflowvec->param->samples.z;

	for (long z = 0; z < mlflowvec->param->samples.z; z++)
	{
		for (long y = 0; y < mlflowvec->param->samples.y; y++)
		{
			for (long x = 0; x < mlflowvec->param->samples.x; x++)
			{
				int curind = z * mlflowvec->param->samples.y * mlflowvec->param->samples.x + y * mlflowvec->param->samples.x + x;
				mlflowvec->islet[curind] = 0;
				// set the inlet
				if (x==1)
				{
					if (y>=10&&y<=Ny-10&&z>=250&&z<=255)
					{
						mlflowvec->flag[curind] = mlflowvec->postflag[curind] = TYPE_F;
						mlflowvec->islet[curind] = 1;
						mlflowvec->rho[curind] = 1.0;
						mlflowvec->mass[curind] = 1.0;
						mlflowvec->massex[curind] = 0.0;
						mlflowvec->phi[curind] = 1.0;
						mlflowvec->u[curind].x = 7.5e-2;
						mlflowvec->u[curind].y = 0.0;
						mlflowvec->u[curind].z = 0.0;
						mlflowvec->forcez[curind] = -1e-5;
						mlflowvec->c_value[curind] = 0e-3f;
						for (int i = 0; i < 10; i++)
						{
							mlflowvec->fMomPost[total_num * i + curind] = mlflowvec->fMom[total_num * i + curind]
								= 0;
							mlflowvec->fMomViewer[total_num * i + curind] = 0;
						}
						for (int i = 0; i < 7; i++)
						{
							mlflowvec->gMomPost[total_num * i + curind] = mlflowvec->gMom[total_num * i + curind]
								= 0;
						}
						mlflowvec->src[curind] = 0e-7;
					}
				
				}
	
			}
		}
	}

}

inline void mrInitHandler3D::mlInitFlowVarCpu(mrFlow3D* mlflowvec)
{
	int total_num = mlflowvec->param->samples.x * mlflowvec->param->samples.y * mlflowvec->param->samples.z;
	for (long z = 0; z < mlflowvec->param->samples.z; z++)
	{
		for (long y = 0; y < mlflowvec->param->samples.y; y++)
		{
			for (long x = 0; x < mlflowvec->param->samples.x; x++)
			{
				int curind = z * mlflowvec->param->samples.y * mlflowvec->param->samples.x + y * mlflowvec->param->samples.x + x;
				if (mlflowvec->islet[curind] == 1) continue;
				// set the fluid
				if (mlflowvec->flag[curind] == TYPE_F)
				{
					mlflowvec->rho[curind] = 1.0;
					mlflowvec->mass[curind] = 1.0;
					mlflowvec->massex[curind] = 0.0;
					mlflowvec->phi[curind] = 1.0;
					mlflowvec->forcez[curind] = -1e-5;
					mlflowvec->c_value[curind] = 0e-7f;
				}
				else
				{
					mlflowvec->rho[curind] = 1.0;
					mlflowvec->mass[curind] = 0.0;
					mlflowvec->massex[curind] = 0.0;
					mlflowvec->phi[curind] = 0.0;
					mlflowvec->forcez[curind] = -1e-5;
					mlflowvec->c_value[curind] = 0e-7f;
				}
				//set the boundary for the inlet
				if (x==1||x==2)
				{
					if (y>=10&&y<=Ny-10&&(z==249||z==256))
					{
						mlflowvec->flag[curind] = mlflowvec->postflag[curind] = TYPE_S;
						mlflowvec->rho[curind] = 1.0;
						mlflowvec->mass[curind] = 0.0;
						mlflowvec->massex[curind] = 0.0;
						mlflowvec->phi[curind] = 0.0;
					}
					if ((y==9||y==Ny-9)&(z>=249&&z<=256))
					{
						mlflowvec->flag[curind] = mlflowvec->postflag[curind] = TYPE_S;
						mlflowvec->rho[curind] = 1.0;
						mlflowvec->mass[curind] = 0.0;
						mlflowvec->massex[curind] = 0.0;
						mlflowvec->phi[curind] = 0.0;
					}
				}

				for (int i = 0; i < 10; i++)
				{
					mlflowvec->fMomPost[total_num * i + curind] = mlflowvec->fMom[total_num * i + curind]
						= 0;
					mlflowvec->fMomViewer[total_num * i + curind] = 0;
				}
				for (int i = 0; i < 7; i++)
				{
					mlflowvec->gMomPost[total_num * i + curind] = mlflowvec->gMom[total_num * i + curind]
						= 0;
				}
				mlflowvec->src[curind] = 0e-7;
			}
		}
	}
}





#endif // !MRINIT3DH_
