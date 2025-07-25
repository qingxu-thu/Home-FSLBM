#pragma once
#ifndef MRINIT2DH_
#define MRINIT2DH_
#include "mrFlow2D.h"
#include "mrConstantParamsCpu2D.h"
#include "PoissonGenerator.h"

class mrInitHandler2D
{
public:
	mrInitHandler2D();
	~mrInitHandler2D();

	void mlInitBoundaryCpu(std::vector<mrFlow2D*>  mlflowvec, int scale, REAL L);
	void mlInitFlowVarCpu(std::vector<mrFlow2D*>  mlflowvec, int scale, REAL L);
	void mlInitBubbleCpu(std::vector<mrFlow2D*> mlflowvec, int numPoints, REAL radius, int scale, REAL L);
private:

};


mrInitHandler2D::mrInitHandler2D()
{
}

mrInitHandler2D::~mrInitHandler2D()
{
}




inline void mrInitHandler2D::mlInitBoundaryCpu(std::vector<mrFlow2D*> mlflowvec, int scale, REAL L)
{
	int Nx = mlflowvec[scale]->param->samples.x;
	int Ny = mlflowvec[scale]->param->samples.y;
#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	for (long y = 0; y < mlflowvec[scale]->param->samples.y; y++)
	{
		for (long x = 0; x < mlflowvec[scale]->param->samples.x; x++)
		{
			int curind = y * mlflowvec[scale]->param->samples.x + x;

			if (mlflowvec[scale]->flag[curind] != ML_SOILD && mlflowvec[scale]->flag[curind] != ML_INVALID)
			{
				REAL _cellsize = mlflowvec[scale]->param->delta_x;
				REAL _endx;
				REAL _endy;

				REAL i, j;
				for (i = mlflowvec[scale]->param->start_pt.x; i < mlflowvec[scale]->param->domian_size.x; i += mlflowvec[scale]->param->delta_x)
				{
				}
				_endx = i - mlflowvec[scale]->param->delta_x;

				for (i = mlflowvec[scale]->param->start_pt.y; i < mlflowvec[scale]->param->domian_size.y; i += mlflowvec[scale]->param->delta_x)
				{
				}
				_endy = i - mlflowvec[scale]->param->delta_x;

				//mlflowvec[scale]->flag[curind] = ML_FLUID;
				if (x == 0)
					mlflowvec[scale]->flag[curind]= mlflowvec[scale]->postflag[curind] = TYPE_S;
				if (x == Nx - 1)
					mlflowvec[scale]->flag[curind] = mlflowvec[scale]->postflag[curind] = TYPE_S;
				if (y == Ny - 1)
					mlflowvec[scale]->flag[curind] = mlflowvec[scale]->postflag[curind] = TYPE_S;
				if (y == 0)
					mlflowvec[scale]->flag[curind] = mlflowvec[scale]->postflag[curind] = TYPE_S;

				if (y > 0 && x > 0 && y <= Ny * 15/50 - 2  && x <= Nx-2 )
				{
					mlflowvec[scale]->flag[curind] = mlflowvec[scale]->postflag[curind] = TYPE_F;
				}

				if (mlflowvec[scale]->flag[curind] == TYPE_S)
					mlflowvec[scale]->tag_matrix[curind] = -1;
				else if(mlflowvec[scale]->flag[curind] == TYPE_F)
					mlflowvec[scale]->tag_matrix[curind] = -1;

				//if (
				//	(powf(x - 0.5 * Nx + 0.5, 2)
				//		)
				//	<= 0.1 * Nx * 0.1 * Nx
				//	&&
				//	y == 0
				//	)
				//{
				//	mlflowvec[scale]->flag[curind] = ML_INLET;
				//}

				//if (
				//	(
				//		powf(x - 0.5 * Nx + 0.5, 2) +
				//		powf(y - 0.4 * Ny + 0.5, 2))
				//	<= 0.03 * Nx * 0.03 * Nx
				//	)
				//{
				//	mlflowvec[scale]->flag[curind] = ML_SOILD;
				//}

			}
		}
	}
}





inline void mrInitHandler2D::mlInitBubbleCpu(std::vector<mrFlow2D*> mlflowvec, int numPoints ,REAL radius_rate, int scale, REAL L)
{
	PoissonGenerator::DefaultPRNG PRNG;
	int max_length = std::max(mlflowvec[scale]->param->samples.x, mlflowvec[scale]->param->samples.y);
	REAL minDist = radius_rate;
	// might be bug here
	radius_rate = 0.006;
	auto Points = PoissonGenerator::generatePoissonPoints(250, PRNG, false, 30, -1.0f);

	//Points.clear();
	
	// Points.push_back({ 0.45756,0.55 });
	// radius_rate = 0.0474513;
	// Points.push_back({ 0.55277,0.55 });
	//radius_rate = 0.13;
	//Points.push_back({ 0.5,0.5615 });

	int total_num = mlflowvec[scale]->param->samples.x * mlflowvec[scale]->param->samples.y;
	//#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	for (long y = 0; y < mlflowvec[scale]->param->samples.y; y++)
	{
		for (long x = 0; x < mlflowvec[scale]->param->samples.x; x++)
		{
			for (int i = 0; i < Points.size(); i++)
			{
				int curind = y * mlflowvec[scale]->param->samples.x + x;
				REAL c_x = Points[i].x * (9.f/10.f) * (REAL)(mlflowvec[scale]->param->samples.x);
				REAL c_y = Points[i].y * (14.f/50.f) * (REAL)(mlflowvec[scale]->param->samples.y);

				c_x +=  1.f/20.f * (float)mlflowvec[scale]->param->samples.x;
				c_y +=  radius_rate * (float)max_length + 2;

				if (
					(powf((float)x - c_x, 2) + powf((float)y - c_y,2) <= radius_rate * radius_rate * (float)max_length * (float)max_length) &&
					(mlflowvec[scale]->flag[curind] != TYPE_S)
					)
				{
					mlflowvec[scale]->flag[curind] = TYPE_G;
					mlflowvec[scale]->postflag[curind] = TYPE_G;
				}
			}
		}
	}
}

inline void mrInitHandler2D::mlInitFlowVarCpu(std::vector<mrFlow2D*> mlflowvec, int scale, REAL L)
{
	int total_num = mlflowvec[scale]->param->samples.x * mlflowvec[scale]->param->samples.y;
	//#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	for (long y = 0; y < mlflowvec[scale]->param->samples.y; y++)
	{
		for (long x = 0; x < mlflowvec[scale]->param->samples.x; x++)
		{
			int curind = y * mlflowvec[scale]->param->samples.x + x;
			REAL rho = 1.0;
			if (mlflowvec[scale]->flag[curind] == TYPE_F)
			{
				mlflowvec[scale]->rho[curind] = rho;
				mlflowvec[scale]->mass[curind] = 1.0;
				mlflowvec[scale]->massex[curind] = 0.0;
				mlflowvec[scale]->phi[curind] = 1.0;
				mlflowvec[scale]->forcey[curind] = -5e-5;
				mlflowvec[scale]->c_value[curind] = 0e-3f;
			}
			else
			{
				mlflowvec[scale]->rho[curind] = 1.0;
				mlflowvec[scale]->mass[curind] = 0.0;
				mlflowvec[scale]->massex[curind] = 0.0;
				mlflowvec[scale]->phi[curind] = 0.0;
				mlflowvec[scale]->forcey[curind] = -5e-5;
				mlflowvec[scale]->c_value[curind] = 0e-3f;
			}
			for (int i = 0; i < 6; i++)
			{
				mlflowvec[scale]->fMomPost[total_num * i + curind] = mlflowvec[scale]->fMom[total_num * i + curind]
					= 0.f;
				mlflowvec[scale]->fMomViewer[total_num * i + curind] = 0.f;
			}
			for (int i = 0; i < 5; i++)
			{
				mlflowvec[scale]->gMomPost[total_num * i + curind] = mlflowvec[scale]->gMom[total_num * i + curind]
					= 0;
			}
			mlflowvec[scale]->src[curind] = 5e-5;

		}
	}
}


#endif // !MRINIT2DH_
