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

	void mlInitBoundaryCpu(mrFlow2D*  mlflowvec);
	void mlInitFlowVarCpu(mrFlow2D* mlflowvec);
	void mlInitBubbleCpu(mrFlow2D* mlflowvec, int numPoints, REAL radius);
private:

};


mrInitHandler2D::mrInitHandler2D()
{
}

mrInitHandler2D::~mrInitHandler2D()
{
}


inline void mrInitHandler2D::mlInitBoundaryCpu(mrFlow2D* mlflowvec)
{
	int Nx = mlflowvec->param->samples.x;
	int Ny = mlflowvec->param->samples.y;
#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	for (long y = 0; y < mlflowvec->param->samples.y; y++)
	{
		for (long x = 0; x < mlflowvec->param->samples.x; x++)
		{
			int curind = y * mlflowvec->param->samples.x + x;

			if (mlflowvec->flag[curind] != ML_SOILD && mlflowvec->flag[curind] != ML_INVALID)
			{
				if (x == 0)
					mlflowvec->flag[curind] = TYPE_S;
				if (x == Nx - 1)
					mlflowvec->flag[curind] = TYPE_S;
				if (y == Ny - 1)
					mlflowvec->flag[curind] = TYPE_S;
				if (y == 0)
					mlflowvec->flag[curind] = TYPE_S;

				if (y > 0 && x > 0 && y <= Ny * 15/50 - 2  && x <= Nx-2 )
				{
					mlflowvec->flag[curind] = TYPE_F;
				}

				if (mlflowvec->flag[curind] == TYPE_S)
					mlflowvec->tag_matrix[curind] = -1;
				else if(mlflowvec->flag[curind] == TYPE_F)
					mlflowvec->tag_matrix[curind] = -1;
			}
		}
	}
}





inline void mrInitHandler2D::mlInitBubbleCpu(mrFlow2D* mlflowvec, int numPoints ,REAL radius_rate)
{
	PoissonGenerator::DefaultPRNG PRNG;
	int max_length = std::max(mlflowvec->param->samples.x, mlflowvec->param->samples.y);
	REAL minDist = radius_rate;
	radius_rate = 0.006;
	auto Points = PoissonGenerator::generatePoissonPoints(250, PRNG, false, 30, -1.0f);

	int total_num = mlflowvec->param->samples.x * mlflowvec->param->samples.y;
	for (long y = 0; y < mlflowvec->param->samples.y; y++)
	{
		for (long x = 0; x < mlflowvec->param->samples.x; x++)
		{
			for (int i = 0; i < Points.size(); i++)
			{
				int curind = y * mlflowvec->param->samples.x + x;
				REAL c_x = Points[i].x * (9.f/10.f) * (REAL)(mlflowvec->param->samples.x);
				REAL c_y = Points[i].y * (14.f/50.f) * (REAL)(mlflowvec->param->samples.y);

				c_x +=  1.f/20.f * (float)mlflowvec->param->samples.x;
				c_y +=  radius_rate * (float)max_length + 2;

				if (
					(powf((float)x - c_x, 2) + powf((float)y - c_y,2) <= radius_rate * radius_rate * (float)max_length * (float)max_length) &&
					(mlflowvec->flag[curind] != TYPE_S)
					)
				{
					mlflowvec->flag[curind] = TYPE_G;
				}
			}
		}
	}
}

inline void mrInitHandler2D::mlInitFlowVarCpu(mrFlow2D* mlflowvec)
{
	int total_num = mlflowvec->param->samples.x * mlflowvec->param->samples.y;
	for (long y = 0; y < mlflowvec->param->samples.y; y++)
	{
		for (long x = 0; x < mlflowvec->param->samples.x; x++)
		{
			int curind = y * mlflowvec->param->samples.x + x;
			if (mlflowvec->flag[curind] == TYPE_F)
			{
				mlflowvec->fMomPost[curind] = mlflowvec->fMom[curind] = 1.0;
				mlflowvec->mass[curind] = 1.0;
				mlflowvec->massex[curind] = 0.0;
				mlflowvec->phi[curind] = 1.0;
				mlflowvec->forcey[curind] = -5e-5;
				mlflowvec->c_value[curind] = 0e-3f;
			}
			else
			{
				mlflowvec->fMomPost[curind] = mlflowvec->fMom[curind]= 1.0;
				mlflowvec->mass[curind] = 0.0;
				mlflowvec->massex[curind] = 0.0;
				mlflowvec->phi[curind] = 0.0;
				mlflowvec->forcey[curind] = -5e-5;
				mlflowvec->c_value[curind] = 0e-3f;
			}
			for (int i = 1; i < 6; i++)
			{
				mlflowvec->fMomPost[total_num * i + curind] = mlflowvec->fMom[total_num * i + curind]
					= 0.f;
			}
			for (int i = 0; i < 5; i++)
			{
				mlflowvec->gMomPost[total_num * i + curind] = mlflowvec->gMom[total_num * i + curind]
					= 0;
			}
			mlflowvec->src[curind] = 5e-5;
		}
	}
}


#endif // !MRINIT2DH_
