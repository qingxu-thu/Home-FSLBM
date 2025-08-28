#pragma once
#ifndef _MRLBMSOLVERGPU3DH_
#define _MRLBMSOLVERGPU3DH_

#include "../cpu/mrFlow3D.h"

extern "C"
{
	void mrSolver3DGpu(mrFlow3D* mlflow, MLFluidParam3D* param, float N, float l0p, float roup, float labma,
		float u0p, int time_step);
	void mrInit3DGpu(mrFlow3D* mlflow, MLFluidParam3D* param);
	void coupling(mrFlow3D* mlflow, MLFluidParam3D* param, float N, float l0p, float roup, float labma,
		float u0p, int time_step);
}
#endif // !_MRLBMSOLVERGPU3DH_
