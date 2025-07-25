#pragma once
#ifndef _MRLBMSOLVERGPU2DH_
#define _MRLBMSOLVERGPU2DH_

#include "../cpu/mrFlow2D.h"

extern "C"
{
	void mrSolver2DGpu(mrFlow2D* mlflow, MLFluidParam2D* param, int time_step);
	void mrInit2DGpu(mrFlow2D* mlflow, MLFluidParam2D* param);
}
#endif // !_MRLBMSOLVERGPU2DH_
