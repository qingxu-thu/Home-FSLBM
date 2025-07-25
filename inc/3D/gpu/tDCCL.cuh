#ifndef tDCCL_CUH_
#define tDCCL_CUH_

#include "../../../common/mlcudaCommon.h"
#include "mrConstantParamsGpu3D.h"
#include "mrUtilFuncGpu3D.h"
#include "mrLbmSolverGpu3D.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/sort.h>
#include <thrust/unique.h>


void connectedComponentLabeling(mrFlow3D* mlflow, int numCols, int numRows, int numDepths);

#endif /* CCL_CUH_ */


