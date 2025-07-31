#include "../inc/3D/cpu/mrSolver3D.h"

int main()
{
	mrSolver3D mlsolver;
	mlsolver.gpuId = 0;
	mrFlow3D* lbmvec = new mrFlow3D();
	mrFlow3D* lbmvec_gpu = 0;
	
	int nx = 600;
	int ny = 300;
	int nz = 300;
	float uop = 18;
	float labma = 0.3;
	float l0p = 10;
	float N = 1.25;

	float L = 1;
	float x0 = 0;
	float y0 = 0;
	float z0 = 0;

	float delta_x = 1;
	float vis = 1.0e-4;
	float gy = 0.0;

	MLMappingParam mparam(uop, labma, l0p, nx, N);
	mparam.tp = mparam.labma * mparam.l0p / (mparam.u0p * mparam.N);

	lbmvec->Create
	(
		x0, y0, z0,
		nx, ny, nz, delta_x,
		nx, ny, nz,
		vis,
		gy
	);

	mlsolver.AttachLbmHost(lbmvec);
	mlsolver.AttachLbmDevice(lbmvec_gpu);
	mlsolver.AttachMapping(mparam);

	mlsolver.mlInit();
	mlsolver.mlTransData2Gpu();

	int numofiteration = 0;
	int numofframe = 0;
	int interationNum = mparam.u0p * mparam.N / (mparam.labma * mparam.l0p * 30);
	std::cout << "interationNum:	" << interationNum << std::endl;

	mlsolver.mlInitGpu();
	mlsolver.mlTransData2Host();

	int upw = nx;
	int uph = nz;
	mlsolver.mlSavePhi(upw, uph, numofframe);
	mlsolver.mlVisVelocitySlice(upw, uph, numofframe);
	mlsolver.mlVisMassSlice(upw, uph, numofframe);
	numofframe++;
	
	while (numofframe <= 600)
	{

		for (int itera = 0; itera < 320; itera++)
		{
			mlsolver.mlIterateCouplingGpu(numofframe*320+itera);
		}
		mlsolver.mlTransData2Host();
		mlsolver.mlVisVelocitySlice(upw, uph, numofframe);
		mlsolver.mlVisMassSlice(upw, uph, numofframe);
		mlsolver.mlSavePhi(upw, uph, numofframe);
		std::cout << "numofiteration: " << numofiteration++ << std::endl;
		numofframe++;
	}

	return 0;
}