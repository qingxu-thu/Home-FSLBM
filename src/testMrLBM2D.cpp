#include "../inc/2D/cpu/mrSolver2D.h"

int main()
{
	mrSolver2D mlsolver;
	mlsolver.gpuId = 0;
	mrFlow2D* lbmvec = new mrFlow2D();
	mrFlow2D* lbmvec_gpu = 0;
	
	int nx = 400;
	int ny = 500;
	float uop = 0.6;
	float labma = 0.375;
	float l0p = 1;
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
		x0, y0,
		nx, ny, delta_x,
		nx, ny,
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
	int uph = ny;
	mlsolver.mlVisVelocitySlice(upw, uph, numofframe);
	mlsolver.mlVisMassSlice(upw, uph, numofframe);
	numofframe++;

	while (numofframe <= 800)
	{

		for (int itera = 0; itera < 100; itera++)
		{
			mlsolver.mlIterateGpu(numofframe);
		}
		mlsolver.mlTransData2Host();
		
        mlsolver.mlVisVelocitySlice(upw, uph, numofframe);
        mlsolver.mlVisMassSlice(upw, uph, numofframe);
		std::cout << "numofiteration: " << numofiteration++ << std::endl;
		numofframe++;
	}

	return 0;
}