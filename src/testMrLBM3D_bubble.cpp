#include "../inc/3D/cpu/mrSolver3D.h"

int main()
{
	mrSolver3D mlsolver;
	mlsolver.gpuId = 0;
	mrFlow3D* mlsmoke0 = new mrFlow3D();
	mrFlow3D* mlsmoke_dev0 = 0;

	std::vector<mrFlow3D*> lbmvec;
	std::vector<mrFlow3D*> lbmvec_gpu;
	lbmvec.push_back(mlsmoke0);
	lbmvec_gpu.push_back(mlsmoke_dev0);

	float scaletime = 1;
	int nx = 600 * scaletime;
	int ny = 300 * scaletime;
	int nz = 300 * scaletime;


	float uop = 18;
	/*
		REAL _uop,
		REAL _labma,
		REAL _l0p,
		REAL _N,
		REAL _roup
	*/

	MLMappingParam mparam(uop, 0.3, 10, nx, 1.25);
	mparam.tp = mparam.labma * mparam.l0p / (mparam.u0p * mparam.N);


	float L = 1;
	float x0 = 0;
	float y0 = 0;
	float z0 = 0;
	float x1 = nx;
	float y1 = ny;
	float dt = 0.02;
	float gyP = 0.0;
	float gxP = 0.0;
	float gzP = -9.8;

	float gyL = -9.8 * (mparam.l0p * mparam.labma * mparam.labma) / (mparam.N * (mparam.u0p * mparam.u0p));
	std::cout << "gyL:	" << gyL << std::endl;

	lbmvec[0]->Create
	(
		0, 0, 0,
		nx, ny, nz, 1,
		nx, ny, nz,
		1.0e-4, //1 * 1e-6
		0
	);

	mlsolver.L = L;
	
	float scale_rate = (mparam.u0p / mparam.labma) * (mparam.u0p / mparam.labma) *
		(mparam.l0p / mparam.N) * (mparam.l0p / mparam.N) * 800 * 1;
	std::cout << "scale_rate:	" << scale_rate << std::endl;

	mlsolver.AttachLbmHost(lbmvec);
	mlsolver.AttachLbmDevice(lbmvec_gpu);
	mlsolver.AttachMapping(mparam);

	mlsolver.mlInit();
	mlsolver.mlTransData2Gpu();

	int numofiteration = 0;
	int numofframe = 0;
	int upw = nx * 1;
	int uph = nz * 1;
	int interationNum = mparam.u0p * mparam.N / (mparam.labma * mparam.l0p * 30);

	std::cout << "interationNum:	" << interationNum << std::endl;

	clock_t start, end;
	mlsolver.istwoway = true;
	start = clock();
	mlsolver.mlInitGpu();
	for (int i = 0; i < (int)lbmvec.size(); i++)
	{
		mlsolver.mlTransData2Host(i);
	}
	mlsolver.mlSavePhi(upw, uph, lbmvec.size(), numofframe);
	mlsolver.mlVisVelocitySlice(upw, uph, lbmvec.size(), numofframe);
	mlsolver.mlVisMassSlice(upw, uph, lbmvec.size(), numofframe);
	numofframe++;

	while (numofframe <= 600)
	{

		for (int itera = 0; itera < 320; itera++)
		{
			mlsolver.mlIterateCouplingGpu(numofframe*320+itera);
		}
		for (int i = 0; i < (int)lbmvec.size(); i++)
		{
			mlsolver.mlTransData2Host(i);
		}
		printf("draw vel\n");
		mlsolver.mlVisVelocitySlice(upw, uph, lbmvec.size(), numofframe);
		printf("draw mass\n");
		mlsolver.mlVisMassSlice(upw, uph, lbmvec.size(), numofframe);
		mlsolver.mlSavePhi(upw, uph, lbmvec.size(), numofframe);
		std::cout << "numofiteration: " << numofiteration++ << std::endl;
		numofframe++;
	}
	end = clock();

	std::cout << "time:	" << (double)(end - start) / CLOCKS_PER_SEC;
	return 0;
}