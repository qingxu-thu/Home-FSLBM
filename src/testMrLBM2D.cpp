// #include "../inc/2D/cpu/mrSolver2D.h"

// int main()
// {
// 	mrSolver2D mlsolver;
// 	mlsolver.gpuId = 0;
// 	mrFlow2D* mlsmoke0 = new mrFlow2D();
// 	mrFlow2D* mlsmoke_dev0 = 0;


// 	std::vector<mrFlow2D*> lbmvec;
// 	std::vector<mrFlow2D*> lbmvec_gpu;
// 	lbmvec.push_back(mlsmoke0);
// 	lbmvec_gpu.push_back(mlsmoke_dev0);


// 	int nx = 400;
// 	int ny = 500;

// 	float uop = 0.6;
// 	MLMappingParam mparam(uop, 0.375, 1, nx, 1.25);



// 	float L = 1;
// 	float x0 = 0;
// 	float y0 = 0;
// 	float z0 = 0;
// 	float x1 = nx;
// 	float y1 = ny;
// 	float gyP = 0.0;
// 	float gxP = 0.0;
// 	float gzP = 0.0;

// 	float gyL = 0.0;
// 	std::cout << "gyL:	" << gyL << std::endl;
// 	lbmvec[0]->Create
// 	(
// 		0, 0,
// 		nx, ny, 1,
// 		nx, ny,
// 		1.0e-4, //1 * 1e-6
// 		0
// 	);

// 	mlsolver.L = L;
// 	mlsolver.AttachLbmHost(lbmvec);
// 	mlsolver.AttachLbmDevice(lbmvec_gpu);
// 	mlsolver.mlInit();
// 	mlsolver.mlTransData2Gpu();
// 	// std::cout << "mlTransData2Gpu" << std::endl;
// 	int numofiteration = 0;
// 	int numofframe = 0;
// 	int upw = nx * 1;
// 	int uph = ny * 1;
// 	int interationNum = mparam.u0p * mparam.N / (mparam.labma * mparam.l0p * 30);

// 	clock_t start, end;
// 	start = clock();
// 	mlsolver.mlInitGpu();
// 	for (int i = 0; i < (int)lbmvec.size(); i++)
// 	{
// 		mlsolver.mlTransData2Host(i);
// 	}
// 	mlsolver.mlVisVelocitySlice(upw, uph, lbmvec.size(), numofframe);
// 	//mlsolver.mlVisTypeSlice(upw, uph, lbmvec.size(), numofframe);
// 	mlsolver.mlVisCurSlice(upw, uph, lbmvec.size(), numofframe);
// 	mlsolver.mlVisMassSlice(upw, uph, lbmvec.size(), numofframe);
// 	mlsolver.mlVisCurSlice(upw, uph, lbmvec.size(), numofframe);
// 	mlsolver.mlVisDisjoinSlice(upw, uph, lbmvec.size(), numofframe);
// 	//mlsolver.mlVisTagSlice(upw, uph, lbmvec.size(), numofframe);
// 	//mlsolver.mlVisDelatMassSlice(upw, uph, lbmvec.size(), numofframe);
// 	mlsolver.mlVisRhoSlice(upw, uph, lbmvec.size(), numofframe);
// 	numofframe++;

// 	while (numofframe <= 6000)
// 	{

// 		for (int itera = 0; itera < 100; itera++)
// 		{
// 			mlsolver.mlIterateGpu(numofframe);
// 		}
// 		for (int i = 0; i < (int)lbmvec.size(); i++)
// 		{
// 			mlsolver.mlTransData2Host(i);
// 		}
// 		//if (numofframe > 4320)
// 		{
// 			mlsolver.mlVisVelocitySlice(upw, uph, lbmvec.size(), numofframe);
// 			//mlsolver.mlVisTypeSlice(upw, uph, lbmvec.size(), numofframe);
// 			mlsolver.mlVisCurSlice(upw, uph, lbmvec.size(), numofframe);
// 			mlsolver.mlVisMassSlice(upw, uph, lbmvec.size(), numofframe);
// 			mlsolver.mlVisCurSlice(upw, uph, lbmvec.size(), numofframe);
// 			mlsolver.mlVisDisjoinSlice(upw, uph, lbmvec.size(), numofframe);
// 			//mlsolver.mlVisTagSlice(upw, uph, lbmvec.size(), numofframe);
// 			//mlsolver.mlVisDelatMassSlice(upw, uph, lbmvec.size(), numofframe);
// 			mlsolver.mlVisRhoSlice(upw, uph, lbmvec.size(), numofframe);
// 		}
// 		std::cout << "numofiteration: " << numofiteration++ << std::endl;
// 		numofframe++;
// 	}
// 	end = clock();

// 	std::cout << "time:	" << (double)(end - start) / CLOCKS_PER_SEC;
// 	return 0;
// }