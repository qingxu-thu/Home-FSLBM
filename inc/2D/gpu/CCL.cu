/* MIT License
 *
 * Copyright (c) 2018 - Daniel Peter Playne
 *
 * Copyright (c) 2019 - Folke Vesterlund
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


 #include "cuda_runtime.h"
 //#include "../../../lw_core_win/mlCoreWinHeader.h"
#include "../../../common/mlCoreWin.h"
#include "../../../common/mlcudaCommon.h"
#include "CCL.cuh"
#include "reduction.cuh"
#include "../../../common/mlcudaCommon.h"
#include "mrConstantParamsGpu2D.h"
#include "mrUtilFuncGpu2D.h"
#include "mrLbmSolverGpu2D.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/sort.h>
#include <thrust/unique.h>


//there is a problem here
#define BLOCK_SIZE_X 20
#define BLOCK_SIZE_Y 1

/* Connected component labeling on binary images based on
 * the article by Playne and Hawick https://ieeexplore.ieee.org/document/8274991. */
void connectedComponentLabeling(mrFlow2D* mlflow, size_t numCols, size_t numRows)
{
	// Create Grid/Block
	dim3 block (BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 grid ((numCols+BLOCK_SIZE_X-1)/BLOCK_SIZE_X,
			(numRows+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y);

	// Initialise labels
	init_labels<<< grid, block >>>(mlflow, numCols, numRows);
	cudaDeviceSynchronize();
	// Analysis
	resolve_labels <<< grid, block >>>(mlflow, numCols, numRows);
	cudaDeviceSynchronize();
	// Label Reduction
	label_reduction <<< grid, block >>>(mlflow, numCols, numRows);
	cudaDeviceSynchronize();
	// Analysis
	resolve_labels <<< grid, block >>>(mlflow, numCols, numRows);
	cudaDeviceSynchronize();
	// Force background to have label zero;
	resolve_background<<<grid, block>>>(mlflow, numCols, numRows);
	cudaDeviceSynchronize();

	int numPixels = numRows * numCols;
	// Allocate GPU data
	// Uses managed data, so no explicit copies are needed
	unsigned int* d_labels;
	cudaMalloc(&d_labels, numPixels * sizeof(unsigned int));


	thrust::device_vector<unsigned int> d_bubble_map(numPixels);
	//unsigned int* d_bubble_map;
	//cudaMalloc(&d_bubble_map, numPixels * sizeof(int));
	//cudaMemset(d_labels, 0, numPixels * sizeof(int));

	unsigned int *d_ptr = thrust::raw_pointer_cast(d_bubble_map.data());

	//unsigned int h_tmp = 0;
	//unsigned int* d_tmp;
	//// �����豸�ڴ�
	//cudaMalloc(&d_tmp, sizeof(int));
	//cudaMemcpy(d_tmp, &h_tmp, sizeof(int), cudaMemcpyHostToDevice);

	renumber_0 << <grid, block >> > (mlflow, numCols, numRows, d_ptr, d_labels);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::sort(d_bubble_map.begin(), d_bubble_map.end());
	thrust::device_vector<unsigned int>::iterator new_end;
	new_end = thrust::unique(d_bubble_map.begin(), d_bubble_map.end());
	d_bubble_map.erase(new_end, d_bubble_map.end());
	unsigned int* d_ptr_2 = thrust::raw_pointer_cast(d_bubble_map.data());

	if (d_bubble_map.size() > 1)
	{
		renumber_1 << <grid, block >> > (mlflow, numCols, numRows, d_bubble_map.size(), d_ptr_2, d_labels);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		renumber_2 << <grid, block >> > (mlflow, numCols, numRows, d_ptr_2, d_labels);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	//cudaFree(d_tmp);
	cudaFree(d_labels);
	//cudaFree(d_bubble_map);

}

/* CUDA kernels
 */
__global__ void init_labels(mrFlow2D* mlflow, const size_t numCols, const size_t numRows) {
	// Calculate index
	const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

	unsigned int* g_labels = mlflow[0].label_matrix;
	const unsigned char* g_image = mlflow[0].input_matrix;
	// Check Thread Range
	if((ix < numCols) && (iy < numRows)) {
		// Fetch five image values
		
		const unsigned char pyx = g_image[iy*numCols + ix];

		// Neighbour Connections
		const bool nym1x   =  (iy > 0) 					  	 ? (pyx == g_image[(iy-1) * numCols + ix  ]) : false;
		const bool nyxm1   =  (ix > 0)  		  			 ? (pyx == g_image[(iy  ) * numCols + ix-1]) : false;
		const bool nym1xm1 = ((iy > 0) && (ix > 0)) 		 ? (pyx == g_image[(iy-1) * numCols + ix-1]) : false;
		const bool nym1xp1 = ((iy > 0) && (ix < numCols -1)) ? (pyx == g_image[(iy-1) * numCols + ix+1]) : false;

		// Label
		unsigned int label;

		// Initialise Label
		// Label will be chosen in the following order:
		// NW > N > NE > E > current position
		label = (nyxm1)   ?  iy   *numCols + ix-1 : iy*numCols + ix;
		label = (nym1xp1) ? (iy-1)*numCols + ix+1 : label;
		label = (nym1x)   ? (iy-1)*numCols + ix   : label;
		label = (nym1xm1) ? (iy-1)*numCols + ix-1 : label;

		// Write to Global Memory
		g_labels[iy*numCols + ix] = label;
	}
}

// Resolve Kernel
__global__ void resolve_labels(mrFlow2D* mlflow,
		const size_t numCols, const size_t numRows) {
	// Calculate index
	const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
							((blockIdx.x * blockDim.x) + threadIdx.x);

	unsigned int* g_labels = mlflow[0].label_matrix;
	// Check Thread Range
	if(id < (numRows* numCols)) {
		// Resolve Label
		g_labels[id] = find_root(g_labels, g_labels[id]);
	}
}

// Label Reduction
__global__ void label_reduction(mrFlow2D* mlflow,
		const size_t numCols, const size_t numRows) {
	// Calculate index
	const unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
	const unsigned int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);

	unsigned int* g_labels = mlflow[0].label_matrix;
	const unsigned char* g_image = mlflow[0].input_matrix;

	// Check Thread Range
	if((ix < numCols) && (iy < numRows)) {
		// Compare Image Values
		const unsigned char pyx = g_image[iy*numCols + ix];
		const bool nym1x = (iy > 0) ? (pyx == g_image[(iy-1)*numCols + ix]) : false;

		if(!nym1x) {
			// Neighbouring values
			const bool nym1xm1 = ((iy > 0) && (ix >  0)) 		 ? (pyx == g_image[(iy-1) * numCols + ix-1]) : false;
			const bool nyxm1   =              (ix >  0) 		 ? (pyx == g_image[(iy  ) * numCols + ix-1]) : false;
			const bool nym1xp1 = ((iy > 0) && (ix < numCols -1)) ? (pyx == g_image[(iy-1) * numCols + ix+1]) : false;

			if(nym1xp1){
				// Check Criticals
				// There are three cases that need a reduction
				if ((nym1xm1 && nyxm1) || (nym1xm1 && !nyxm1)){
					// Get labels
					unsigned int label1 = g_labels[(iy  )*numCols + ix  ];
					unsigned int label2 = g_labels[(iy-1)*numCols + ix+1];

					// Reduction
					reduction(g_labels, label1, label2);
				}

				if (!nym1xm1 && nyxm1){
					// Get labels
					unsigned int label1 = g_labels[(iy)*numCols + ix  ];
					unsigned int label2 = g_labels[(iy)*numCols + ix-1];

					// Reduction
					reduction(g_labels, label1, label2);
				}
			}
		}
	}
}

// Force background to get label zero;
__global__ void resolve_background(mrFlow2D* mlflow,
		const size_t numCols, const size_t numRows){
	// Calculate index
	const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
							((blockIdx.x * blockDim.x) + threadIdx.x);

	unsigned int* g_labels = mlflow[0].label_matrix;
	const unsigned char* g_image = mlflow[0].input_matrix;
	if(id < numRows*numCols){
		g_labels[id] = ((int) g_image[id] > 0) ? g_labels[id]+1 : 0;
		//if ((int) id >= (399 - 97) * 400 + 2 && (int)id <= (399 - 97) * 400 + 7)
		//{
		//	printf("id %d g_label[id] %d,g_image[id] %d\n", id, g_labels[id], g_image[id]);
		//}
	}
}


__global__ void renumber_0(mrFlow2D* mlflow,
	const size_t numCols, const size_t numRows, unsigned int* d_bubble_map, unsigned int* d_label) {
	// Calculate index
	const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
		((blockIdx.x * blockDim.x) + threadIdx.x);

	unsigned int* g_labels = mlflow[0].label_matrix;
	if (id < numRows * numCols) {
		if (g_labels[id] == id + 1) {
			// atomicAdd(bubble_count, 1);
			d_bubble_map[id] = id + 1;
			//printf("bubble_count: %d\n", *bubble_count);
		}
		else
		{
			d_bubble_map[id] = 0;
		}
	}
}


__global__ void renumber_1(mrFlow2D* mlflow,
	const size_t numCols, const size_t numRows, size_t bubble_count, unsigned int* d_bubble_map, unsigned int* d_label) {
	const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
		((blockIdx.x * blockDim.x) + threadIdx.x);

	// unsigned int* g_labels = mlflow[0].label_matrix;
	if (id < bubble_count) {
		if (id > 0)
		{
			//printf("id %d d_bubble_map: %d\n", id, d_bubble_map[id]);
			d_label[d_bubble_map[id] - 1] = id;
			// printf("d_label: %d\n", id);
		}
	}
}
__global__ void renumber_2(mrFlow2D* mlflow,
	const size_t numCols, const size_t numRows, unsigned int* d_bubble_map, unsigned int* d_label)
{
	const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
		((blockIdx.x * blockDim.x) + threadIdx.x);

	unsigned int* g_labels = mlflow[0].label_matrix;
	if (id < numRows * numCols) {
		if (g_labels[id] > 0)
		{
			g_labels[id] = d_label[g_labels[id] - 1];
			// printf("g_labels: %d\n",g_labels[id]);
		}
	}
		
}