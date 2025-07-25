
#include "../../../common/mlcudaCommon.h"
#include "mrConstantParamsGpu2D.h"
#include "mrUtilFuncGpu2D.h"
#include "mrLbmSolverGpu2D.h"
#include "CCL.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/host_vector.h>
#include <math_constants.h>

//fix the volume freeze for the initial bubble

__host__ __device__ inline void swap(int& a, int& b) {
	int temp = a;
	a = b;
	b = temp;
}

__device__ inline void swap(float& a, float& b) {
	int temp = a;
	a = b;
	b = temp;
}

__device__ inline void swap(float3& a, float3& b) {
	float3 temp = a;
	a = b;
	b = temp;
}

__host__ bool isRenamed(int* renameVec, int id) {
	return renameVec[id] != id;
}

__host__ __device__
bool operator==(const int2& a, const int2& b) {
	return (a.x == b.x) && (a.y == b.y);
}

__host__ __device__
bool operator<(const int2& a, const int2& b) {
	if (a.x != b.x)
		return a.x < b.x;
	else
		return a.y < b.y;
}

// can't use the vector in the kernel

__global__ void bubble_merge(mrFlow2D* mlflow, int bubble_0, int bubble_1)
{
	// printf("merge bubble %d volume %f, bubble %d volume %f\n", bubble_0, mlflow[0].bubble.volume[bubble_0], bubble_1, mlflow[0].bubble.volume[bubble_1]);
	mlflow[0].bubble.freeze[bubble_0] = (int)((mlflow[0].bubble.freeze[bubble_0] + mlflow[0].bubble.freeze[bubble_1])>0);

	if (mlflow[0].bubble.freeze[bubble_0] > 0)
	{
		mlflow[0].bubble.init_volume[bubble_0] += mlflow[0].bubble.init_volume[bubble_1];
		mlflow[0].bubble.volume[bubble_0] += mlflow[0].bubble.volume[bubble_1];
		mlflow[0].bubble.rho[bubble_0] = mlflow[0].bubble.init_volume[bubble_0] / mlflow[0].bubble.volume[bubble_0];
		mlflow[0].bubble.pure_gas_volume[bubble_0] += mlflow[0].bubble.pure_gas_volume[bubble_1];
	}
	else
	{
		mlflow[0].bubble.init_volume[bubble_0] += mlflow[0].bubble.init_volume[bubble_1];
		mlflow[0].bubble.volume[bubble_0] += mlflow[0].bubble.volume[bubble_1];
		mlflow[0].bubble.rho[bubble_0] = 1.f;
		mlflow[0].bubble.pure_gas_volume[bubble_0] += mlflow[0].bubble.pure_gas_volume[bubble_1];
	}

}

__global__ void bubble_swap(mrFlow2D* mlflow, int bubble_0, int bubble_1)
{
	//printf("bubble count %d\n", mlflow[0].bubble.bubble_count);
	swap(mlflow[0].bubble.init_volume[bubble_0], mlflow[0].bubble.init_volume[bubble_1]);
	swap(mlflow[0].bubble.volume[bubble_0], mlflow[0].bubble.volume[bubble_1]);
	swap(mlflow[0].bubble.rho[bubble_0], mlflow[0].bubble.rho[bubble_1]);
	swap(mlflow[0].bubble.pure_gas_volume[bubble_0], mlflow[0].bubble.pure_gas_volume[bubble_1]);
	swap(mlflow[0].bubble.freeze[bubble_0], mlflow[0].bubble.freeze[bubble_1]);

}




__global__ void reportInterfaceToLiquidConversion_kernel(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		// need to clear the previous tag
		if (mlflow[0].previous_tag[curind] > 0)
		{
			int prev_tag = mlflow[0].previous_tag[curind];
			{
				{
					mlflow[0].split_detector[curind] = 1;
					mlflow[0].split_tag_record[curind] = prev_tag;
					mlflow[0].split_flag = 1;
				}
			}
		}

	}
}


__device__ void reportInterfaceToLiquidConversion(mrFlow2D* mlflow, int x, int y, int sample_x, int sample_y)
{
	int old_tag = -1;
	int curind = y * sample_x + x;
	mlflow[0].previous_tag[curind] = mlflow[0].tag_matrix[curind];
	mlflow[0].tag_matrix[curind] = -1;
}


__global__ void assign_tag_kernel(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;

	// need to fix the following
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		mlflow[0].merge_detector[curind] &&
		mlflow[0].tag_matrix[curind] == -1
		)
	{
		int thisCellID = mlflow[0].tag_matrix[curind];
		for (int i = 1; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int ind_back = y1 * sample_x + x1;
			if ((x1 >= 0 && x1 <= sample_x - 1) &&
				(y1 >= 0 && y1 <= sample_y - 1)
				)
			{
				if (mlflow[0].tag_matrix[ind_back] > -1)
				{
					thisCellID = mlflow[0].tag_matrix[ind_back];
				}
			}
		}
		mlflow[0].previous_merge_tag[curind] = (thisCellID > 0) ? thisCellID : -1;
	}
	else
	{
		mlflow[0].merge_detector[curind] = 0;
	}
}



__global__ void reportLiquidToInterfaceConversion(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;

	// need to fix the following
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1) &&
		mlflow[0].merge_detector[curind]
		)
	{

		mlflow[0].merge_detector[curind] = 0;
		int thisCellID = -1;


		for (int i = 1; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;
			int ind_back = y1 * sample_x + x1;
			
			if ((x1 >= 0 && x1 <= sample_x - 1) &&
				(y1 >= 0 && y1 <= sample_y - 1)
				)
			{
				if (mlflow[0].tag_matrix[ind_back] > -1)
				{

					if (thisCellID < 0)
					{
						
						thisCellID = mlflow[0].tag_matrix[ind_back];
						mlflow[0].merge_record[curind + sample_num * (i-1)] = {-1, -1};
					}
					else
					{
						if (thisCellID != mlflow[0].tag_matrix[ind_back])
						{
							mlflow[0].merge_detector[curind] = 1;

							if (thisCellID < mlflow[0].tag_matrix[ind_back])
								mlflow[0].merge_record[curind+sample_num* (i - 1)] = { thisCellID, mlflow[0].tag_matrix[ind_back] };
							else
								mlflow[0].merge_record[curind + sample_num * (i - 1)] = { mlflow[0].tag_matrix[ind_back], thisCellID };
							mlflow[0].merge_flag = 1;

						}
						else
						{
							mlflow[0].merge_record[curind + sample_num * (i - 1)] = { -1, - 1 };
						}
					}
				}
			}
		}
	}
}

__global__ void update_merge_tag_kernel(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;

	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		if (mlflow[0].previous_merge_tag[curind] > 0 && mlflow[0].merge_detector[curind])

		{
			mlflow[0].tag_matrix[curind] = mlflow[0].previous_merge_tag[curind];
			mlflow[0].previous_merge_tag[curind] = -1;
		}
		else
		{
			mlflow[0].previous_merge_tag[curind] = -1;
		}
	}
}



__global__ void update_merge_tag(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int* renameVec, int renameVecSize)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		if (mlflow[0].tag_matrix[curind] > -1)
		{
			if (mlflow[0].tag_matrix[curind]-1<renameVecSize)
				mlflow[0].tag_matrix[curind] = renameVec[mlflow[0].tag_matrix[curind]-1]+1;
		}
		if (mlflow[0].split_tag_record[curind] > 0)
		{
			if (mlflow[0].split_tag_record[curind] - 1 < renameVecSize)
				mlflow[0].split_tag_record[curind] = renameVec[mlflow[0].split_tag_record[curind] - 1] + 1;
		}
		if (mlflow[0].tag_matrix[curind] > mlflow[0].bubble.bubble_count)
		{
			printf("remove tag change\n");
			mlflow[0].tag_matrix[curind] = -1;
			mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((mlflow[0].flag[curind] & ~TYPE_SU) | TYPE_F);
			const float rhon = mlflow[0].rho[curind];
			float massn = mlflow[0].mass[curind];
			float massexn = massn - rhon; // dump mass-rho difference into excess mass
			massn = rhon; // fluid cell mass has to equal rho
			float phin = 1.0f; // set phi[n] to 1.0f for fluid cells

			int counter = 0; // count (fluid|interface) neighbors
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;


				int ind_back = y1 * sample_x + x1;
				const unsigned char flagsji_su = mlflow[0].flag[ind_back] & (TYPE_SU | TYPE_S); // extract SURFACE flags
				counter += (int)(flagsji_su == TYPE_F || flagsji_su == TYPE_I || flagsji_su == TYPE_IF || flagsji_su == TYPE_GI); // avoid branching

			}
			massn += counter > 0 ? 0.0f : massexn; // if excess mass can't be distributed to neighboring interface or fluid cells, add it to local mass (ensure mass conservation)
			massexn = counter > 0 ? massexn / (float)counter : 0.0f; // divide excess mass up for all interface or fluid neighbors
			mlflow[0].mass[curind] = massn; // update mass
			mlflow[0].massex[curind] = massexn; // update excess mass
			mlflow[0].phi[curind] = phin; // update phi
		}
	}
}


__global__ void convertIntToUnsignedChar(mrFlow2D* mlflow, int sample_x, int sample_y) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
		const unsigned char flagsn_bo = mlflow[0].flag[curind] & TYPE_BO;
		if ((flagsn_sus == TYPE_G || flagsn_sus == TYPE_I)&&(flagsn_bo !=TYPE_S))
			mlflow[0].input_matrix[curind] = 255;
		else
			mlflow[0].input_matrix[curind] = 0;
	}

}

__global__ void parse_label(mrFlow2D* mlflow, MLFluidParam2D* param, int sample_x, int sample_y)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		if ((int)mlflow[0].label_matrix[curind] > 0)
		{

			int label = (int)mlflow[0].label_matrix[curind];
			atomicMax(&mlflow[0].bubble.label_num, (int)mlflow[0].label_matrix[curind]);
			atomicAdd(&mlflow[0].bubble.label_volume[label -1], 1.f - mlflow[0].phi[curind]);
			const unsigned char flagsji = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
			float type_flag = (float)(flagsji == TYPE_G);
			atomicAdd(&mlflow[0].bubble.pure_label_gas_volume[label - 1], type_flag);
		}
		mlflow[0].view_label_matrix[curind] = (int)mlflow[0].label_matrix[curind];
	}
}


__global__ void split_processing(mrFlow2D* mlflow, MLFluidParam2D* param, int2 * split_record, int sample_x, int sample_y)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		if (mlflow[0].split_detector[curind] == 1)
		{
			bool real_split = false;
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int ind_back = y1 * sample_x + x1;
				if ((x1 >= 0 && x1 <= sample_x - 1) &&
					(y1 >= 0 && y1 <= sample_y - 1))

					if (mlflow[0].tag_matrix[ind_back] == mlflow[0].split_tag_record[curind])
					{

						split_record[curind + (i - 1) *sample_x * sample_y] = { mlflow[0].tag_matrix[ind_back], (int)mlflow[0].label_matrix[ind_back] };
						real_split = true;
					}
					else
					{
						split_record[curind + (i - 1) * sample_x * sample_y] = { -1,-1 };
					}
				
			}
			if (!real_split)
			{
				mlflow[0].split_detector[curind] = 0;
				mlflow[0].split_tag_record[curind] = -1;
				for (int i = 1; i < 9; i++)
				{
					split_record[curind + (i - 1) * sample_x * sample_y] = { -1,-1 };
				}
			}

		}
		else
		{
			for (int i = 1; i < 9; i++)
			{
				split_record[curind + (i - 1) * sample_x * sample_y] = { -1,-1 };
			}
		}
	}

}


__global__ void ResetLabelVolume(mrFlow2D* mlflow)
{
	for (int i = 0; i < mlflow[0].bubble.label_num; i++)
	{
		mlflow[0].bubble.label_volume[i] = 0.f;
		mlflow[0].bubble.pure_label_gas_volume[i] = 0.f;
	}
	mlflow[0].bubble.label_num = 0;
}

__global__ void bubble_create(mrFlow2D* mlflow, int2* d_ptr, int d_ptr_size)
{

	for (int i = 0; i < d_ptr_size; i++)
	{
		int bubble_0 = d_ptr[i].x;
		int new_label = d_ptr[i].y;
		int curr_num = mlflow[0].bubble.bubble_count;	
		mlflow[0].bubble.init_volume[curr_num] = mlflow[0].bubble.label_volume[new_label-1] * mlflow[0].bubble.rho[bubble_0 - 1];
		mlflow[0].bubble.volume[curr_num] = mlflow[0].bubble.label_volume[new_label-1];

		
		mlflow[0].bubble.pure_gas_volume[curr_num] = mlflow[0].bubble.pure_label_gas_volume[new_label - 1];
		if (mlflow[0].bubble.pure_gas_volume[curr_num] < 0.f)
		{
			mlflow[0].bubble.freeze[curr_num] = 0;
			mlflow[0].bubble.rho[curr_num] = 1.f;
		}
		else
			{
				mlflow[0].bubble.freeze[curr_num] = 1;
				mlflow[0].bubble.rho[curr_num] = mlflow[0].bubble.rho[bubble_0 - 1];
			}


		mlflow[0].bubble.bubble_count++;
	}
}

__global__ void update_split_tag(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int* label_renameVec_)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{

		int label = (int)mlflow[0].label_matrix[curind];
		if (label > 0)
		{
			if (label_renameVec_[label - 1] > -1)
			{
				mlflow[0].tag_matrix[curind] = label_renameVec_[label - 1] + 1;

			}
		}
		else
		{
			mlflow[0].tag_matrix[curind] = -1;
		}
	}
}

__global__ void bubble_volume_update_kernel(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
		if (mlflow[0].delta_phi[curind]!= 0)
		{
			int tag = mlflow[0].tag_matrix[curind];
			if (tag <= 0)
			{
				tag = mlflow[0].previous_tag[curind];
				mlflow[0].previous_tag[curind] = -1;

			}
			atomicAdd(&mlflow[0].bubble.volume[tag - 1], -mlflow[0].delta_phi[curind]);
			mlflow[0].delta_phi[curind] = 0;
		}
	}
}

__global__ void bubble_volume_g_update_kernel(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S);
		
		float factor = 0.85f;
		if (time < 400) {
			factor =  0.85;
		} else if (time <= 600) {
			factor = 0.85 / (1.f + expf(-0.05 * ((float)time - 500.f))); // Sigmoid 函数
		} else {
			factor = 0.0;
		}

		if (mlflow[0].delta_g[curind] != 0)
		{
			int tag = mlflow[0].tag_matrix[curind];
			if (mlflow[0].flag[curind]  == TYPE_I)
			{
				{
					atomicAdd(&mlflow[0].bubble.init_volume[tag - 1], 1.f/3.f *factor * mlflow[0].delta_g[curind] * mlflow[0].rho[curind]);
				}
			}
			mlflow[0].delta_g[curind] = 0;	
		}
		
	}
}


__global__ void clear_detector(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		mlflow[0].split_detector[curind] = 0;
		mlflow[0].split_tag_record[curind] = -1;
		mlflow[0].merge_detector[curind] = 0;
	}
	if (curind == 1)
	{
		mlflow[0].split_flag = 0;
		mlflow[0].merge_flag = 0;
		mlflow[0].split_record_length = 0;
		mlflow[0].merge_record_length = 0;
	}
}


__host__ void registerMerge(int b0, int b1, int* renameVec_)
{
	// identical bubbles can not be merged
	if (b0 == b1) { return; }

	// ensure that b0 < b1 (increasing ID ordering is required for some functions later on)
	if (b1 < b0) { swap(b0, b1); }

	// if bubble b1 is also marked for merging with another bubble, e.g., bubble b2
	if (isRenamed(renameVec_, b1))
	{
		// mark bubble b2 for merging with b0 (b2 = renameVec_[b1])
		registerMerge(b0, renameVec_[b1], renameVec_);
		// mark bubble b1 for merging with bubble b0 or bubble b2 (depending on the ID order found in
		// registerMerge(b0,b2) above)
		renameVec_[b1] = renameVec_[renameVec_[b1]];
	}
	else
	{
		// mark bubble b1 for merging with bubble b0
		renameVec_[b1] = b0;
	}
}

__host__ void resolve_name_merge(int* renameVec_, int renameVec_size)
{


	for (int i = renameVec_size - 1; i > 0; --i)
	{
		int id = renameVec_[i];

		while (renameVec_[id] != id)
		{
			id = renameVec_[id];
		}
		renameVec_[i] = id;
	}
}



void resolveParaTransitiveRenames(thrust::device_vector<int2> d_vec, int* renameVec, int renameVec_size)
{


	thrust::sort(d_vec.begin(), d_vec.end());
	thrust::device_vector<int2>::iterator new_end;
	new_end = thrust::unique(d_vec.begin(), d_vec.end());
	d_vec.erase(new_end, d_vec.end());
	d_vec.erase(d_vec.begin());

	thrust::host_vector<int2> h_vec(d_vec);

	for (int i = 0; i < h_vec.size(); i++)
	{
		int bubble_0 = h_vec[i].x - 1;
		int bubble_1 = h_vec[i].y - 1;
		// printf("merge bubble %d %d \n", bubble_0, bubble_1);
		registerMerge(bubble_0, bubble_1, renameVec);
	}

	resolve_name_merge(renameVec, renameVec_size);
}

__global__ void getBubbleCountKernel(mrFlow2D* d_mlflow, int* d_bubble_count) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_bubble_count = d_mlflow[0].bubble.bubble_count;
	}
}

__global__ void PushBubbleCountKernel(mrFlow2D* d_mlflow, int* d_bubble_count) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_mlflow[0].bubble.bubble_count = *d_bubble_count;
	}
}

__global__ void getBubbleLabelNum(mrFlow2D* d_mlflow, int* d_bubble_count) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_bubble_count = d_mlflow[0].bubble.label_num;
	}
}

__global__ void reduceBubbleLabelNum(mrFlow2D* d_mlflow) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_mlflow[0].bubble.bubble_count--;
	}
}



void resolveSplitBubble(mrFlow2D* mlflow, MLFluidParam2D* param, thrust::device_vector<int2> d_vec)
{
	thrust::sort(d_vec.begin(), d_vec.end());
	thrust::device_vector<int2>::iterator new_end;
	new_end = thrust::unique(d_vec.begin(), d_vec.end());
	d_vec.erase(new_end, d_vec.end());
	d_vec.erase(d_vec.begin());

	int2* d_ptr = thrust::raw_pointer_cast(d_vec.data());
	// get the number of previous bubbles
	int* d_bubble_count;
	int renameVec_size;
	cudaMalloc(&d_bubble_count, sizeof(int));
	getBubbleCountKernel << <1, 1 >> > (mlflow, d_bubble_count);
	cudaDeviceSynchronize();
	cudaMemcpy(&renameVec_size, d_bubble_count, sizeof(int), cudaMemcpyDeviceToHost);
	// get the detected bubbles
	int* d_label_num;
	int label_num;
	cudaMalloc(&d_label_num, sizeof(int));
	getBubbleLabelNum << <1, 1 >> > (mlflow, d_label_num);
	cudaDeviceSynchronize();
	cudaMemcpy(&label_num, d_label_num, sizeof(int), cudaMemcpyDeviceToHost);

	int* h_renameVec_ = (int*)malloc(renameVec_size * sizeof(int));
	thrust::host_vector<int> new_bubbles;

	for (int i = 0; i < renameVec_size; i++)
	{
		h_renameVec_[i] = -1;
		new_bubbles.push_back(i);
	}

	int* label_renameVec_ = (int*)malloc(label_num * sizeof(int));

	for (int i = 0; i < label_num; i++)
		label_renameVec_[i] = -1;

	int bubble_count_rec = renameVec_size;


	std::vector<int2> h_ptr(d_vec.size());
	thrust::copy(d_vec.begin(), d_vec.end(), h_ptr.begin());


	bubble_create << <1, 1 >> >
		(
			mlflow, d_ptr, d_vec.size()
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	for (int i = 0; i < h_ptr.size(); i++)
	{
		int bubble_0 = h_ptr[i].x - 1;
		int bubble_1 = h_ptr[i].y - 1;

		label_renameVec_[bubble_1] = bubble_count_rec;
		bubble_count_rec++;
		h_renameVec_[bubble_0] = 1;
		new_bubbles.push_back(bubble_1);

	}

	int bubble_count_rec_2 = bubble_count_rec;

	for (int i = 0; i < renameVec_size; i++)
	{
		if (h_renameVec_[i] > 0)
		{

			label_renameVec_[new_bubbles[bubble_count_rec_2 - 1]] = i;
			bubble_swap << <1, 1 >> >
				(mlflow, i, bubble_count_rec_2 - 1);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			reduceBubbleLabelNum << <1, 1 >> > (mlflow);
			cudaDeviceSynchronize();
			bubble_count_rec_2--;
		}
	}

	assert(label_num == bubble_count_rec_2);

	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);

	int* d_label_renameVec;
	cudaMalloc(&d_label_renameVec, label_num * sizeof(int));

	cudaMemcpy(d_label_renameVec, label_renameVec_, label_num * sizeof(int), cudaMemcpyHostToDevice);

	update_split_tag << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_num, d_label_renameVec);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}


__global__ void PushLabelNumKernel(mrFlow2D* d_mlflow, int *d_bubble_count) {

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_mlflow[0].bubble.label_num = *d_bubble_count;
	}
}



void mergeAndReorderBubbleVector(mrFlow2D* mlflow, MLFluidParam2D* param, thrust::device_vector<int2> d_vec)
{

	int* d_bubble_count;
	int renameVec_size;
	cudaMalloc(&d_bubble_count, sizeof(int));
	getBubbleCountKernel << <1, 1 >> > (mlflow, d_bubble_count);
	cudaDeviceSynchronize();
	cudaMemcpy(&renameVec_size, d_bubble_count, sizeof(int), cudaMemcpyDeviceToHost);

	if (renameVec_size < 2) return;
	int* renameVec_ = (int*)malloc(renameVec_size * sizeof(int));
	for (int i = 0; i < renameVec_size; i++) {
		renameVec_[i] = i;
	}

	
	resolveParaTransitiveRenames(d_vec, renameVec_, renameVec_size);


	for (int i = 0; i < renameVec_size; ++i)
	{
	

		if (renameVec_[i] != i)
		{

			bubble_merge << <1, 1 >> >
				(
					mlflow, renameVec_[i], i
					);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}
	}

	thrust::host_vector<int> exchangeVector(renameVec_size);
	for (size_t i = 0; i < exchangeVector.size(); ++i)
	{
		exchangeVector[i] = i;
	}


	int gapPointer = 1;
	while (gapPointer < renameVec_size && !isRenamed(renameVec_, gapPointer)) // find first renamed bubble
	{
		++gapPointer;
	}

	
	int lastPointer = renameVec_size - 1;

	while (isRenamed(renameVec_, lastPointer) && lastPointer > 0) // find last non-renamed bubble
	{
		--lastPointer;
	}

	while (lastPointer > gapPointer)
	{
		
		bubble_swap << <1, 1 >> >
			(
				mlflow, gapPointer, lastPointer
				);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


		exchangeVector[lastPointer] = gapPointer;
		exchangeVector[gapPointer] = lastPointer;

		do
		{
			--lastPointer;
		} while (isRenamed(renameVec_, lastPointer) && lastPointer > gapPointer);

		do
		{
			++gapPointer;
		} while (gapPointer < renameVec_size && !isRenamed(renameVec_, gapPointer));
	}

	int newSize = lastPointer + 1;

	int h_tmp = newSize;
	int* d_tmp;

	cudaMalloc(&d_tmp, sizeof(int));

	cudaMemcpy(d_tmp, &h_tmp, sizeof(int), cudaMemcpyHostToDevice);

	
	PushBubbleCountKernel << <1, 1 >> > (mlflow, d_tmp);
	cudaDeviceSynchronize();

	for (int i = 0; i < renameVec_size; ++i)
	{
		renameVec_[i] = exchangeVector[renameVec_[i]];
	}

	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);

	int* d_label_renameVec;
	cudaMalloc(&d_label_renameVec, renameVec_size * sizeof(int));
	cudaMemcpy(d_label_renameVec, renameVec_, renameVec_size * sizeof(int), cudaMemcpyHostToDevice);
	update_merge_tag << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_num, d_label_renameVec, renameVec_size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}



__global__ void getSplitRecordLength(mrFlow2D* d_mlflow, int* d_bubble_count) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_bubble_count = d_mlflow[0].split_record_length;
	}
}

__global__ void print_label(mrFlow2D* d_mlflow)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < d_mlflow[0].bubble.label_num; i++)
			printf("label %d volume %f\n", i, d_mlflow[0].bubble.label_volume[i]);
	}
}

__global__ void print_bubble(mrFlow2D* d_mlflow)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < d_mlflow[0].bubble.bubble_count; i++)
			printf("bubble %d volume %f init volume %f bubble rho %f\n", i, d_mlflow[0].bubble.volume[i], d_mlflow[0].bubble.init_volume[i], d_mlflow[0].bubble.rho[i]);
	}
}

void handleSplits(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);

	convertIntToUnsignedChar << <grid1, threads1 >> > (mlflow, sample_x, sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	connectedComponentLabeling(mlflow, (size_t)sample_x, (size_t)sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	int h_tmp = 0;
	int* d_tmp;
	cudaMalloc(&d_tmp, sizeof(int));
	cudaMemcpy(d_tmp, &h_tmp, sizeof(int), cudaMemcpyHostToDevice);

	PushLabelNumKernel << <1, 1 >> > (mlflow, d_tmp);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	parse_label << <grid1, threads1 >> > (mlflow, param, sample_x, sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::device_vector<int2> d_vec(sample_num * 8);
	split_processing << <grid1, threads1 >> >
		(
			mlflow, param, thrust::raw_pointer_cast(d_vec.data()),
			sample_x, sample_y
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	resolveSplitBubble(mlflow, param, d_vec);
	ResetLabelVolume << <1, 1 >> > (mlflow);
	cudaDeviceSynchronize();

}


void ClearDectector(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);

	clear_detector << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void bubble_volume_update(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;

	int sample_num = sample_x * sample_y;

	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	bubble_volume_update_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void getBubbleVolumeKernel(mrFlow2D* d_mlflow, float* d_ptr) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i=0;i<d_mlflow[0].bubble.bubble_count;i++)
			d_ptr[i] = d_mlflow[0].bubble.volume[i];
	}
}

__global__ void getBubbleInitVolumeKernel(mrFlow2D* d_mlflow, float* d_ptr) {

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i=0;i<d_mlflow[0].bubble.bubble_count;i++)
			d_ptr[i] = d_mlflow[0].bubble.init_volume[i];
	}
}


__host__ bool islow(std::vector<float> renameVec, int id) {
	return renameVec[id] < 1.f;
}

__global__ void bubble_remove_last(mrFlow2D* mlflow, int sample_x)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y  + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (curind < mlflow[0].bubble.max_bubble_count)
	{
		if (curind > mlflow[0].bubble.bubble_count - 1)
		{
			mlflow[0].bubble.init_volume[curind] = 0;
			mlflow[0].bubble.volume[curind] = 0;
			mlflow[0].bubble.rho[curind] = 0;
			mlflow[0].bubble.pure_gas_volume[curind] = 0;
			mlflow[0].bubble.freeze[curind] = 0;
		}
	}

}

void bubble_remove(mrFlow2D* mlflow, MLFluidParam2D* param)
{

	int* d_bubble_count;
	int renameVec_size;
	cudaMalloc(&d_bubble_count, sizeof(int));
	getBubbleCountKernel << <1, 1 >> > (mlflow, d_bubble_count);
	cudaDeviceSynchronize();
	cudaMemcpy(&renameVec_size, d_bubble_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int* renameVec_ = (int*)malloc(renameVec_size * sizeof(int));
	for (int i = 0; i < renameVec_size; i++) {
		renameVec_[i] = i;
	}

	thrust::device_vector<float> d_volume(renameVec_size);

	getBubbleVolumeKernel << <1, 1 >> > (mlflow, thrust::raw_pointer_cast(d_volume.data()));
	cudaDeviceSynchronize(); 
	std::vector<float> h_volume(renameVec_size);

	thrust::copy(d_volume.begin(), d_volume.end(), h_volume.begin());


	thrust::host_vector<int> exchangeVector(renameVec_size);
	for (size_t i = 0; i < exchangeVector.size(); ++i)
	{
		exchangeVector[i] = i;

	}

	int gapPointer = 0;

	while (gapPointer < renameVec_size && !islow(h_volume, gapPointer)) // find first renamed bubble
	{
		++gapPointer;
	}

	int lastPointer = renameVec_size - 1;

	while (islow(h_volume, lastPointer) && lastPointer > 0) // find last non-renamed bubble
	{
		--lastPointer;
	}

	while (lastPointer > gapPointer)
	{

		bubble_swap << <1, 1 >> >
			(
				mlflow, gapPointer, lastPointer
				);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		
		exchangeVector[lastPointer] = gapPointer;
		exchangeVector[gapPointer] = lastPointer;

		do
		{
			--lastPointer;
		} while (islow(h_volume, lastPointer) && lastPointer > gapPointer);

		do
		{
			++gapPointer;
		} while (gapPointer < renameVec_size && !islow(h_volume, gapPointer));
	}

	int newSize = lastPointer + 1;

	int h_tmp = newSize;
	int* d_tmp;
	
	cudaMalloc(&d_tmp, sizeof(int));
	cudaMemcpy(d_tmp, &h_tmp, sizeof(int), cudaMemcpyHostToDevice);


	PushBubbleCountKernel << <1, 1 >> > (mlflow, d_tmp);
	cudaDeviceSynchronize();
	
	for (int i = 0; i < renameVec_size; ++i)
	{
		renameVec_[i] = exchangeVector[renameVec_[i]];
	}

	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	// need to set renameVec_ to gpus!!!

	int* d_label_renameVec;
	cudaMalloc(&d_label_renameVec, renameVec_size * sizeof(int));
	cudaMemcpy(d_label_renameVec, renameVec_, renameVec_size * sizeof(int), cudaMemcpyHostToDevice);
	
	update_merge_tag << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_num, d_label_renameVec, renameVec_size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	bubble_remove_last << <grid1, threads1 >> > (mlflow, sample_x);
	checkCudaErrors(cudaDeviceSynchronize());

}


void bubble_remove_initial(mrFlow2D* mlflow, MLFluidParam2D* param)
{

	int* d_bubble_count;
	int renameVec_size;
	cudaMalloc(&d_bubble_count, sizeof(int));
	getBubbleCountKernel << <1, 1 >> > (mlflow, d_bubble_count);
	cudaDeviceSynchronize();
	cudaMemcpy(&renameVec_size, d_bubble_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int* renameVec_ = (int*)malloc(renameVec_size * sizeof(int));
	for (int i = 0; i < renameVec_size; i++) {
		renameVec_[i] = i;
	}

	thrust::device_vector<float> d_volume(renameVec_size);
	getBubbleInitVolumeKernel << <1, 1 >> > (mlflow, thrust::raw_pointer_cast(d_volume.data()));
	cudaDeviceSynchronize(); 
	std::vector<float> h_volume(renameVec_size);

	thrust::copy(d_volume.begin(), d_volume.end(), h_volume.begin());


	thrust::host_vector<int> exchangeVector(renameVec_size);
	for (size_t i = 0; i < exchangeVector.size(); ++i)
	{
		exchangeVector[i] = i;
	}

	int gapPointer = 0;
	while (gapPointer < renameVec_size && !islow(h_volume, gapPointer)) // find first renamed bubble
	{
		++gapPointer;
	}


	int lastPointer = renameVec_size - 1;

	while (islow(h_volume, lastPointer) && lastPointer > 0) // find last non-renamed bubble
	{
		--lastPointer;
	}

	while (lastPointer > gapPointer)
	{
		bubble_swap << <1, 1 >> >
			(
				mlflow, gapPointer, lastPointer
				);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		exchangeVector[lastPointer] = gapPointer;
		exchangeVector[gapPointer] = lastPointer;

		do
		{
			--lastPointer;
		} while (islow(h_volume, lastPointer) && lastPointer > gapPointer);
		do
		{
			++gapPointer;
		} while (gapPointer < renameVec_size && !islow(h_volume, gapPointer));
	}

	int newSize = lastPointer + 1;

	int h_tmp = newSize;
	int* d_tmp;

	cudaMalloc(&d_tmp, sizeof(int));
	cudaMemcpy(d_tmp, &h_tmp, sizeof(int), cudaMemcpyHostToDevice);

	PushBubbleCountKernel << <1, 1 >> > (mlflow, d_tmp);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	for (int i = 0; i < renameVec_size; ++i)
	{
		renameVec_[i] = exchangeVector[renameVec_[i]];
	}

	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	

	int* d_label_renameVec;
	cudaMalloc(&d_label_renameVec, renameVec_size * sizeof(int));
	cudaMemcpy(d_label_renameVec, renameVec_, renameVec_size * sizeof(int), cudaMemcpyHostToDevice);

	update_merge_tag << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_num, d_label_renameVec, renameVec_size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	bubble_remove_last << <grid1, threads1 >> > (mlflow, sample_x);
	checkCudaErrors(cudaDeviceSynchronize());

}


__global__ void bubble_rho_update_kernel(mrFlow2D* mlflow)
{
	for (int i = 0; i < mlflow[0].bubble.bubble_count; i++)
	{	
		mlflow[0].bubble.rho[i] = mlflow[0].bubble.init_volume[i] / mlflow[0].bubble.volume[i];
	}

}


__global__ void ResetMergeKernel(mrFlow2D* mlflow) {

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		mlflow[0].merge_record_length = 0;
		mlflow[0].merge_flag = 0;
	}
}

__global__ void ResetSplitKernel(mrFlow2D* mlflow) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		mlflow[0].split_record_length = 0;
		mlflow[0].split_flag = 0;
	}
}

__global__ void MergeSplitDetectorKernel(mrFlow2D* mlflow,int * merge_flag, int * split_flag) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*merge_flag = mlflow[0].merge_flag;
		*split_flag = mlflow[0].split_flag;
	}
}

__global__ void getMergeRecordLength(mrFlow2D* mlflow, int* merge_record_length) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*merge_record_length = mlflow[0].merge_record_length;
	}
}

__global__ void copyMergeRecordKernel(mrFlow2D* mlflow, int2* d_vec, int sample_x, int sample_y)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		for (int z = 0; z < 8; z++)
		{
			if (mlflow[0].merge_detector[curind] > 0)
			{
				d_vec[curind+sample_x*sample_y*z] = mlflow[0].merge_record[curind + sample_x * sample_y * z];
				mlflow[0].merge_record[curind + sample_x * sample_y * z] = { -1,-1 };
			}
			else
			{
				d_vec[curind + sample_x * sample_y * z] = { -1,-1 };
				mlflow[0].merge_record[curind + sample_x * sample_y * z] = { -1,-1 };
			}
		}
	}
}



void update_bubble(mrFlow2D* mlflow, MLFluidParam2D* param)
{

	bubble_volume_update(mlflow, param);
	bubble_rho_update_kernel << <1, 1 >> > (mlflow);
	cudaDeviceSynchronize();

	int* d_merge_flag;
	int merge_flag;
	int* d_split_flag;
	int split_flag;

	cudaMalloc(&d_merge_flag, sizeof(int));
	cudaMalloc(&d_split_flag, sizeof(int));

	MergeSplitDetectorKernel << <1, 1 >> > (mlflow, d_merge_flag, d_split_flag);

	cudaDeviceSynchronize();
	cudaMemcpy(&merge_flag, d_merge_flag, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&split_flag, d_split_flag, sizeof(int), cudaMemcpyDeviceToHost);
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;

	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);

	if (merge_flag>0)
	{
		thrust::device_vector<int2> d_vec(sample_num*8);

		copyMergeRecordKernel << <grid1, threads1 >> > (mlflow, thrust::raw_pointer_cast(d_vec.data()),sample_x,sample_y);
		//cudaDeviceSynchronize();
		cudaDeviceSynchronize();

		mergeAndReorderBubbleVector(mlflow, param, d_vec);
		ResetMergeKernel << <1, 1 >> > (mlflow);
		cudaDeviceSynchronize();
	}

	bubble_remove(mlflow, param);

	if (split_flag > 0)
	{
		handleSplits(mlflow, param);
		ResetSplitKernel << <1, 1 >> > (mlflow);
		cudaDeviceSynchronize();
		bubble_remove_last << <grid1, threads1 >> > (mlflow, sample_x);
	}

	bubble_remove(mlflow, param);
	bubble_remove_last << <grid1, threads1 >> > (mlflow, sample_x);
	checkCudaErrors(cudaDeviceSynchronize());


	if (merge_flag>0||split_flag>0)
		ClearDectector(mlflow, param);
}


__global__ void surface_1(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S); // extract SURFACE flags
		if (flagsn_sus == TYPE_IF)
		{
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int dz = int(ez2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;
				const unsigned char flagsji = mlflow[0].flag[ind_back];
				const unsigned char flagsji_su = flagsji & (TYPE_SU | TYPE_S); // extract SURFACE flags
				const unsigned char flagsji_r = flagsji & ~TYPE_SU; // extract all non-SURFACE flags
				if (flagsji_su == TYPE_IG) mlflow[0].flag[ind_back] = (MLLATTICENODE_SURFACE_FLAG)(flagsji_r | TYPE_I); // prevent interface neighbor cells from becoming gas
				else if (flagsji_su == TYPE_G) mlflow[0].flag[ind_back] = (MLLATTICENODE_SURFACE_FLAG)(flagsji_r | TYPE_GI); // neighbor cell was gas and must change to interface

			}
		}
	}
}


__global__ void surface_2(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S); // extract SURFACE flags
		mrUtilFuncGpu2D mrutilfunc;
		if (flagsn_sus == TYPE_GI) { // initialize the fi of gas cells that should become interface
			float rhon, uxn, uyn, uzn; // average over all fluid/interface neighbors
			float rhot = 0.0f, uxt = 0.0f, uyt = 0.0f, uzt = 0.0f, counter = 0.0f; // average over all fluid/interface neighbors
			float rho_gt = 0, c_k = 0.f;
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;
				const unsigned char flagsji_sus = mlflow[0].flag[ind_back] & (TYPE_SU | TYPE_S); // extract SURFACE flags

				if (flagsji_sus == TYPE_F || flagsji_sus == TYPE_I || flagsji_sus == TYPE_IF) { // fluid or interface or (interface->fluid) neighbor
					counter += 1.0f;
					rhot += mlflow[0].rho[ind_back];
					uxt += mlflow[0].u[ind_back].x;
					uyt += mlflow[0].u[ind_back].y;
					uzt += mlflow[0].u[ind_back].z;
					if (i < 5)
					{
						rho_gt += mlflow[0].c_value[ind_back];
						c_k += 1.0f;
					}
				}
			}
			rhon = counter > 0.0f ? rhot / counter : 1.0f;
			uxn = counter > 0.0f ? uxt / counter : 0.0f;
			uyn = counter > 0.0f ? uyt / counter : 0.0f;
			uzn = counter > 0.0f ? uzt / counter : 0.0f;

			rho_gt = c_k > 0.0f ? rho_gt / c_k : 0.0f;

			float feq[9];

			float3 u_2{ uxn,uyn,uzn };
			u_2 = normalizing_clamp(u_2, 0.4);
			uxn = u_2.x;
			uyn = u_2.y;
			uzn = u_2.z;

			mrutilfunc.calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs

			for (int i = 0; i < 9; i++)
			{
				feq[i] += w2d_gpu[i];
			}
		

			REAL invRho = 1 / rhon;
			REAL pixx = feq[1] + feq[2] + feq[5] + feq[6] + feq[7] + feq[8];
			REAL piyy = feq[3] + feq[4] + feq[5] + feq[6] + feq[7] + feq[8];
			REAL pixy = feq[5] + feq[6] - feq[7] - feq[8];
			pixx = 1 * (pixx * invRho - cs2);
			piyy = 1 * (piyy * invRho - cs2);
			pixy = 1 * (pixy * invRho);
			mlflow[0].fMomPost[curind * 6 + 0] = rhon;
			mlflow[0].fMomPost[curind * 6 + 1] = uxn;
			mlflow[0].fMomPost[curind * 6 + 2] = uyn;
			mlflow[0].fMomPost[curind * 6 + 3] = pixx;
			mlflow[0].fMomPost[curind * 6 + 4] = piyy;
			mlflow[0].fMomPost[curind * 6 + 5] = pixy;

			mlflow[0].rho[curind] = rhon;
			mlflow[0].u[curind].x = uxn;
			mlflow[0].u[curind].y = uyn;
			mlflow[0].c_value[curind] = rho_gt;

			float geq[5];
			mrutilfunc.calculate_g_eq(rho_gt, uxn, uyn, uzn, geq); // calculate equilibrium DDFs
			for (int i = 0; i < 5; i++)
				mlflow[0].gMom[curind + i * sample_num] = geq[i];

		}
		else if (flagsn_sus == TYPE_IG) { // flag interface->gas is set
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);

				int x1 = x - dx;
				int y1 = y - dy;


				int ind_back = y1 * sample_x + x1;

				const unsigned char flagsji = mlflow[0].flag[ind_back];

				const unsigned char flagsji_su = flagsji & (TYPE_SU | TYPE_S); // extract SURFACE flags
				const unsigned char flagsji_r = flagsji & (~TYPE_SU); // extract all non-SURFACE flags

				if (flagsji_su == TYPE_F || flagsji_su == TYPE_IF) {

					mlflow[0].flag[ind_back] = (MLLATTICENODE_SURFACE_FLAG)(flagsji_r | TYPE_I); // prevent fluid or interface neighbors that turn to fluid from being/becoming fluid
					mlflow[0].merge_detector[ind_back] = 1;
				}

			}

		}
	}
}

__global__ void surface_3(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	mrUtilFuncGpu2D mrutilfunc;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn_sus = mlflow[0].flag[curind] & (TYPE_SU | TYPE_S); // extract SURFACE flags
		if (flagsn_sus & TYPE_S) return;
		const float rhon = mlflow[0].rho[curind]; // density of cell n
		float massn = mlflow[0].mass[curind]; // mass of cell n
		float massexn = 0.0f; // excess mass of cell n
		float phin = 0.0f;
		if (flagsn_sus == TYPE_F) { // regular fluid cell
			massexn = massn - rhon; // dump mass-rho difference into excess mass
			massn = rhon; // fluid cell mass has to equal rho
			phin = 1.0f;

			mlflow[0].previous_tag[curind] = mlflow[0].tag_matrix[curind];
			mlflow[0].tag_matrix[curind] = -1;
		}
		else if (flagsn_sus == TYPE_I) { // regular interface cell
			massexn = massn > rhon ? massn - rhon : massn < 0.0f ? massn : 0.0f; // allow interface cells with mass>rho or mass<0
			massn = clamp(massn, 0.0f, rhon);
			phin = mrutilfunc.calculate_phi(rhon, massn, TYPE_I); // calculate fill level for next step (only necessary for interface cells)
			// printf("phin %f", phin);
		}
		else if (flagsn_sus == TYPE_G) { // regular gas cell
			massexn = massn; // dump remaining mass into excess mass
			massn = 0.0f;
			phin = 0.0f;
		}
		else if (flagsn_sus == TYPE_IF) { // flag interface->fluid is set
			mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((mlflow[0].flag[curind] & ~TYPE_SU) | TYPE_F); // cell becomes fluid

			// we can not do it here, we need to do it in the other next step
			reportInterfaceToLiquidConversion(mlflow, x, y, sample_x, sample_y); // report interface->fluid conversion

			massexn = massn - rhon; // dump mass-rho difference into excess mass
			massn = rhon; // fluid cell mass has to equal rho
			phin = 1.0f; // set phi[n] to 1.0f for fluid cells
		}
		else if (flagsn_sus == TYPE_IG) { // flag interface->gas is set
			mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((mlflow[0].flag[curind] & ~TYPE_SU) | TYPE_G); // cell becomes gas
			massexn = massn; // dump remaining mass into excess mass
			massn = 0.0f; // gas mass has to be zero
			phin = 0.0f; // set phi[n] to 0.0f for gas cells
			
			int tag = mlflow[0].tag_matrix[curind];
			atomicAdd(&mlflow[0].bubble.pure_label_gas_volume[tag - 1], 1.f);
		}
		else if (flagsn_sus == TYPE_GI) { // flag gas->interface is set
			mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((mlflow[0].flag[curind] & ~TYPE_SU) | TYPE_I); // cell becomes interface
			massexn = massn > rhon ? massn - rhon : massn < 0.0f ? massn : 0.0f; // allow interface cells with mass>rho or mass<0
			massn = clamp(massn, 0.0f, rhon);
			phin = mrutilfunc.calculate_phi(rhon, massn, TYPE_I); // calculate fill level for next step (only necessary for interface cells)
		
			int tag = mlflow[0].tag_matrix[curind];
			atomicAdd(&mlflow[0].bubble.pure_label_gas_volume[tag - 1], -1.f);
		
		}
		int counter = 0; // count (fluid|interface) neighbors
		for (int i = 1; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;


			int ind_back = y1 * sample_x + x1;
			const unsigned char flagsji_su = mlflow[0].flag[ind_back] & (TYPE_SU | TYPE_S); // extract SURFACE flags
			counter += (int)(flagsji_su == TYPE_F || flagsji_su == TYPE_I || flagsji_su == TYPE_IF || flagsji_su == TYPE_GI); // avoid branching

		}
		massn += counter > 0 ? 0.0f : massexn; // if excess mass can't be distributed to neighboring interface or fluid cells, add it to local mass (ensure mass conservation)
		massexn = counter > 0 ? massexn / (float)counter : 0.0f; // divide excess mass up for all interface or fluid neighbors

		mlflow[0].mass[curind] = massn; // update mass
		mlflow[0].massex[curind] = massexn; // update excess mass

		mlflow[0].delta_phi[curind] = phin - mlflow[0].phi[curind]; // calculate phi difference for next step

		// reconstruct true phi
		if ((mlflow[0].flag[curind] & (TYPE_SU | TYPE_S))==TYPE_I)
		{
			float rhon_g = 0.f;
			for (int i = 0; i < 5; i++)
				rhon_g += mlflow[0].gMom[curind + i * sample_num];
			mlflow[0].delta_g[curind] -= rhon_g * (mlflow[0].delta_phi[curind]);
		}

		mlflow[0].phi[curind] = phin; // update phi

	}
}



__global__ void ResetMassexn(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;

	mlflow[0].mass_surplus[curind] = 0.f;
	mlflow[0].mass_deficit[curind] = 0.f;
	mlflow[0].massex[curind] = 0.0f;
	mlflow[0].fMomViewer[sample_num + curind] = 0.f;

}


__global__ void calculate_disjoint(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	RigidFuncGpu2D mrrigidbody;
	
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		// mlflow[0].fMomViewer[curind + sample_num] = 0.f;
		mlflow[0].fMomViewer[sample_num * 2 + curind] = 0.f;
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_su == TYPE_I)
		{
		float massn = mlflow[0].mass[curind];
		float phij[9]; // cache fill level of neighbor lattice points
		for (int i = 1; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			int ind_back = y1 * sample_x + x1;
			massn += mlflow[0].massex[ind_back]; // distribute excess mass from last step which is stored in neighbors
		}
		for (int i = 1; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);

			int x1 = x - dx;
			int y1 = y - dy;

			int ind_back = y1 * sample_x + x1;
			if (mlflow[0].flag[ind_back] & TYPE_SU == TYPE_G)
				phij[i] = 0.f;
			else
				phij[i] = mlflow[0].phi[ind_back]; // cache fill level of neighbor lattice points

		}
		float rhon = 0.0f; 
		rhon = mlflow[0].fMom[curind * 6 + 0];
		phij[0] = mrutilfunc.calculate_phi(rhon, massn, flagsn); // don't load phi[n] from memory, instead recalculate it with mass corrected by excess mass
		int tag_curind = mlflow[0].tag_matrix[curind] - 1;
		float3 normal = mrutilfunc.calculate_normal(phij);
		float disjoint = 0.f;
		int max_ids = -1;
		//printf("normal %f %f\n",normal.x,normal.y);

		for (int jk = 1; jk < 16; jk++)
		{
			int x12 = round((float)x - (float) jk * 0.2f * normal.x);
			int y12 = round((float)y - (float) jk * 0.2f * normal.y);

			if (x12 >= 0 && x12 < sample_x && y12 >= 0 && y12 < sample_y)
			{
				int ind_back = y12 * sample_x + x12;
				if (mlflow[0].tag_matrix[ind_back] > 0)
				{
					int tag_neighbor = mlflow[0].tag_matrix[ind_back] - 1;
					if (tag_curind!=tag_neighbor && mlflow[0].flag[ind_back] == TYPE_I)
					{
						float center_offset = mrutilfunc.plic_cube(phij[0], normal);
						float alpha = mlflow[0].phi[ind_back]; 
						float dis = abs((float) jk * 0.2f * normal.x);
						//float d = fmaxf(dis * center_offset/0.5f - center_offset,0.f);
						float d = abs(dis/(normal.x+1e-8));
						//printf("ind_back %d tag_neighbor %d d %f\n",ind_back,tag_neighbor,d);
						if (disjoint< 1.f - d/3.f)
						{
							disjoint = 1.f - d/3.f;
							max_ids = tag_neighbor;
						}
					}
				}
			}
		}

		if (disjoint>0)
		{
		atomicAdd(&mlflow[0].fMomViewer[sample_num + curind], disjoint);
		}
	}
	}
}

__global__ void set_atmosphere(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	RigidFuncGpu2D mrrigidbody;
	if (x==2&&y==sample_y-2)
	{
		if (mlflow[0].tag_matrix[curind] > 0)
		{
			
			mlflow[0].bubble.rho[mlflow[0].tag_matrix[curind] - 1] = 1.f;
			mlflow[0].bubble.init_volume[mlflow[0].tag_matrix[curind] - 1] = mlflow[0].bubble.volume[mlflow[0].tag_matrix[curind] - 1];
		}

	}
}

// fix the flag as the FLUID3X
__global__ void stream_collide(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	RigidFuncGpu2D mrrigidbody;
	
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		mlflow[0].fMomViewer[curind] = 0.f;
		float Omega = 1 / ((1e-4) * 3.0f + 0.5f);
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_bo == TYPE_S) return; // cell processed here is fluid or interface


		if (flagsn_su == TYPE_G) return;
		
		float fhn[9]{};
		float fon[9]{};


		for (int i = 0; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			int ind_back = y1 * sample_x + x1;

			if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
			{
                float uxn_ = 0.f;
                float uyn_ = 0.f;
                float uzn_ = 0.f;
				REAL rhoVar = mlflow[0].fMom[curind * 6 + 0];
				
				
				REAL ux_x = mlflow[0].fMom[curind * 6 + 1];
				REAL uy_y = mlflow[0].fMom[curind * 6 + 2];
				
				REAL ux = uxn_;
				REAL uy = uyn_;

				REAL pixx = mlflow[0].fMom[curind * 6 + 3];
				REAL piyy = mlflow[0].fMom[curind * 6 + 4];
				REAL pixy = mlflow[0].fMom[curind * 6 + 5];

				// printf("uy %f, uyn_ %f\n", uy_y, uyn_);

				pixx = ux * ux + (pixx - ux_x * ux_x);
				piyy = uy * uy + (piyy - uy_y * uy_y);
				pixy = ux * uy + (pixy - ux_x * uy_y);

				
				mrutilfunc.mlCalDistributionD2Q9AtIndex(
					rhoVar, uxn_, uyn_, pixx, piyy, pixy, i, fhn[i]
				);

				fhn[i] -= w2d_gpu[i];
				
				float feq[9]{};
				mrutilfunc.calculate_f_eq(rhoVar, 0.f, 0.f, 0.f, feq); // calculate equilibrium DDFs
				fhn[i] = feq[i];
				
			}
			else
			{
				REAL rhoVar = mlflow[0].fMom[ind_back * 6 + 0];
				REAL ux = mlflow[0].fMom[ind_back * 6 + 1];
				REAL uy = mlflow[0].fMom[ind_back * 6 + 2];
				REAL pixx = mlflow[0].fMom[ind_back * 6 + 3];
				REAL piyy = mlflow[0].fMom[ind_back * 6 + 4];
				REAL pixy = mlflow[0].fMom[ind_back * 6 + 5];
				// might not need clamp
				float3 u_2{ ux,uy,0.f };
				u_2 = normalizing_clamp(u_2, 0.4);
				ux = u_2.x;
				uy = u_2.y;

				mrutilfunc.mlCalDistributionD2Q9AtIndex(
					rhoVar, ux, uy, pixx, pixy, piyy, i, fhn[i]
				);
				fhn[i] -= w2d_gpu[i];

			}

			REAL rhoVar = mlflow[0].fMom[curind * 6 + 0];
			REAL ux = mlflow[0].fMom[curind * 6 + 1];
			REAL uy = mlflow[0].fMom[curind * 6 + 2];
			REAL pixx = mlflow[0].fMom[curind * 6 + 3];
			REAL piyy = mlflow[0].fMom[curind * 6 + 4];
			REAL pixy = mlflow[0].fMom[curind * 6 + 5];
			// might not need clamp
			float3 u_2{ ux,uy,0.f };
			u_2 = normalizing_clamp(u_2, 0.4);
			ux = u_2.x;
			uy = u_2.y;

			mrutilfunc.mlCalDistributionD2Q9AtIndex(
				rhoVar, ux, uy, pixx, pixy, piyy, i, fon[i]
			);
			fon[i] -= w2d_gpu[i];
		}

		//fon[0] = fhn[0];

		float massn = mlflow[0].mass[curind];

		for (int i = 1; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			int ind_back = y1 * sample_x + x1;
			massn += mlflow[0].massex[ind_back]; // distribute excess mass from last step which is stored in neighbors
		}

		if (flagsn_su == TYPE_F) {
			for (int i = 1; i < 9; i++)
			{
				massn += fhn[i] - fon[i]; // neighbor is fluid or interface cell
			}
		}
		else if (flagsn_su == TYPE_I)
		{ // cell is interface
			float phij[9]; // cache fill level of neighbor lattice points
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);

				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;

				phij[i] = mlflow[0].phi[ind_back]; // cache fill level of neighbor lattice points

				if ((mlflow[0].flag[ind_back] & TYPE_BO)==TYPE_S)
					{	
						//printf("TYPE I %d",curind);
						//flag_sk = 1;
						if (x1==0)
						{
							int ind_k = y1 * sample_x + 1;
							phij[i] = mlflow[0].phi[ind_k]; 
						}
						else if (x1==sample_x-1)
						{
							int ind_k = y1 * sample_x + sample_x - 2;
							phij[i] = mlflow[0].phi[ind_k]; 
						}
						else if (y1==0)
						{
							int ind_k = 1 * sample_x + x1;
							phij[i] = mlflow[0].phi[ind_k]; 
						}
						else if (y1==sample_y-1)
						{
							int ind_k = (sample_y - 2) * sample_x + x1;
							phij[i] = mlflow[0].phi[ind_k]; 
						}
					}
			}
			float rhon = 0.0f, uxn = 0.0f, uyn = 0.0f, uzn = 0.0f, rho_laplace = 0.0f; // no surface tension if rho_laplace is not overwritten later

			//mrutilfunc.calculate_rho_u(fon, rhon, uxn, uyn, uzn);

			// can we use rhon_ here? but not calculate_rho_u rerecosntruct???
			rhon = mlflow[0].fMom[curind * 6 + 0];
			uxn = mlflow[0].fMom[curind * 6 + 1];
			uyn = mlflow[0].fMom[curind * 6 + 2];


			// why not double limitation????
			float3 u_{ uxn,uyn,uzn };
			u_ = normalizing_clamp(u_, 0.4);
			uxn = u_.x;
			uyn = u_.y;
			uzn = u_.z;

			phij[0] = mrutilfunc.calculate_phi(rhon, massn, flagsn); // don't load phi[n] from memory, instead recalculate it with mass corrected by excess mass

			float3 normal = mrutilfunc.calculate_normal(phij);
			float curv = mrutilfunc.calculate_curvature(phij, 0);
			rho_laplace = def_6_sigma == 0.0f ? 0.0f : def_6_sigma * curv; // surface tension least squares fit (PLIC, most accurate)
			

			int tag_curind = mlflow[0].tag_matrix[curind] - 1;
			
			// float disjoint = 0;


			float disjoint = mlflow[0].fMomViewer[sample_num + curind];
		
			//disjoint = disjoint / 400.f;
			// if (disjoint>0)
			// 	printf("disjoint %f\n",disjoint);
			mlflow[0].fMomViewer[sample_num * 2 + curind] = disjoint;
			mlflow[0].fMomViewer[curind] = mrutilfunc.calculate_curvature(phij,0);
			//}

			// if (curind>106173 && curind<106183)
			// {
			// 	float cur = mrutilfunc.calculate_curvature(phij,1);
			// 	printf("curvature %f\n", cur);

			// }


			
			// float3 normal = mrutilfunc.calculate_normal(phij);
			float feg[9]; // reconstruct f from neighbor gas lattice points
			const float rho2tmp = 0.5f / rhon; // apply external volume force (Guo forcing, Krueger p.233f)
			//NEED TO CHECK
			
			// need to fix the following code	
			float uxntmp = fma(mlflow[0].forcex[curind] * rhon, rho2tmp, uxn);// limit velocity (for stability purposes)
			float uyntmp = fma(mlflow[0].forcey[curind] * rhon, rho2tmp, uyn);// force term: F*dt/(2*rho)
			float uzntmp = uzn;
			float3 u_2{ uxntmp,uyntmp,uzntmp };

			u_2 = normalizing_clamp(u_2, 0.4);
			uxntmp = u_2.x;
			uyntmp = u_2.y;
			uzntmp = u_2.z;

			float factor_ = 1.f;
			float rho_k = 1.f;
			if (mlflow[0].tag_matrix[curind] > 0)
			{
				rho_k = mlflow[0].bubble.rho[mlflow[0].tag_matrix[curind] - 1];
				
			}
			
			rho_laplace = def_6_sigma == 0.0f ? 0.0f : def_6_sigma * curv;
			float in_rho = rho_k -  rho_laplace - 0.05 * disjoint;
			

			mrutilfunc.calculate_f_eq(in_rho, uxntmp, uyntmp, uzntmp, feg); // calculate gas equilibrium DDFs with constant ambient pressure
			

			unsigned char flagsj_su[9]; // cache neighbor flags for multiple readings
			unsigned char flagsj_bo[9];
			int num;
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int ind_back = y1 * sample_x + x1;
				flagsj_su[i] = mlflow[0].flag[ind_back] & TYPE_SU;
				flagsj_bo[i] = mlflow[0].flag[ind_back] & TYPE_BO;
				if (flagsj_bo[i] == TYPE_S)
					num++;
			}
			int flag[9]{ 0 };

			if (num < 3)
			{
				for (int i = 1; i < 9; i++)
				{
					if (normal.x * ex2d_gpu[i] + normal.y * ey2d_gpu[i] >= 0)
					{
						flag[i] = 1;
					}
				}
			}

			for (int i = 1; i < 9; i++)
			{ 
				massn += flagsj_su[i] & (TYPE_F | TYPE_I) ? flagsj_su[i] == TYPE_F ? fhn[i] - fon[index2dInv_gpu[i]] : 0.5f * (phij[i] + phij[0]) * (fhn[i] - fon[index2dInv_gpu[i]]) : 0.0f; // neighbor is fluid or interface cell	
			}

			float fhn_temp[9]{};

			for (int i = 1; i < 9; i++)
			{ // calculate reconstructed gas DDFs
				fhn_temp[i] = feg[index2dInv_gpu[i]] - fon[i] + feg[i];
			}

			int flag_s[9]{ 0 };
			
			for (int i = 1; i < 9; i++)
			{
				flag[i] = 0;
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int ind_back = y1 * sample_x + x1;
				//wrong (need to find the regular for the load_f )
				
				if (mlflow[0].flag[ind_back]==TYPE_S)
				{
					if (x1==0||x1==sample_x-1)
					{
						if (sign(normal.y)!= -1 *sign((float)dy))
							flag_s[i] = 1;
					}
					if (y1==0||y1==sample_y-1)
					{
						if (sign(normal.x)!= -1 * sign((float)dx))
							flag_s[i] = 1;
					}
				}
				if (flagsj_su[i] == TYPE_G||flag_s[i] == 1)
					fhn[i] = fhn_temp[index2dInv_gpu[i]];
			}
		}

		mlflow[0].mass[curind] = massn;
		


		REAL pop[9]{};

		// streaming		

		float f_eq_k[9];


		for (int i = 0; i < 9; i++)
		{
			pop[i] =  fhn[i] + w2d_gpu[i];
		}

		REAL FX = mlflow[0].forcex[curind];
		REAL FY = mlflow[0].forcey[curind];


		float rhon = 0.f, uxn = 0.f, uyn = 0.f, uzn = 0.f;


		rhon = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8];
		float fxn = FX * rhon, fyn = FY * rhon, fzn = 0.f;
		REAL invRho = 1 / rhon;
		uxn = ((pop[1] - pop[2] + pop[5] - pop[8] - pop[6] + pop[7]) * invRho + 0.5 * FX * invRho);
		uyn = ((pop[3] - pop[4] + pop[5] + pop[8] - pop[6] - pop[7]) * invRho + 0.5 * FY * invRho);
		REAL pixx = pop[1] + pop[2] + pop[5] + pop[8] + pop[6] + pop[7];
		REAL piyy = pop[3] + pop[4] + pop[5] + pop[8] + pop[6] + pop[7];
		REAL pixy = pop[5] - pop[8] + pop[6] - pop[7];

		
		float3 u_{ uxn,uyn,uzn };
		u_ = normalizing_clamp(u_, 0.4);
		uxn = u_.x;
		uyn = u_.y;
		uzn = u_.z;
		if (flagsn_su == TYPE_I)
		{
			bool TYPE_NO_F = true, TYPE_NO_G = true; // temporary flags for no fluid or gas neighbors
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);

				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;
				const unsigned char flagsji_su = mlflow[0].flag[ind_back] & TYPE_SU; // extract SURFACE flags
				TYPE_NO_F = TYPE_NO_F && flagsji_su != TYPE_F;
				TYPE_NO_G = TYPE_NO_G && flagsji_su != TYPE_G;
			}
			REAL massn = mlflow[0].mass[curind];
			//printf("massn : %f, rhon: %f\n", massn, rhon);
			if (massn > rhon || TYPE_NO_G)
			{
				mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((flagsn & ~TYPE_SU) | TYPE_IF); // set flag interface->fluid
			}
			else if (massn < 0.0f || TYPE_NO_F)
			{
				mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)((flagsn & ~TYPE_SU) | TYPE_IG); // set flag interface->gas
			}
		}


		

		int ind_front  = (sample_y-3) * sample_x + 5;
		for (int ij = -2;ij<2;ij++)
			for(int jk=-2;jk<2;jk++)
			{
				int x12 = x + jk;
				int y12 = y + ij;
				if (x12 >= 0 && x12 < sample_x && y12 >= 0 && y12 < sample_y)
				{
					int ind_back = (y + ij) * sample_x + x + jk;
				if (mlflow[0].tag_matrix[ind_back] > 0)
				{
				//if (mlflow[0].bubble.volume[mlflow[0].tag_matrix[curind] - 1] > 47000.f)
				Omega =  1.f / 0.9f;
			}
			if (mlflow[0].flag[ind_back] == TYPE_S)
			{
				Omega =  1.f / (3.0 * (1e-3) + 0.5f);
				break;
			}
			}
			}

	

		mrutilfunc.mlGetPIAfterCollision(
			rhon,
			uxn, uyn, FX, FY,
			Omega,
			pixx, piyy, pixy
		);
		pixx = 1 * (pixx * invRho - 1.0 * cs2);
		piyy = 1 * (piyy * invRho - 1.0 * cs2);
		pixy = 1 * (pixy * invRho);

		mlflow[0].fMomPost[curind * 6 + 0] = rhon;
		mlflow[0].fMomPost[curind * 6 + 1] = uxn + FX * invRho / 2.0f;
		mlflow[0].fMomPost[curind * 6 + 2] = uyn + FY * invRho / 2.0f;
		mlflow[0].fMomPost[curind * 6 + 3] = pixx;
		mlflow[0].fMomPost[curind * 6 + 4] = piyy;
		mlflow[0].fMomPost[curind * 6 + 5] = pixy;


		mlflow[0].rho[curind] = rhon; // update density field
		mlflow[0].u[curind].x = uxn + FX * invRho / 2.0f;// + FX * invRho / 2.0f;     // why not the forcing terms??
		mlflow[0].u[curind].y = uyn + FY * invRho / 2.0f;// + FY * invRho / 2.0f;
		mlflow[0].u[curind].z = uzn;

	}
}

__global__ void Init2D(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;


	mrUtilFuncGpu2D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{

		unsigned char flagsn = mlflow[0].flag[curind];
		const unsigned char flagsn_bo = flagsn & TYPE_BO; // extract boundary flags

		unsigned char flagsj[9]{};
		for (int i = 0; i < 9; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			if (x1 >= 0 && x1 < sample_x && y1 >= 0 && y1 < sample_y)
			{
				int ind_back = y1 * sample_x + x1;
				flagsj[i] = mlflow[0].flag[ind_back];
			}
		}

		// need to change for the moving boundary the speed need to change for no zero initial
		if (flagsn_bo == TYPE_S) { // cell is solid
			bool TYPE_ONLY_S = true; // has only solid neighbors
			for (int i = 1; i < 9; i++) TYPE_ONLY_S = TYPE_ONLY_S && (flagsj[i] & TYPE_BO) == TYPE_S;
			if (TYPE_ONLY_S) {
				mlflow[0].u[curind].x = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
				mlflow[0].u[curind].y = 0.0f;
				mlflow[0].u[curind].z = 0.0f;
			}
		}
		if (flagsn_bo == TYPE_S) {
			mlflow[0].u[curind].x = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
			mlflow[0].u[curind].y = 0.0f;
			mlflow[0].u[curind].z = 0.0f;
		}



		float feq[9]{}; // f_equilibrium

		mrutilfunc.calculate_f_eq(mlflow[0].rho[curind], mlflow[0].u[curind].x, mlflow[0].u[curind].y, mlflow[0].u[curind].z, feq);

		float geq[5]{};
		mrutilfunc.calculate_g_eq(mlflow[0].c_value[curind], mlflow[0].u[curind].x, mlflow[0].u[curind].y, mlflow[0].u[curind].z, geq);

		// separate block to avoid variable name conflicts
		float phin = mlflow[0].phi[curind];
		if (!(flagsn & (TYPE_S | TYPE_E | TYPE_T | TYPE_F | TYPE_I)))
			flagsn = (flagsn & ~TYPE_SU) | TYPE_G; // change all non-fluid and non-interface flags to gas
		if ((flagsn & TYPE_SU) == TYPE_G)
		{ // cell with updated flags is gas
			bool change = false; // check if cell has to be changed to interface
			for (int i = 1; i < 9; i++)
				change = change || (flagsj[i] & TYPE_SU) == TYPE_F; // if neighbor flag fluid is set, the cell must be interface
			if (change)
			{ // create interface automatically if phi has not explicitely defined for the interface layer
				flagsn = (flagsn & ~TYPE_SU) | TYPE_I; // cell must be interface
				phin = 0.5f;
				float rhon, uxn, uyn, uzn; // initialize interface cells with average density/velocity of fluid neighbors

				 // average over all fluid/interface neighbors
				float rhot = 0.0f, uxt = 0.0f, uyt = 0.0f, uzt = 0.0f, counter = 0.0f; // average over all fluid/interface neighbors
				for (int i = 1; i < 9; i++)
				{
					int dx = int(ex2d_gpu[i]);
					int dy = int(ey2d_gpu[i]);
					int x1 = x - dx;
					int y1 = y - dy;
					int ind_back = y1 * sample_x + x1;
					const unsigned char flagsji_su = mlflow[0].flag[ind_back] & TYPE_SU;
					if (flagsji_su == TYPE_F) { // fluid or interface or (interface->fluid) neighbor
						counter += 1.0f;
						rhot += mlflow[0].rho[ind_back];
						uxt += mlflow[0].u[ind_back].x;
						uyt += mlflow[0].u[ind_back].y;
						uzt += mlflow[0].u[ind_back].z;
					}

				}
				rhon = counter > 0.0f ? rhot / counter : 1.0f;
				uxn = counter > 0.0f ? uxt / counter : 0.0f;
				uyn = counter > 0.0f ? uyt / counter : 0.0f;
				uzn = 0.0f;
				mrutilfunc.calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs

				float rhon_g; // initialize interface cells with average density/velocity of fluid neighbors
				 // average over all fluid/interface neighbors
				float rhogt = 0.0f, c_k = 0.0f; // average over all fluid/interface neighbors
				for (int i = 1; i < 5; i++)
				{
					int dx = int(ex2d_gpu[i]);
					int dy = int(ey2d_gpu[i]);
					int x1 = x - dx;
					int y1 = y - dy;
					int ind_back = y1 * sample_x + x1;
					const unsigned char flagsji_su = mlflow[0].flag[ind_back] & TYPE_SU;
					if (flagsji_su == TYPE_F) { // fluid or interface or (interface->fluid) neighbor
						c_k += 1.0f;
						rhogt += mlflow[0].c_value[ind_back];
					}
				}
				rhon_g = c_k > 0.0f ? rhogt / c_k : 0.0f;
				mrutilfunc.calculate_g_eq(rhon_g, uxn, uyn, uzn, geq); // calculate equilibrium DDFs
			}
		}
		if ((flagsn & TYPE_SU) == TYPE_G) { // cell with updated flags is still gas
			mlflow[0].u[curind].x = 0.0f; // reset velocity for solid lattice points with only boundary neighbors
			mlflow[0].u[curind].y = 0.0f;
			mlflow[0].u[curind].z = 0.0f;
			phin = 0.0f;
		}
		else if ((flagsn & TYPE_SU) == TYPE_I && (phin < 0.0f || phin>1.0f)) {
			phin = 0.5f; // cell should be interface, but phi was invalid

		}
		else if ((flagsn & TYPE_SU) == TYPE_F) {
			phin = 1.0f;
		}
		mlflow[0].phi[curind] = phin;
		mlflow[0].mass[curind] = phin * mlflow[0].rho[curind];
		mlflow[0].massex[curind] = 0.0f; // reset excess mass
		mlflow[0].flag[curind] = (MLLATTICENODE_SURFACE_FLAG)flagsn;

		for (int i = 0; i < 9; i++)
		{
			feq[i] += w2d_gpu[i];
		}
		REAL invRho = 1 / mlflow[0].rho[curind];
		REAL pixx = feq[1] + feq[2] + feq[5] + feq[8] + feq[6] + feq[7];
		REAL piyy = feq[3] + feq[4] + feq[5] + feq[8] + feq[6] + feq[7];
		REAL pixy = feq[5] - feq[8] + feq[6] - feq[7];
		pixx = 1 * (pixx * invRho - cs2);
		piyy = 1 * (piyy * invRho - cs2);
		pixy = 1 * (pixy * invRho);
		mlflow[0].fMomPost[curind * 6 + 0] = mlflow[0].fMom[curind * 6 + 0] = mlflow[0].rho[curind];
		mlflow[0].fMomPost[curind * 6 + 1] = mlflow[0].fMom[curind * 6 + 1] = mlflow[0].u[curind].x;
		mlflow[0].fMomPost[curind * 6 + 2] = mlflow[0].fMom[curind * 6 + 2] = mlflow[0].u[curind].y;
		mlflow[0].fMomPost[curind * 6 + 3] = mlflow[0].fMom[curind * 6 + 3] = pixx;
		mlflow[0].fMomPost[curind * 6 + 4] = mlflow[0].fMom[curind * 6 + 4] = piyy;
		mlflow[0].fMomPost[curind * 6 + 5] = mlflow[0].fMom[curind * 6 + 5] = pixy;

		for (int i = 0; i < 5; i++)
		{
			mlflow[0].gMom[curind + i * sample_num] = geq[i];
			mlflow[0].gMomPost[curind + i * sample_num] = geq[i];
		}

	}

}


__global__ void InitTag(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	/*
	*
	struct mlBubble2D
	{
	REAL* volume;
	REAL* init_volume;
	REAL* rho;
	// REAL* volume_diff;

	float3* center;
	REAL* disjoint_pressure;
	REAL* label_volume;
	REAL label_num;
	// REAL *T;

	unsigned int max_bubble_count = 1024;
	int bubble_count = 0;
	};
	*

	int* tag_matrix;
	unsigned char* input_matrix;
	unsigned int* label_matrix;

	bool* split_detector;
	int2* split_record;
	int* split_tag_record;
	int split_record_length;

	bool* merge_detector;
	int2* merge_record;
	int merge_record_length;

	bool merge_flag;
	bool split_flag;

	delta_phi
	*/
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		unsigned char flagsn = mlflow[0].flag[curind];
		const unsigned char flagsn_bo = flagsn & TYPE_BO; // extract boundary flags

		if (flagsn_bo == TYPE_S)
			mlflow[0].tag_matrix[curind] = -1;
		if (flagsn == TYPE_F)
			mlflow[0].tag_matrix[curind] = -1;
		mlflow[0].previous_tag[curind] = -1;
	}
}




__global__ void create_bubble_label(mrFlow2D* mlflow)
{
	for (int i = 0; i < mlflow[0].bubble.label_num; i++)
	{
		mlflow[0].bubble.volume[i] = mlflow[0].bubble.label_volume[i];
		mlflow[0].bubble.init_volume[i] = mlflow[0].bubble.label_volume[i];
		mlflow[0].bubble.rho[i] = 1.0f;
		mlflow[0].bubble.pure_gas_volume[i] = mlflow[0].bubble.pure_label_gas_volume[i];
		// fix the volume for the initial bubble
		mlflow[0].bubble.freeze[i] = 1;
	}
	mlflow[0].bubble.bubble_count = mlflow[0].bubble.label_num;

}


__global__ void update_init_tag(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{

		if ((int)mlflow[0].label_matrix[curind] > 0)
		{

			mlflow[0].tag_matrix[curind] = (int)mlflow[0].label_matrix[curind];

		}

	}

}

void InitBubble(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	InitTag << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	convertIntToUnsignedChar << <grid1, threads1 >> > (mlflow, sample_x, sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	connectedComponentLabeling(mlflow, (size_t)sample_x, (size_t)sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int h_tmp = 0;
	int* d_tmp;

	cudaMalloc(&d_tmp, sizeof(int));
	cudaMemcpy(d_tmp, &h_tmp, sizeof(int), cudaMemcpyHostToDevice);

	PushLabelNumKernel << <1, 1 >> > (mlflow, d_tmp);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	parse_label << <grid1, threads1 >> > (mlflow, param, sample_x, sample_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	create_bubble_label << <1, 1 >> > (mlflow);
	cudaDeviceSynchronize();
	update_init_tag << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_num);
	cudaDeviceSynchronize();
	ClearDectector(mlflow, param);
	

}


__host__ __device__
void MomSwap2D(REAL*& pt1, REAL*& pt2) {
	REAL* temp = pt1;
	pt1 = pt2;
	pt2 = temp;
}

__global__ void mrSolver2D_step2Kernel(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	MomSwap2D(mlflow[0].fMom, mlflow[0].fMomPost);
}

__global__ void mrSolver2D_g_step2Kernel(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	MomSwap2D(mlflow[0].gMom, mlflow[0].gMomPost);
}

void mrInit2DGpu(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	mrUtilFuncGpu2D mrutilfunc;
	Init2D << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	InitBubble(mlflow, param);
	ResetLabelVolume<<<1,1>>>(mlflow);
	cudaDeviceSynchronize();
}


__global__ void g_stream_collide(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num, int time)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_bo == TYPE_S || flagsn_su == TYPE_G) return;


		// inject the g's streaming
		REAL pop_g[5]{};
		float g_eq_k[5];
		float rhon_g = 0.f, uxn_g = 0.f, uyn_g = 0.f, uzn_g = 0.f;
		for (int i = 0; i < 5; i++)
			rhon_g += mlflow[0].gMom[curind + i * sample_num];
		mrutilfunc.calculate_g_eq(rhon_g, uxn_g, uyn_g, uzn_g, g_eq_k);
		

		for (int i = 0; i < 5; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			int ind_back = y1 * sample_x + x1;

			if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
			{
				//pop_g[i] =  mlflow[0].gMom[curind + index2dInv_gpu[i] * sample_num];
				int y_ = (y1 > sample_y - 2) ? 1 : y1;
				y_ = (y_ < 1) ? sample_y - 2 : y_;
				int x_ = (x1 > sample_x - 2) ? 1 : x1;
				x_ = (x_ < 1) ? sample_x - 2 : x_;
				int curind_ =  y_ * sample_x + x_;
				if (mlflow[0].flag[curind_]==TYPE_F && mlflow[0].flag[ind_back] == TYPE_I)
				{
					pop_g[i] = mlflow[0].gMom[curind_ + i * sample_num];
				}
			}
			else
			{
				pop_g[i] = mlflow[0].gMom[ind_back + i * sample_num];
			}

		}

		// NEED TO CHECK
		REAL FX = mlflow[0].forcex[curind];
		REAL FY = mlflow[0].forcey[curind];


		float rhon = 0.f, uxn = 0.f, uyn = 0.f, uzn = 0.f;
		float fxn = FX, fyn = FY, fzn = 0.f;

		//mrutilfunc.calculate_rho_u(pop, rhon, uxn, uyn, uzn);
		rhon = mlflow[0].fMom[curind * 6 + 0];
		uxn = mlflow[0].fMom[curind * 6 + 1];
		uyn = mlflow[0].fMom[curind * 6 + 2];
		
		// D2Q5 g
		float g_eq[5];
		
		float rhon_gt = 0.f;
		for (int i = 0; i < 5; i++)
			rhon_gt += pop_g[i];

		mrutilfunc.calculate_g_eq(rhon_gt, uxn, uyn, uzn, g_eq);
		mlflow[0].c_value[curind] = rhon_gt;
		// low order bgk
		REAL w = 1.0f / 0.6f;
		float src = 0.f;
		//REAL c_tau = fma(w, -0.5f, 1.0f);
		for (int i = 0; i < 5; i++)
		{
			//if (time<200)
			src = mlflow[0].src[curind];
			REAL pop_out;
			pop_out = fma(1.0f - w, pop_g[i], fma(w, g_eq[i], src));
			mlflow[0].gMomPost[curind + i * sample_num] = pop_out;
		}
	}
}


__global__ void g_reconstruction(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int curind = y * sample_x + x;
	mrUtilFuncGpu2D mrutilfunc;
	if (
		(x >= 0 && x <= sample_x - 1) &&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		const unsigned char flagsn = mlflow[0].flag[curind]; // cache flags[n] for multiple readings
		const unsigned char flagsn_bo = flagsn & TYPE_BO, flagsn_su = flagsn & TYPE_SU; // extract boundary and surface flags
		if (flagsn_bo == TYPE_S || flagsn_su == TYPE_G) return; // cell processed here is fluid or interface


		// g temporary streaming
		float ghn[5];
		float gon[5];
		float g_eq_k[5];
		float rhon_g = 0.f, uxn_g = 0.f, uyn_g = 0.f, uzn_g = 0.f;
		
		for (int i = 0; i < 5; i++)
			rhon_g += mlflow[0].gMom[curind + i * sample_num];

		mrutilfunc.calculate_g_eq(rhon_g, uxn_g, uyn_g, uzn_g, g_eq_k);
		// mlflow[0].c_value[curind] = rhon_g;

		for (int i = 0; i < 5; i++)
		{
			int dx = int(ex2d_gpu[i]);
			int dy = int(ey2d_gpu[i]);
			int x1 = x - dx;
			int y1 = y - dy;

			int ind_back = y1 * sample_x + x1;
			gon[i] = mlflow[0].gMom[curind + i * sample_num];

			if ((mlflow[0].flag[ind_back] & TYPE_BO) == TYPE_S)
			{

				int y_ = (y1 > sample_y - 2) ? 1 : y1;
				y_ = (y_ < 1) ? sample_y - 2 : y_;
				int x_ = (x1 > sample_x - 2) ? 1 : x1;
				x_ = (x_ < 1) ? sample_x - 2 : x_;
				int curind_ =  y_ * sample_x + x_;
				if (mlflow[0].flag[curind_]==TYPE_F && mlflow[0].flag[ind_back] == TYPE_I)
				{
					ghn[i] = mlflow[0].gMom[curind_ + i * sample_num];
				}
				else
					ghn[i] = mlflow[0].gMom[curind + index2dInv_gpu[i] * sample_num];
			}
			else
			{
				ghn[i] = mlflow[0].gMom[ind_back + i * sample_num];
			}

		}
		gon[0] = ghn[0];


		if (flagsn_su == TYPE_I)
		{ // cell is interface
			float phij[9]; // cache fill level of neighbor lattice points
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);

				int x1 = x - dx;
				int y1 = y - dy;

				int ind_back = y1 * sample_x + x1;
				phij[i] = mlflow[0].phi[ind_back]; // cache fill level of neighbor lattice points

			}
			float rhon = 0.0f, uxn = 0.0f, uyn = 0.0f, uzn = 0.0f, rho_laplace = 0.0f; // no surface tension if rho_laplace is not overwritten later
			rhon = mlflow[0].fMom[curind * 6 + 0];
			uxn = mlflow[0].fMom[curind * 6 + 1];
			uyn = mlflow[0].fMom[curind * 6 + 2];

			float3 normal = mrutilfunc.calculate_normal(phij);
			

			REAL rho_k = 1.f;
			if (mlflow[0].tag_matrix[curind] > 0)
			{
				rho_k = mlflow[0].bubble.rho[mlflow[0].tag_matrix[curind] - 1];
				if (mlflow[0].bubble.volume[mlflow[0].tag_matrix[curind] - 1] > 40000.f)
					rho_k = 1.f;
			}
			float in_rho = K_h  /3.f * rho_k;
			


			// henry's law
			float geg[5]{};
			// mrutilfunc.calculate_g_eq(K_h /3.f* rho_k, uxn, uyn, uzn, geg);

			mrutilfunc.calculate_g_eq(in_rho, uxn, uyn, uzn, geg);
			unsigned char flagsj_su[9]; // cache neighbor flags for multiple readings
			unsigned char flagsj_bo[9];
			for (int i = 1; i < 9; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int ind_back = y1 * sample_x + x1;
				flagsj_su[i] = mlflow[0].flag[ind_back] & TYPE_SU;
				flagsj_bo[i] = mlflow[0].flag[ind_back] & TYPE_BO;
			}
		

			// g's reconstruction
			float g_delta = 0.f;

			for (int i = 1; i < 5; i++)
			{
				g_delta += flagsj_su[i] == TYPE_F ? ghn[i] - gon[index2dInv_gpu[i]] : 0.f;
				//printf("g_delta %f %f\n",ghn[i], gon[index2dInv_gpu[i]]);
			}
			
			mlflow[0].delta_g[curind] += g_delta;

			for (int i = 1; i < 5; i++)
			{
				ghn[i] = geg[index2dInv_gpu[i]] - gon[i] + geg[i];
			}
			for (int i = 1; i < 5; i++)
			{
				int dx = int(ex2d_gpu[i]);
				int dy = int(ey2d_gpu[i]);
				int x1 = x - dx;
				int y1 = y - dy;
				int ind_back = y1 * sample_x + x1;
				//wrong (need to find the regular for the load_f )
				if (flagsj_su[i] == TYPE_G)
					mlflow[0].gMom[ind_back + i * sample_num] = ghn[index2dInv_gpu[i]];

			}
		}

	}
}


__host__ __device__
void MomSwap2D(MLLATTICENODE_SURFACE_FLAG*& pt1, MLLATTICENODE_SURFACE_FLAG*& pt2) {
	MLLATTICENODE_SURFACE_FLAG* temp = pt1;
	pt1 = pt2;
	pt2 = temp;
}


__global__ void update_solid_flag(mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	MomSwap2D(mlflow[0].flag, mlflow[0].postflag);
}


void g_handle(mrFlow2D* mlflow, MLFluidParam2D* param ,int time)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;
	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);
	g_reconstruction << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	g_stream_collide << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num, time
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	bubble_volume_g_update_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num, time
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	mrSolver2D_g_step2Kernel << <1, 1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	bubble_rho_update_kernel << <1, 1 >> > (mlflow);

}




void mrSolver2DGpu(mrFlow2D* mlflow, MLFluidParam2D* param, int time_step)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int t = time_step;
	//printf("time t", time_step);
	int sample_num = sample_x * sample_y;

	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y),
		1
	);

	calculate_disjoint << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	set_atmosphere << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num,t
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());		
	stream_collide << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num, t
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	ResetMassexn << <grid1, threads1 >> >
	(
		mlflow,
		sample_x, sample_y,
		sample_num
		);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	surface_1 << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	surface_2 << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	surface_3 << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	mrSolver2D_step2Kernel << <1, 1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	assign_tag_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	update_merge_tag_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	reportLiquidToInterfaceConversion << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	reportInterfaceToLiquidConversion_kernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	update_bubble(mlflow, param);
	g_handle(mlflow, param, time_step);
	if (time_step>2600 && time_step % 500==0)
		print_bubble << <1, 1 >> > (mlflow);
		checkCudaErrors(cudaDeviceSynchronize());

	if (time_step % 100 == 0)
	{
		print_bubble << <1, 1 >> > (mlflow);
	}
}
