// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


 //#include "../../../lw_core_win/mlCoreWinHeader.h"
#include "../../../common/mlCoreWin.h"
#include "../../../common/mlcudaCommon.h"

#include "../../../common/mlcudaCommon.h"
#include "mrConstantParamsGpu3D.h"
#include "mrUtilFuncGpu3D.h"
#include "mrLbmSolverGpu3D.h"

#pragma push_macro("REAL")
#undef REAL
#include <opencv2/core.hpp>
#include "cuda_types3.hpp"
#include "cuda_mat3.hpp"
#pragma pop_macro("REAL")

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/sort.h>
#include <thrust/unique.h>

using namespace cv;
namespace {

    // Only use it with unsigned numeric types
    template <typename T>
    __device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
        return (bitmap >> pos) & 1;
    }

    // Only use it with unsigned numeric types
    //template <typename T>
    //__device__ __forceinline__ void SetBit(T &bitmap, unsigned char pos) {
    //    bitmap |= (1 << pos);
    //}

    // Returns the root index of the UFTree
    __device__ unsigned Find(const int *s_buf, unsigned n) {
        while (s_buf[n] != n) {
            n = s_buf[n];
        }
        return n;
    }

    __device__ unsigned FindAndCompress(int *s_buf, unsigned n) {
        unsigned id = n;
        while (s_buf[n] != n) {
            n = s_buf[n];
            s_buf[id] = n;
        }
        return n;
    }

    // Merges the UFTrees of a and b, linking one root to the other
    __device__ void Union(int *s_buf, unsigned a, unsigned b) {

        bool done;

        do {

            a = Find(s_buf, a);
            b = Find(s_buf, b);

            if (a < b) {
                int old = atomicMin(s_buf + b, a);
                done = (old == b);
                b = old;
            }
            else if (b < a) {
                int old = atomicMin(s_buf + a, b);
                done = (old == a);
                a = old;
            }
            else {
                done = true;
            }

        } while (!done);

    }


    __global__ void InitLabeling(cv::cuda::PtrStepSz3i labels) {
        unsigned x = (blockIdx.x * BLOCK_X + threadIdx.x) * 2;
        unsigned y = (blockIdx.y * BLOCK_Y + threadIdx.y) * 2;
        unsigned z = (blockIdx.z * BLOCK_Z + threadIdx.z) * 2;
        unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

        if (x < labels.x && y < labels.y && z < labels.z) {
            labels[labels_index] = (int) labels_index;
        }
    }

    __global__ void Merge(const cv::cuda::PtrStepSz3b img, cv::cuda::PtrStepSz3i labels, unsigned char* last_cube_fg) {

        unsigned x = (blockIdx.x * BLOCK_X + threadIdx.x) * 2;
        unsigned y = (blockIdx.y * BLOCK_Y + threadIdx.y) * 2;
        unsigned z = (blockIdx.z * BLOCK_Z + threadIdx.z) * 2;
        unsigned img_index = z * (img.stepz / img.elem_size) + y * (img.stepy / img.elem_size) + x;
        unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

        if (x < labels.x && y < labels.y && z < labels.z) {

            const unsigned long long P0 = 0x77707770777;

            unsigned long long P = 0ULL;
            unsigned char foreground = 0;
            unsigned short buffer;

            {
                if (x + 1 < img.x) {
                    buffer = *reinterpret_cast<unsigned short *>(img.data + img_index);
                    if (buffer & 1) {
                        P |= P0;
                        foreground |= 1;
                    }
                    if (buffer & (1 << 8)) {
                        P |= (P0 << 1);
                        foreground |= (1 << 1);
                    }

                    if (y + 1 < img.y) {
                        buffer = *reinterpret_cast<unsigned short *>(img.data + img_index + img.stepy / img.elem_size);
                        if (buffer & 1) {
                            P |= (P0 << 4);
                            foreground |= (1 << 2);
                        }
                        if (buffer & (1 << 8)) {
                            P |= (P0 << 5);
                            foreground |= (1 << 3);
                        }
                    }

                    if (z + 1 < img.z) {
                        buffer = *reinterpret_cast<unsigned short *>(img.data + img_index + img.stepz / img.elem_size);
                        if (buffer & 1) {
                            P |= (P0 << 16);
                            foreground |= (1 << 4);
                        }
                        if (buffer & (1 << 8)) {
                            P |= (P0 << 17);
                            foreground |= (1 << 5);
                        }

                        if (y + 1 < img.y) {
                            buffer = *reinterpret_cast<unsigned short *>(img.data + img_index + img.stepz / img.elem_size + img.stepy / img.elem_size);
                            if (buffer & 1) {
                                P |= (P0 << 20);
                                foreground |= (1 << 6);
                            }
                            if (buffer & (1 << 8)) {
                                P |= (P0 << 21);
                                foreground |= (1 << 7);
                            }

                        }

                    }

                }
                else {
                    if (img[img_index]) {
                        P |= P0;
                        foreground |= 1;
                    }

                    if (y + 1 < labels.y) {
                        if (img[img_index + img.stepy / img.elem_size]) {
                            P |= (P0 << 4);
                            foreground |= (1 << 2);
                        }
                    }

                    if (z + 1 < labels.z) {

                        if (img[img_index + img.stepz / img.elem_size]) {
                            P |= (P0 << 16);
                            foreground |= (1 << 4);
                        }

                        if (y + 1 < labels.y) {
                            if (img[img_index + img.stepz / img.elem_size + img.stepy / img.elem_size]) {
                                P |= (P0 << 20);
                                foreground |= (1 << 6);
                            }
                        }

                    }
                }
            }
         // Store foreground voxels bitmask into memory
            if (x + 1 < labels.x) {
                labels[labels_index + 1] = foreground;
            }
            else if (y + 1 < labels.y) {
                labels[labels_index + labels.stepy / labels.elem_size] = foreground;
            }
            else if (z + 1 < labels.z) {
                labels[labels_index + labels.stepz / labels.elem_size] = foreground;
            }
            else {
                *last_cube_fg = foreground;
            }


            // checks on borders

            if (x == 0) {
                P &= 0xEEEEEEEEEEEEEEEE;
            }
            if (x + 1 >= img.x) {
                P &= 0x3333333333333333;
            }
            else if (x + 2 >= img.x) {
                P &= 0x7777777777777777;
            }

            if (y == 0) {
                P &= 0xFFF0FFF0FFF0FFF0;
            }
            if (y + 1 >= img.y) {
                P &= 0x00FF00FF00FF00FF;
            }
            else if (y + 2 >= img.y) {
                P &= 0x0FFF0FFF0FFF0FFF;
            }

            if (z == 0) {
                P &= 0xFFFFFFFFFFFF0000;
            }
            if (z + 1 >= img.z) {
                P &= 0x00000000FFFFFFFF;
            }

            // P is now ready to be used to find neighbour blocks
            // P value avoids range errors

            if (P > 0) {

                // Lower plane
                unsigned char * plane_data = img.data + img_index - img.stepz;
                unsigned lower_plane_index = labels_index - 2 * (labels.stepz / labels.elem_size);

                if (HasBit(P, 0) && plane_data[0 - img.stepy - 1]) {
                    Union(labels.data, labels_index, lower_plane_index - 2 * (labels.stepy / labels.elem_size + 1));
                }

                if ((HasBit(P, 1) && plane_data[0 - img.stepy]) || (HasBit(P, 2) && plane_data[0 - img.stepy + 1])) {
                    Union(labels.data, labels_index, lower_plane_index - 2 * (labels.stepy / labels.elem_size));
                }

                if (HasBit(P, 3) && plane_data[0 - img.stepy + 2]) {
                    Union(labels.data, labels_index, lower_plane_index - 2 * (labels.stepy / labels.elem_size - 1));
                }

                if ((HasBit(P, 4) && plane_data[-1]) || (HasBit(P, 8) && plane_data[img.stepy - 1])) {
                    Union(labels.data, labels_index, lower_plane_index - 2);
                }

                if ((HasBit(P, 5) && plane_data[0]) || (HasBit(P, 6) && plane_data[1]) || (HasBit(P, 9) && plane_data[img.stepy]) || (HasBit(P, 10) && plane_data[img.stepy + 1])) {
                    Union(labels.data, labels_index, lower_plane_index);
                }

                if ((HasBit(P, 7) && plane_data[2]) || (HasBit(P, 11) && plane_data[img.stepy + 2])) {
                    Union(labels.data, labels_index, lower_plane_index + 2);
                }

                if (HasBit(P, 12) && plane_data[2 * img.stepy - 1]) {
                    Union(labels.data, labels_index, lower_plane_index + 2 * (labels.stepy / labels.elem_size - 1));
                }

                if ((HasBit(P, 13) && plane_data[2 * img.stepy]) || (HasBit(P, 14) && plane_data[2 * img.stepy + 1])) {
                    Union(labels.data, labels_index, lower_plane_index + 2 * (labels.stepy / labels.elem_size));
                }

                if (HasBit(P, 15) && plane_data[2 * img.stepy + 2]) {
                    Union(labels.data, labels_index, lower_plane_index + 2 * (labels.stepy / labels.elem_size + 1));
                }

                // Current planes
                plane_data += img.stepz;

                if ((HasBit(P, 16) && plane_data[0 - img.stepy - 1]) || (HasBit(P, 32) && plane_data[img.stepz - img.stepy - 1])) {
                    Union(labels.data, labels_index, labels_index - 2 * (labels.stepy / labels.elem_size + 1));
                }

                if ((HasBit(P, 17) && plane_data[0 - img.stepy]) || (HasBit(P, 18) && plane_data[0 - img.stepy + 1]) || (HasBit(P, 33) && plane_data[img.stepz - img.stepy]) || (HasBit(P, 34) && plane_data[img.stepz - img.stepy + 1])) {
                    Union(labels.data, labels_index, labels_index - 2 * (labels.stepy / labels.elem_size));
                }

                if ((HasBit(P, 19) && plane_data[0 - img.stepy + 2]) || (HasBit(P, 35) && plane_data[img.stepz - img.stepy + 2])) {
                    Union(labels.data, labels_index, labels_index - 2 * (labels.stepy / labels.elem_size - 1));
                }

                if ((HasBit(P, 20) && plane_data[-1]) || (HasBit(P, 24) && plane_data[img.stepy - 1]) || (HasBit(P, 36) && plane_data[img.stepz - 1]) || (HasBit(P, 40) && plane_data[img.stepz + img.stepy - 1])) {
                    Union(labels.data, labels_index, labels_index - 2);
                }
            }

        }
    }

    __global__ void PathCompression(cv::cuda::PtrStepSz3i labels) {

        unsigned x = 2 * (blockIdx.x * BLOCK_X + threadIdx.x);
        unsigned y = 2 * (blockIdx.y * BLOCK_Y + threadIdx.y);
        unsigned z = 2 * (blockIdx.z * BLOCK_Z + threadIdx.z);
        unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

        if (x < labels.x && y < labels.y && z < labels.z) {
            FindAndCompress(labels.data, labels_index);
        }
    }

    __global__ void FinalLabeling(const cv::cuda::PtrStepSz3b img, cv::cuda::PtrStepSz3i labels, unsigned char* last_cube_fg) {

        unsigned x = 2 * (blockIdx.x * BLOCK_X + threadIdx.x);
        unsigned y = 2 * (blockIdx.y * BLOCK_Y + threadIdx.y);
        unsigned z = 2 * (blockIdx.z * BLOCK_Z + threadIdx.z);
        unsigned labels_index = z * (labels.stepz / labels.elem_size) + y * (labels.stepy / labels.elem_size) + x;

        if (x < labels.x && y < labels.y && z < labels.z) {

            int label;
            unsigned char foreground;
            unsigned long long buffer;

            if (x + 1 < labels.x) {
                buffer = *reinterpret_cast<unsigned long long *>(labels.data + labels_index);
                label = (buffer & (0xFFFFFFFF)) + 1;
                foreground = (buffer >> 32) & 0xFFFFFFFF;
            }
            else {
                label = labels[labels_index] + 1;
                if (y + 1 < labels.y) {
                    foreground = labels[labels_index + labels.stepy / labels.elem_size];
                }
                else if (z + 1 < labels.z) {
                    foreground = labels[labels_index + labels.stepz / labels.elem_size];
                }
                else {
                    foreground = *last_cube_fg;
                }
            }
            if (x + 1 < labels.x) {
                *reinterpret_cast<unsigned long long *>(labels.data + labels_index) =
                    (static_cast<unsigned long long>(((foreground >> 1) & 1) * label) << 32) | (((foreground >> 0) & 1) * label);

                if (y + 1 < labels.y) {
                    *reinterpret_cast<unsigned long long *>(labels.data + labels_index + labels.stepy / labels.elem_size) =
                        (static_cast<unsigned long long>(((foreground >> 3) & 1) * label) << 32) | (((foreground >> 2) & 1) * label);
                }

                if (z + 1 < labels.z) {
                    *reinterpret_cast<unsigned long long *>(labels.data + labels_index + labels.stepz / labels.elem_size) =
                        (static_cast<unsigned long long>(((foreground >> 5) & 1) * label) << 32) | (((foreground >> 4) & 1) * label);

                    if (y + 1 < labels.y) {
                        *reinterpret_cast<unsigned long long *>(labels.data + labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size)) =
                            (static_cast<unsigned long long>(((foreground >> 7) & 1) * label) << 32) | (((foreground >> 6) & 1) * label);
                    }

                }

            }
            else {
                labels[labels_index] = ((foreground >> 0) & 1) * label;
                if (y + 1 < labels.y) {
                    labels[labels_index + (labels.stepy / labels.elem_size)] = ((foreground >> 2) & 1) * label;
                }

                if (z + 1 < labels.z) {

                    labels[labels_index + labels.stepz / labels.elem_size] = ((foreground >> 4) & 1) * label;

                    if (y + 1 < labels.y) {
                        labels[labels_index + labels.stepz / labels.elem_size + (labels.stepy / labels.elem_size)] = ((foreground >> 6) & 1) * label;
                    }

                }
            }

        }

    }

}

__global__ void device_upload(mrFlow3D* mlflow, int* d_img_labels, unsigned char* d_imgs, int sample_x, int sample_y, int sample_z) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    int curind = z * sample_x * sample_y + y * sample_x + x;
    if (
        (x >= 0 && x <= sample_x - 1) &&
        (y >= 0 && y <= sample_y - 1) &&
        (z >= 0 && z <= sample_z - 1)
        )
    {
        d_img_labels[curind] = mlflow[0].label_matrix[curind];
        d_imgs[curind] = mlflow[0].input_matrix[curind];
    }
}


__global__ void device_download(mrFlow3D* mlflow, cv::cuda::PtrStepSz3i d_img_labels, cv::cuda::PtrStepSz3b d_imgs, int sample_x, int sample_y, int sample_z) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    int curind = z * sample_x * sample_y + y * sample_x + x;
    if (
        (x >= 0 && x <= sample_x - 1) &&
        (y >= 0 && y <= sample_y - 1) &&
        (z >= 0 && z <= sample_z - 1)
        )
    {
        mlflow[0].label_matrix[curind] = d_img_labels.data[curind];
		mlflow[0].input_matrix[curind] = d_imgs.data[curind];
    }
}


__global__ void renumber_0(
    int sample_x, int sample_y, int sample_z, int* d_bubble_map, cv::cuda::PtrStepSz3i d_img_labels) {
    // Calculate index
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    int curind = z * sample_x * sample_y + y * sample_x + x;
    if (curind < sample_x * sample_y * sample_z) {
        d_bubble_map[curind] = d_img_labels.data[curind];
    }
}

__global__ void renumber_1(int sample_x,int sample_y, int sample_z, int bubble_count, int* d_bubble_map, cv::cuda::PtrStepSz3i d_label) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    int curind = z * sample_x * sample_y + y * sample_x + x;
    if (curind < bubble_count) {
        if (curind > 0)
        {
            d_label.data[d_bubble_map[curind] - 1] = curind;
        }
    }
}

__global__ void renumber_2(mrFlow3D* mlflow,
    int sample_x, int sample_y, int sample_z, int bubble_count, int* d_bubble_map, cv::cuda::PtrStepSz3i d_label)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    int curind = z * sample_x * sample_y + y * sample_x + x;

    int* g_labels = mlflow[0].label_matrix;
    if (curind < sample_x *sample_y* sample_z) {
        if (g_labels[curind] > 0)
        {
            g_labels[curind] = d_label.data[g_labels[curind] - 1];
        }
    }
}


void PerformLabeling(mrFlow3D* mlflow, int sample_x, int sample_y, int sample_z) {
    
    int* d_img_labels; 

    unsigned char* d_imgs; 
    int total_num = sample_x * sample_y * sample_z;
    cudaMalloc(&d_imgs, total_num * sizeof(unsigned char*));
    cudaMalloc(&d_img_labels, total_num * sizeof(int));

    thrust::device_vector<int> d_bubble_map(total_num);
    int* d_ptr_1 = thrust::raw_pointer_cast(d_bubble_map.data());

    dim3 threads1(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 grid1(
        ceil(float(sample_x) / threads1.x),
        ceil(float(sample_y) / threads1.y),
        ceil(float(sample_z) / threads1.z)
    );

    // upload the data to the opencv format
    device_upload << <grid1, threads1 >> > (mlflow, d_img_labels, d_imgs,sample_x,sample_y,sample_z);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // prepare the data for the algorithm
	size_t stepy_ = sizeof(int) * static_cast<size_t>(sample_x);
    size_t stepz_ = stepy_ * static_cast<size_t>(sample_y);
    cv::cuda::PtrStepSz3i d_img_labels_(sample_x, sample_y, sample_z, d_img_labels, stepy_, stepz_);
    size_t stepy_2 = sizeof(unsigned char) * static_cast<size_t>(sample_x);
    size_t stepz_2 = stepy_2 * static_cast<size_t>(sample_y);
    cv::cuda::PtrStepSz3b d_img_(sample_x, sample_y, sample_z, d_imgs, stepy_2, stepz_2);
    unsigned char* last_cube_fg_;
    bool allocated_last_cude_fg_ = false;
    if ((sample_x % 2 == 1) && (sample_y % 2 == 1) && (sample_z % 2 == 1)) {
        if (sample_x > 1 && sample_y > 1) {
            last_cube_fg_ = reinterpret_cast<unsigned char*>(d_img_labels_.data + (sample_z - 1) * d_img_labels_.stepz + (sample_y - 2) * d_img_labels_.stepy) + sample_x - 2;
        }
        else if (sample_x > 1 && sample_z > 1) {
            last_cube_fg_ = reinterpret_cast<unsigned char*>(d_img_labels_.data + (sample_z - 2) * d_img_labels_.stepz + (sample_y - 1) * d_img_labels_.stepy) + sample_x - 2;
        }
        else if (sample_y > 1 && sample_z > 1) {
            last_cube_fg_ = reinterpret_cast<unsigned char*>(d_img_labels_.data + (sample_z - 2) * d_img_labels_.stepz + (sample_y - 2) * d_img_labels_.stepy) + sample_x - 1;
        }
        else {
            cudaMalloc(&last_cube_fg_, sizeof(unsigned char));
            allocated_last_cude_fg_ = true;
        }
    }
    dim3 grid_size_ = dim3(((sample_x + 1) / 2 + BLOCK_X - 1) / BLOCK_X, ((sample_y + 1) / 2 + BLOCK_Y - 1) / BLOCK_Y, ((sample_z + 1) / 2 + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block_size_ = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);

    // start CCL
    InitLabeling << <grid_size_, block_size_ >> > (d_img_labels_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Merge << <grid_size_, block_size_ >> > (d_img_, d_img_labels_, last_cube_fg_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    PathCompression << <grid_size_, block_size_ >> > (d_img_labels_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    FinalLabeling << <grid_size_, block_size_ >> > (d_img_, d_img_labels_, last_cube_fg_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (allocated_last_cude_fg_) {
        cudaFree(last_cube_fg_);
    }

    // download the data to the mlflow
    device_download << <grid1, threads1 >> > (mlflow, d_img_labels_, d_img_, sample_x, sample_y, sample_z);

    // renumber the label
    renumber_0 << <grid1, threads1 >> > (sample_x, sample_y, sample_z, d_ptr_1, d_img_labels_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    thrust::sort(d_bubble_map.begin(), d_bubble_map.end());
    thrust::device_vector<int>::iterator new_end;
    new_end = thrust::unique(d_bubble_map.begin(), d_bubble_map.end());
    d_bubble_map.erase(new_end, d_bubble_map.end());
    int* d_ptr_2 = thrust::raw_pointer_cast(d_bubble_map.data());
    if (d_bubble_map.size() > 1)
    {
        renumber_1 << <grid1, threads1 >> > (sample_x, sample_y, sample_z, d_bubble_map.size(), d_ptr_2, d_img_labels_);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        renumber_2 << <grid1, threads1 >> > (mlflow, sample_x, sample_y, sample_z, d_bubble_map.size(), d_ptr_2, d_img_labels_);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
    cudaDeviceSynchronize();
    cudaFree(d_img_labels);
    cudaFree(d_imgs);
}

void connectedComponentLabeling(mrFlow3D* mlflow, int numCols, int numRows, int numDepths)
{
    PerformLabeling(mlflow, numCols, numRows, numDepths);
}
