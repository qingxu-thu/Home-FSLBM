# Kinetic Free-Surface Flows and Foams with Sharp Interfaces

## Project Overview

This is the code release for "Kinetic Free-Surface Flows and Foams with Sharp Interfaces". The project implements HOME (Li et al. 2023) based free-surface flow and foam simulation based on the Lattice Boltzmann Method (LBM).

## Key Features

- 2D foam drained simulation
- 3D pouring water 

NOTE: In the current version, we do not adopt the reuse of memory for bubble index for better understanding. We use tag, previous_tag, previous_merge_tag, input_matrix and label_matrix.
Actually, the reuse is simple. Actully, the input_matrix and label_matrix are only needed in
the CCL processes, where previous_tag and previous_merge_tag have already finished their jobs.
Therefore we only need three of them.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU (CUDA compatible, recommended compute capability 6.0 or higher)

### Software Dependencies

#### Required Dependencies
- **CMake**: Version 3.10 or higher
- **C++ Compiler**: Compiler supporting C++14 standard (GCC 7+, Clang 5+, MSVC 2017+)
- **CUDA Toolkit**: Version 11.8 or higher (for GPU acceleration)
- **OpenCV**: For image processing and visualization

#### Optional Dependencies
- **Eigen**: Linear algebra library (included in 3rdParty directory)

We only test our code in Linux (Ubuntu 18.04+) with 3090 and A100 GPU cards, for Windows version, welcome for the testing and we can discuss the problem in the issues.


## Build and Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd FSLBM
```

### 2. Build the Project
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

### 2D Simulation (with the extrance of testMrLBM2D)
```bash
./lbm_flow_proj  # Run 2D free-surface flow simulation
```

### 3D Simulation (with the extrance of testMrLBM3D_bubble)
```bash
./lbm_flow_proj  # Run 3D bubble dynamics simulation
```


## License

This project is licensed under the [License Type] License. See the LICENSE file for details.


## Acknowledgments

We thank all researchers and developers who have contributed to this project. The CCL part is from CUDA_CCL(2D) (https://github.com/FolkeV/CUDA_CCL/tree/master) and BUF in YACCLAB(3D) (https://github.com/prittt/YACCLAB). The 2D Possion disk sampling is from poisson-disk-generator (https://github.com/corporateshark/poisson-disk-generator).



## Citation

If you use this code in your research, please cite the following reference:

```
@article{kinetic_free_surface_2024,
  title={Kinetic Free-Surface Flows and Foams with Sharp Interfaces},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  volume={[Volume Number]},
  pages={[Page Range]}
}
```

## Reference

[1] Li W, Wang T, Pan Z, et al. High-order moment-encoded kinetic simulation of turbulent flows[J]. ACM Transactions on Graphics (TOG), 2023, 42(6): 1-13.

[2] S. Allegretti, F. Bolelli, C. Grana, "Optimized Block-Based Algorithms to Label Connected Components on GPUs," in IEEE Transactions on Parallel and Distributed Systems, 2019.
