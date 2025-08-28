#pragma once
 
#include <omp.h>
/* ------------------------------ MEMORY SIZE ------------------------------ */
#define BLOCK_NX 4
#define BLOCK_NY 4
#define BLOCK_NZ 4

#define BLOCK_X 8
#define BLOCK_Y 4
#define BLOCK_Z 4

#define BLOCK_LBM_SIZE (BLOCK_NX * BLOCK_NY * BLOCK_NZ)

#define QF  9         // number of velocities on each face