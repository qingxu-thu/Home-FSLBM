#pragma once
#ifndef MRCONSTANTPARAMSCPU2DH_
#define MRCONSTANTPARAMSCPU2DH_
#define def_c  0.57735027f
#define def_w0 (1.0f/2.25f) // center (0)
#define def_ws (1.0f/9.0f) // straight (1-4)
#define def_we (1.0f/36.0f) // edge (5-8)
#define def_6_sigma 6.0f * 2.5e-7


//#define fma(a,b,c) a*b+c
//#define cb(x) x*x*x
//#define sq(x) x*x
//#define cbrt(x) powf(x, 1.0/3.0)
//#define clamp(x, a, b) fmin(fmax(x, a), b)

//                          0  1  2  3  4  5  6  7  8  
const float ex2d_cpu[9] = { 0, 1,-1, 0, 0, 1,-1, 1,-1 };
const float ey2d_cpu[9] = { 0, 0, 0, 1,-1, 1,-1,-1, 1 };
const float ez2d_cpu[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
const int index2dInv_cpu[9] = { 0,2,1,4,3,6,5,8,7 };
const float w2d_cpu[9] = { def_w0, def_ws, def_ws, def_ws, def_ws, def_we, def_we, def_we, def_we };

const float as2_cpu = 3.0;
const float cs2_cpu = 1.0 / 3.0;
#endif // !MRCONSTANTPARAMSCPU3DH_
