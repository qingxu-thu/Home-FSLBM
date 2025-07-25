#pragma once
#ifndef _MRCONSTANTPARAMSGPU3DH_
#define _MRCONSTANTPARAMSGPU3DH_



#define def_c  0.57735027f
#define def_w0 (1.0f/3.375f) // center (0)
#define def_ws (1.0f/13.5f) // straight (1-6)
#define def_we (1.0f/54.0f) // edge (7-18)
#define def_wc (1.0f/216.0f) // corner (19-26)
#define def_6_sigma 6.0f * 4e-3f   // 3e-3
#define K_h 1e-3

__constant__ float d3q7_w[7] = { 1.f / 4.f, 1.f / 8.f,1.f / 8.f, 1.f / 8.f, 1.f / 8.f, 1.f / 8.f, 1.f / 8.f };
//                                  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
__constant__ float ex3d_gpu[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
__constant__ float ey3d_gpu[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
__constant__ float ez3d_gpu[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };
__constant__ int index3dInv_gpu[27] = { 0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17,20,19,22,21,24,23,26,25 };
__constant__ float w3d_gpu[27] = { def_w0,def_ws, def_ws, def_ws, def_ws, def_ws, def_ws,
		def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we,
		def_wc, def_wc, def_wc, def_wc, def_wc, def_wc, def_wc, def_wc };
__constant__ float as2 = 3.0f;
__constant__ float cs2 = 1.0f / 3.0f;


#endif // !_MRCONSTANTPARAMSGPU3DH_
