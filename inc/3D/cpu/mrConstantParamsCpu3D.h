#pragma once

#ifndef MRCONSTANTPARAMSCPU3DH_
#define MRCONSTANTPARAMSCPU3DH_

#define def_c 0.57735027f //1/sqrt(3)
#define def_w0 (1.0f/3.375f) // center (0)
#define def_ws (1.0f/13.5f) // straight (1-6)
#define def_we (1.0f/54.0f) // edge (7-18)
#define def_wc (1.0f/216.0f) // corner (19-26)
#define def_6_sigma 6.0f * 0.00f

const float ex3d_cpu[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
const float ey3d_cpu[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
const float ez3d_cpu[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };
const int index3dInv_cpu[27] = { 0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17,20,19,22,21,24,23,26,25 };

const float w3d_cpu[27] = { def_w0, def_ws, def_ws, def_ws, def_ws, def_ws, def_ws,
		def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we,
		def_wc, def_wc, def_wc, def_wc, def_wc, def_wc, def_wc, def_wc };
const float as2_cpu = 3.0;
const float cs2_cpu = 1.0 / 3.0;

#endif // !MRCONSTANTPARAMSCPU3DH_
