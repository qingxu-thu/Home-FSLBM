#pragma once
#ifndef _MRUTILFUNCGU3DH_
#define _MRUTILFUNCGU3DH_

#include "cuda_runtime.h"
//#include "../../../lw_core_win/mlCoreWinHeader.h"
#include "../../../common/mlCoreWin.h"
#include "../../../common/mlLatticeNode.h"
#include "mrLbmSolverGpu3D.h"
#include "mrConstantParamsGpu3D.h"

class mrUtilFuncGpu3D
{
public:
	MLFUNC_TYPE void calculate_rho_u(REAL* pop, REAL& rho, REAL& ux, REAL& uy, REAL& uz);
	MLFUNC_TYPE void calculate_forcing_terms(REAL ux, REAL uy, REAL uz, REAL fx, REAL fy, REAL fz, REAL* Fin);
	MLFUNC_TYPE void calculate_f_eq(const float rho, float ux, float uy, float  uz, float* feq);
	MLFUNC_TYPE void calculate_g_eq(const float rho, float ux, float uy, float  uz, float* feq);
	MLFUNC_TYPE float calculate_phi(const float rhon, const float massn, const unsigned char flagsn);
	MLFUNC_TYPE float calculate_curvature(const float* phit);
	MLFUNC_TYPE float3 calculate_normal(const float* phit);
	MLFUNC_TYPE float plic_cube(const float V0, const float3 n);

	MLFUNC_TYPE void mlCalDistributionFourthOrderD3Q27AtIndex(
		float rho,
		float ux, float uy, float uz,
		float pixx, float pixy, float pixz, float piyy, float piyz, float pizz,
		int i, float& f_out);

	MLFUNC_TYPE void mlConvertCmrMoment_d3q7(float ux, float uy, float uz, float * node_in_out);
	MLFUNC_TYPE void mlConvertCmrF_d3q7(float U, float V, float W, float* node_in_out);

	MLFUNC_TYPE void mlGetPIAfterCollision(
		float R, float U, float V, float W, float Fx, float Fy, float Fz, float omega, /*MlLatticeNodeD3Q27 node_in_out,*/
		float& pixx_t45,
		float& pixy_t90,
		float& pixz_t90,
		float& piyy_t45,
		float& piyz_t90,
		float& pizz_t45
	);
};

inline MLFUNC_TYPE float rsqrt_(const float x) {
	return 1.0f / sqrt(x);
}

inline MLFUNC_TYPE float sq(const float x) {
	return x * x;
}
inline MLFUNC_TYPE float cb(const float x) {
	return x * x * x;
}
inline MLFUNC_TYPE float sign(const float x) {
	return x >= 0.0f ? 1.0f : -1.0f;
}

inline MLFUNC_TYPE int imin(int a, int b)
{
	return a < b ? a : b;

}


inline MLFUNC_TYPE float clamp(float x, float a, float b)
{
	return fmin(fmax(x, a), b);
}
//inline MLFUNC_TYPE float cbrt(float x)
//{
//	return powf(x, 1.0 / 3.0);
//}

inline MLFUNC_TYPE float3 cross(const float3& v1, const float3& v2) {
	return { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x };
}

inline MLFUNC_TYPE float dot(const float3& v1, const float3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
inline MLFUNC_TYPE float length(const float3& v) {
	return sqrt(sq(v.x) + sq(v.y) + sq(v.z));
}
inline MLFUNC_TYPE float3 normalize(const float3& v, const float s = 1.0f) {
	if (length(v) == 0) {
		return { 0,0,0 };
	}
	else
	{
	const float n = s / length(v);
	return { v.x * n, v.y * n, v.z * n
	};
	}
}

inline MLFUNC_TYPE float3 normalizing_clamp(const float3 u, float n_lim)
{
	float n = length(u);
	if (n > n_lim) {
		return normalize(u, n_lim);
	}
	else {
		return u;
	}
}



inline MLFUNC_TYPE REAL plic_cube_reduced(const REAL V, const REAL n1, const REAL n2, const REAL n3) { // optimized solution from SZ and Kawano, source: https://doi.org/10.3390/computation10020021
    const REAL n12 = n1 + n2, n3V = n3 * V;
    if (n12 <= 2.0 * n3V) return n3V + 0.5 * n12; // case (5)
    const REAL sqn1 = sq(n1), n26 = 6.0 * n2, v1 = sqn1 / n26; // after case (5) check n2>0 is true
    if (v1 <= n3V && n3V < v1 + 0.5 * (n2 - n1)) return 0.5 * (n1 + sqrt(sqn1 + 8.0 * n2 * (n3V - v1))); // case (2)
    const REAL V6 = n1 * n26 * n3V;
    if (n3V < v1) return cbrt(V6); // case (1)
    const REAL v3 = n3 < n12 ? (sq(n3) * (3.0 * n12 - n3) + sqn1 * (n1 - 3.0 * n3) + sq(n2) * (n2 - 3.0 * n3)) / (n1 * n26) : 0.5 * n12; // after case (2) check n1>0 is true
    const REAL sqn12 = sqn1 + sq(n2), V6cbn12 = V6 - cb(n1) - cb(n2);
    const bool case34 = n3V < v3; // true: case (3), false: case (4)
    const REAL a = case34 ? V6cbn12 : 0.5 * (V6cbn12 - cb(n3));
    const REAL b = case34 ? sqn12 : 0.5 * (sqn12 + sq(n3));
    const REAL c = case34 ? n12 : 0.5;
    const REAL t = sqrt(sq(c) - b);
    return c - 2.0 * t * sinf(0.33333334 * asinf((cb(c) - 0.5 * a - 1.5 * b * c) / cb(t)));
}




inline MLFUNC_TYPE void lu_solve(float* M, float* x, float* b, const int N, const int Nsol)
{ // solves system of N linear equations M*x=b within dimensionality Nsol<=N
	for (int i = 0; i < Nsol; i++) { // decompose M in M=L*U
		for (int j = i + 1; j < Nsol; j++) {
			M[N * j + i] /= M[N * i + i];
			for (int k = i + 1; k < Nsol; k++) M[N * j + k] -= M[N * j + i] * M[N * i + k];
		}
	}
	for (int i = 0; i < Nsol; i++) { // find solution of L*y=b
		x[i] = b[i];
		for (int k = 0; k < i; k++) x[i] -= M[N * i + k] * x[k];
	}
	for (int i = Nsol - 1; i >= 0; i--) { // find solution of U*x=y
		for (int k = i + 1; k < Nsol; k++) x[i] -= M[N * i + k] * x[k];
		x[i] /= M[N * i + i];
	}
}

inline MLFUNC_TYPE REAL mrUtilFuncGpu3D::plic_cube(const REAL V0, const float3 n) { // unit cube - plane intersection: volume V0 in [0,1], normal vector n -> plane offset d0
    const REAL ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z), V = 0.5 - fabsf(V0 - 0.5), l = ax + ay + az; // eliminate symmetry cases, normalize n using L1 norm
    const REAL n1 = fmin(fmin(ax, ay), az) / l;
    const REAL n3 = fmax(fmax(ax, ay), az) / l;
    const REAL n2 = fdimf(1.0, n1 + n3); // ensure n2>=0
    const REAL d = plic_cube_reduced(V, n1, n2, n3); // calculate PLIC with reduced symmetry
    return l * copysignf(0.5 - d, V0 - 0.5); // rescale result and apply symmetry for V0>0.5
}



inline MLFUNC_TYPE void mrUtilFuncGpu3D::mlCalDistributionFourthOrderD3Q27AtIndex(
	float rho,
	float ux, float uy, float uz,
	float pixx, float pixy, float pixz, float piyy, float piyz, float pizz,
	int i, float& f_out)
{


	float A0 = rho;
	float Ax = ux * A0;
	float Ay = uy * A0;
	float Az = uz * A0;

	float Axx = rho * pixx;
	float Ayy = rho * piyy;
	float Azz = rho * pizz;
	float Axy = rho * pixy;
	float Axz = rho * pixz;
	float Ayz = rho * piyz;

	float Axxy = -2 * rho * uy * ux * ux + 2 * Axy * ux + Axx * uy;
	float Axyy = -2 * rho * ux * uy * uy + 2 * Axy * uy + Ayy * ux;
	float Axxz = -2 * rho * uz * ux * ux + 2 * Axz * ux + Axx * uz;
	float Axzz = -2 * rho * ux * uz * uz + 2 * Axz * uz + Azz * ux;
	float Ayyz = -2 * rho * uz * uy * uy + 2 * Ayz * uy + Ayy * uz;
	float Ayzz = -2 * rho * uy * uz * uz + 2 * Ayz * uz + Azz * uy;
	float Axyz = Axz * uy + Ayz * ux + Axy * uz - 2 * rho * ux * uy * uz;


	float W0 = 8.0 / 27;        // weight dist 0 population (0, 0, 0)
	float W1 = 2.0 / 27;        // weight dist 1 populations (1, 0, 0)
	float W2 = 1.0 / 54;        // weight dist 2 populations (1, 1, 0)
	float W3 = 1.0 / 216;       // weight dist 3 populations (1, 1, 1)

	float Ax_t3 = Ax * 3;
	float Ay_t3 = Ay * 3;
	float Az_t3 = Az * 3;

	float Axx_t3 = 3 * Axx;
	float Ayy_t3 = 3 * Ayy;
	float Azz_t3 = 3 * Azz;
	float Axy_t9 = 9 * Axy;
	float Axz_t9 = 9 * Axz;
	float Ayz_t9 = 9 * Ayz;

	float Axxy_t9 = Axxy * 9;
	float Axyy_t9 = Axyy * 9;
	float Axzz_t9 = Axzz * 9;
	float Ayzz_t9 = Ayzz * 9;
	float Axxz_t9 = Axxz * 9;
	float Ayyz_t9 = Ayyz * 9;
	float Axyz_t27 = Axyz * 27;

	float com0 = A0 - (Axx_t3) * 0.5 - (Ayy_t3) * 0.5 - (Azz_t3) * 0.5;
	switch (i)
	{
	case 0:
		f_out = W0 * (com0); break;
	case 1:
		f_out = W1 * (com0 + Ax_t3 + 1.5 * Axx_t3 - (Axyy_t9) * 0.5 - (Axzz_t9) * 0.5); break;
	case 2:
		f_out = W1 * (com0 - Ax_t3 + 1.5 * Axx_t3 + (Axyy_t9) * 0.5 + (Axzz_t9) * 0.5); break;
	case 3:
		f_out = W1 * (com0 + Ay_t3 + 1.5 * Ayy_t3 - (Axxy_t9) * 0.5 - (Ayzz_t9) * 0.5); break;
	case 4:
		f_out = W1 * (com0 - Ay_t3 + 1.5 * Ayy_t3 + (Ayzz_t9) * 0.5 + (Axxy_t9) * 0.5); break;
	case 5:
		f_out = W1 * (com0 + Az_t3 + 1.5 * Azz_t3 - (Axxz_t9) * 0.5 - (Ayyz_t9) * 0.5); break;
	case 6:
		f_out = W1 * (com0 - Az_t3 + 1.5 * Azz_t3 + (Axxz_t9) * 0.5 + (Ayyz_t9) * 0.5); break;

	case 7:
		f_out = W2 * (A0 + Ax_t3 + Ay_t3 + Axx_t3 + Ayy_t3 - (Azz_t3) * 0.5 + Axy_t9 + Axxy_t9 + Axyy_t9 - (Axzz_t9) * 0.5 - (Ayzz_t9) * 0.5); break;
	case 8:
		f_out = W2 * (A0 - Ax_t3 - Ay_t3 + Axx_t3 + Ayy_t3 - (Azz_t3) * 0.5 + Axy_t9 - Axxy_t9 - Axyy_t9 + (Axzz_t9) * 0.5 + (Ayzz_t9) * 0.5); break;
	case 9:
		f_out = W2 * (A0 + Ax_t3 + Az_t3 + Axx_t3 - (Ayy_t3) * 0.5 + Azz_t3 + Axz_t9 + Axxz_t9 - (Axyy_t9) * 0.5 + Axzz_t9 - (Ayyz_t9) * 0.5); break;
	case 10:
		f_out = W2 * (A0 - Ax_t3 - Az_t3 + Axx_t3 - (Ayy_t3) * 0.5 + Azz_t3 + Axz_t9 - Axxz_t9 + (Axyy_t9) * 0.5 - Axzz_t9 + (Ayyz_t9) * 0.5); break;
	case 11:
		f_out = W2 * (A0 + Ay_t3 + Az_t3 - (Axx_t3) * 0.5 + Ayy_t3 + Azz_t3 + Ayz_t9 - (Axxy_t9) * 0.5 - (Axxz_t9) * 0.5 + Ayyz_t9 + Ayzz_t9); break;
	case 12:
		f_out = W2 * (A0 - Ay_t3 - Az_t3 - (Axx_t3) * 0.5 + Ayy_t3 + Azz_t3 + Ayz_t9 + (Axxy_t9) * 0.5 + (Axxz_t9) * 0.5 - Ayyz_t9 - Ayzz_t9); break;
	case 13:
		f_out = W2 * (A0 + Ax_t3 - Ay_t3 + Axx_t3 + Ayy_t3 - (Azz_t3) * 0.5 - Axy_t9 - Axxy_t9 + Axyy_t9 - (Axzz_t9) * 0.5 + (Ayzz_t9) * 0.5); break;
	case 14:
		f_out = W2 * (A0 - Ax_t3 + Ay_t3 + Axx_t3 + Ayy_t3 - (Azz_t3) * 0.5 - Axy_t9 + Axxy_t9 - Axyy_t9 + (Axzz_t9) * 0.5 - (Ayzz_t9) * 0.5); break;
	case 15:
		f_out = W2 * (A0 + Ax_t3 - Az_t3 + Axx_t3 - (Ayy_t3) * 0.5 + Azz_t3 - Axz_t9 - Axxz_t9 - (Axyy_t9) * 0.5 + Axzz_t9 + (Ayyz_t9) * 0.5); break;
	case 16:
		f_out = W2 * (A0 - Ax_t3 + Az_t3 + Axx_t3 - (Ayy_t3) * 0.5 + Azz_t3 - Axz_t9 + Axxz_t9 + (Axyy_t9) * 0.5 - Axzz_t9 - (Ayyz_t9) * 0.5); break;
	case 17:
		f_out = W2 * (A0 + Ay_t3 - Az_t3 - (Axx_t3) * 0.5 + Ayy_t3 + Azz_t3 - Ayz_t9 - (Axxy_t9) * 0.5 + (Axxz_t9) * 0.5 - Ayyz_t9 + Ayzz_t9); break;
	case 18:
		f_out = W2 * (A0 - Ay_t3 + Az_t3 - (Axx_t3) * 0.5 + Ayy_t3 + Azz_t3 - Ayz_t9 + (Axxy_t9) * 0.5 - (Axxz_t9) * 0.5 + Ayyz_t9 - Ayzz_t9); break;

	case 19:
		f_out = W3 * (A0 + Ax_t3 + Axx_t3 + Axxy_t9 + Axxz_t9 + Axy_t9 + Axyy_t9 + Axyz_t27 + Axz_t9 + Axzz_t9 + Ay_t3 + Ayy_t3 + Ayyz_t9 + Ayz_t9 + Ayzz_t9 + Az_t3 + Azz_t3); break;
	case 20:
		f_out = W3 * (A0 - Ax_t3 + Axx_t3 - Axxy_t9 - Axxz_t9 + Axy_t9 - Axyy_t9 - Axyz_t27 + Axz_t9 - Axzz_t9 - Ay_t3 + Ayy_t3 - Ayyz_t9 + Ayz_t9 - Ayzz_t9 - Az_t3 + Azz_t3); break;
	case 21:
		f_out = W3 * (A0 + Ax_t3 + Axx_t3 + Axxy_t9 - Axxz_t9 + Axy_t9 + Axyy_t9 - Axyz_t27 - Axz_t9 + Axzz_t9 + Ay_t3 + Ayy_t3 - Ayyz_t9 - Ayz_t9 + Ayzz_t9 - Az_t3 + Azz_t3); break;
	case 22:
		f_out = W3 * (A0 - Ax_t3 + Axx_t3 - Axxy_t9 + Axxz_t9 + Axy_t9 - Axyy_t9 + Axyz_t27 - Axz_t9 - Axzz_t9 - Ay_t3 + Ayy_t3 + Ayyz_t9 - Ayz_t9 - Ayzz_t9 + Az_t3 + Azz_t3); break;
	case 23:
		f_out = W3 * (A0 + Ax_t3 + Axx_t3 - Axxy_t9 + Axxz_t9 - Axy_t9 + Axyy_t9 - Axyz_t27 + Axz_t9 + Axzz_t9 - Ay_t3 + Ayy_t3 + Ayyz_t9 - Ayz_t9 - Ayzz_t9 + Az_t3 + Azz_t3); break;
	case 24:
		f_out = W3 * (A0 - Ax_t3 + Axx_t3 + Axxy_t9 - Axxz_t9 - Axy_t9 - Axyy_t9 + Axyz_t27 + Axz_t9 - Axzz_t9 + Ay_t3 + Ayy_t3 - Ayyz_t9 - Ayz_t9 + Ayzz_t9 - Az_t3 + Azz_t3); break;
	case 25:
		f_out = W3 * (A0 - Ax_t3 + Axx_t3 + Axxy_t9 + Axxz_t9 - Axy_t9 - Axyy_t9 - Axyz_t27 - Axz_t9 - Axzz_t9 + Ay_t3 + Ayy_t3 + Ayyz_t9 + Ayz_t9 + Ayzz_t9 + Az_t3 + Azz_t3); break;
	case 26:
		f_out = W3 * (A0 + Ax_t3 + Axx_t3 - Axxy_t9 - Axxz_t9 - Axy_t9 + Axyy_t9 + Axyz_t27 - Axz_t9 + Axzz_t9 - Ay_t3 + Ayy_t3 - Ayyz_t9 + Ayz_t9 - Ayzz_t9 - Az_t3 + Azz_t3); break;
	}
}





inline MLFUNC_TYPE void mrUtilFuncGpu3D::calculate_rho_u(REAL* pop, REAL& rho, REAL& ux, REAL& uy, REAL& uz)
{

	for (int i = 0; i < 27; i++)
		rho += pop[i];
	rho += 1.0f;
	for (int i = 0; i < 27; i++)
	{
		ux += pop[i] * ex3d_gpu[i];
		uy += pop[i] * ey3d_gpu[i];
		uz += pop[i] * ez3d_gpu[i];
	}
	ux = ux / rho;
	uy = uy / rho;
	uz = uz / rho;
}



// fix the f_eq as the original ones
inline MLFUNC_TYPE void mrUtilFuncGpu3D::calculate_f_eq(const float rho, float ux, float uy, float  uz, float* feq)
{
	// printf("rho: %f, ux: %f, uy: %f, uz: %f\n", rho, ux, uy, uz);
	const float c3 = -3.0f * (sq(ux) + sq(uy) + sq(uz));
	// printf("c3: %f\n", c3);
	const float	rhom1 = rho - 1.0f; // c3 = -2*sq(u)/(2*sq(c)), rhom1 is arithmetic optimization to minimize digit extinction
	ux *= 3.0f;
	uy *= 3.0f;
	uz *= 3.0f;
	feq[0] = def_w0 * fma(rho, 0.5f * c3, rhom1);
	const float u0 = ux + uy, u1 = ux + uz, u2 = uy + uz, u3 = ux - uy, u4 = ux - uz, u5 = uy - uz, u6 = ux + uy + uz, u7 = ux + uy - uz, u8 = ux - uy + uz, u9 = -ux + uy + uz;
	const float rhos = def_ws * rho, rhoe = def_we * rho, rhoc = def_wc * rho, rhom1s = def_ws * rhom1, rhom1e = def_we * rhom1, rhom1c = def_wc * rhom1;
	feq[1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
	feq[7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[8] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	feq[9] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[10] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +0+ -0-
	feq[11] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), u2), rhom1e); feq[12] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), -u2), rhom1e); // 0++ 0--
	feq[13] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), u3), rhom1e); feq[14] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), -u3), rhom1e); // +-0 -+0
	feq[15] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), u4), rhom1e); feq[16] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), -u4), rhom1e); // +0- -0+
	feq[17] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), u5), rhom1e); feq[18] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), -u5), rhom1e); // 0+- 0-+
	feq[19] = fma(rhoc, fma(0.5f, fma(u6, u6, c3), u6), rhom1c); feq[20] = fma(rhoc, fma(0.5f, fma(u6, u6, c3), -u6), rhom1c); // +++ ---
	feq[21] = fma(rhoc, fma(0.5f, fma(u7, u7, c3), u7), rhom1c); feq[22] = fma(rhoc, fma(0.5f, fma(u7, u7, c3), -u7), rhom1c); // ++- --+
	feq[23] = fma(rhoc, fma(0.5f, fma(u8, u8, c3), u8), rhom1c); feq[24] = fma(rhoc, fma(0.5f, fma(u8, u8, c3), -u8), rhom1c); // +-+ -+-
	feq[25] = fma(rhoc, fma(0.5f, fma(u9, u9, c3), u9), rhom1c); feq[26] = fma(rhoc, fma(0.5f, fma(u9, u9, c3), -u9), rhom1c); // -++ +--

	//float uznk = 0.0;
	//float rhok = 0.0;
	//for (int i = 0; i < 27; i++)
	//{
	//	rhok += feq[i];
	//	//printf("f_eq[%d]: %f\n", i, f_eq[i]);
	//	uznk += feq[i] * ez3d_gpu[i];
	//}
	//uznk /= (rhok + 1.0f);
	///*if (uzn != uznk)*/
	//printf("uzn_before: %f, uzn_after: %f, rho_before %f, rho_after %f\n", uz, uznk, rho, rhok);


}

inline MLFUNC_TYPE void mrUtilFuncGpu3D::calculate_g_eq(const float rho, float ux, float uy, float  uz, float* feq)
{
	// g is updated by the D2Q5

	// const float c3 = -3.0f * (sq(ux) + sq(uy) + sq(uz));
	// printf("c3: %f\n", c3);
	const float	rhom1 = rho; // c3 = -2*sq(u)/(2*sq(c)), rhom1 is arithmetic optimization to minimize digit extinction
	ux *= 4.0f;
	uy *= 4.0f;
	uz *= 4.0f;

	feq[0] = d3q7_w[0] * rhom1;

	//const float rhos = d3q7_w[1] * rho, rhom1s = d3q7_w[1] * rhom1;
	feq[1] = d3q7_w[1] * (1 + (float)ex3d_gpu[1] * ux + (float)ey3d_gpu[1] * uy + (float)ez3d_gpu[1] * uz) * rhom1;
	feq[2] = d3q7_w[2] * (1 + (float)ex3d_gpu[2] * ux + (float)ey3d_gpu[2] * uy + (float)ez3d_gpu[2] * uz) * rhom1;
	feq[3] = d3q7_w[3] * (1 + (float)ex3d_gpu[3] * ux + (float)ey3d_gpu[3] * uy + (float)ez3d_gpu[3] * uz) * rhom1;
	feq[4] = d3q7_w[4] * (1 + (float)ex3d_gpu[4] * ux + (float)ey3d_gpu[4] * uy + (float)ez3d_gpu[4] * uz) * rhom1;
	feq[5] = d3q7_w[5] * (1 + (float)ex3d_gpu[5] * ux + (float)ey3d_gpu[5] * uy + (float)ez3d_gpu[5] * uz) * rhom1;
	feq[6] = d3q7_w[6] * (1 + (float)ex3d_gpu[6] * ux + (float)ey3d_gpu[6] * uy + (float)ez3d_gpu[6] * uz) * rhom1;

	//need to change
}

inline MLFUNC_TYPE void mrUtilFuncGpu3D::calculate_forcing_terms(REAL ux, REAL uy, REAL uz, REAL fx, REAL fy, REAL fz, REAL* Fin)
{
	const REAL uF = -0.33333334f * fma(ux, fx, fma(uy, fy, uz * fz));
	Fin[0] = 9.0f * def_w0 * uF; // 000 (identical for all velocity sets)
	for (int i = 1; i < 27; i++) { // loop is entirely unrolled by compiler, no unnecessary FLOPs are happening
		Fin[i] = 9.0f * w3d_gpu[i] * fma(ex3d_gpu[i] * fx + ey3d_gpu[i] * fy
			+ ez3d_gpu[i] * fz, ex3d_gpu[i] * ux
			+ ey3d_gpu[i] * uy +
			ez3d_gpu[i] * uz + 0.33333334f, uF);
	}
}
inline MLFUNC_TYPE float mrUtilFuncGpu3D::calculate_phi(const float rhon, const float massn, const unsigned char flagsn)
{
	return flagsn & TYPE_F ? 1.0f : flagsn & TYPE_I ? rhon > 0.0f ? clamp(massn / rhon, 0.0f, 1.0f) : 0.5f : 0.0f;
}


inline MLFUNC_TYPE float3 mrUtilFuncGpu3D::calculate_normal(const float* phit)
{
	float phij[27];
	// get_remaining_neighbor_phij(n, phit, phi, phij); // complete neighborhood from whatever velocity set is selected to D3Q27
	for (int i = 0; i < 27; i++)
		phij[i] = phit[index3dInv_gpu[i]];

	//const float3 bz = calculate_normal_py(phij); // new coordinate system: bz is normal to surface, bx and by are tangent to surface

	float3 bz;
	bz.x = 4.0f * (phij[2] - phij[1]) + 2.0f * (phij[8] - phij[7] + phij[10] - phij[9] + phij[14] - phij[13] + phij[16] - phij[15]) + phij[20] - phij[19] + phij[22] - phij[21] + phij[24] - phij[23] + phij[25] - phij[26];
	bz.y = 4.0f * (phij[4] - phij[3]) + 2.0f * (phij[8] - phij[7] + phij[12] - phij[11] + phij[13] - phij[14] + phij[18] - phij[17]) + phij[20] - phij[19] + phij[22] - phij[21] + phij[23] - phij[24] + phij[26] - phij[25];
	bz.z = 4.0f * (phij[6] - phij[5]) + 2.0f * (phij[10] - phij[9] + phij[12] - phij[11] + phij[15] - phij[16] + phij[17] - phij[18]) + phij[20] - phij[19] + phij[21] - phij[22] + phij[24] - phij[23] + phij[26] - phij[25];
	//printf("bz.x: %f, bz.y: %f, bz.z: %f\n", bz.x, bz.y, bz.z);
	bz = normalize(bz);
	return bz;
}

inline MLFUNC_TYPE float mrUtilFuncGpu3D::calculate_curvature(const float* phit)
{
	float phij[27];
	// get_remaining_neighbor_phij(n, phit, phi, phij); // complete neighborhood from whatever velocity set is selected to D3Q27
	for (int i = 0; i < 27; i++)
		phij[i] = phit[index3dInv_gpu[i]];

	//const float3 bz = calculate_normal_py(phij); // new coordinate system: bz is normal to surface, bx and by are tangent to surface

	float3 bz;
	bz.x = 4.0f * (phij[2] - phij[1]) + 2.0f * (phij[8] - phij[7] + phij[10] - phij[9] + phij[14] - phij[13] + phij[16] - phij[15]) + phij[20] - phij[19] + phij[22] - phij[21] + phij[24] - phij[23] + phij[25] - phij[26];
	bz.y = 4.0f * (phij[4] - phij[3]) + 2.0f * (phij[8] - phij[7] + phij[12] - phij[11] + phij[13] - phij[14] + phij[18] - phij[17]) + phij[20] - phij[19] + phij[22] - phij[21] + phij[23] - phij[24] + phij[26] - phij[25];
	bz.z = 4.0f * (phij[6] - phij[5]) + 2.0f * (phij[10] - phij[9] + phij[12] - phij[11] + phij[15] - phij[16] + phij[17] - phij[18]) + phij[20] - phij[19] + phij[21] - phij[22] + phij[24] - phij[23] + phij[26] - phij[25];
	bz = normalize(bz);

	const float3 rn = { 0.56270900f, 0.32704452f, 0.75921047f }; // random normalized vector that is just by random chance not collinear with bz
	const float3 by = normalize(cross(bz, rn)); // normalize() is necessary here because bz and rn are not perpendicular
	const float3 bx = cross(by, bz);

	int number = 0; // number of neighboring interface points

	float3 p[24]; // number of neighboring interface points is less or equal than than 26 minus 1 gas and minus 1 fluid point = 24
	const float center_offset = plic_cube(phij[0], bz); // calculate z-offset PLIC of center point only once

	for (int i = 1; i < 27; i++) { // iterate over neighbors, no loop unrolling here (50% better perfoemance without loop unrolling)
		if (phij[i] > 0.0f && phij[i] < 1.0f) { // limit neighbors to interface cells
			// might be wrong
			const float3 ei = { (float)ex3d_gpu[i], (float)ey3d_gpu[i], (float)ez3d_gpu[i] }; // assume neighbor normal vector is the same as center normal vector
			const float offset = plic_cube(phij[i], bz) - center_offset;

			p[number] = { dot(ei, bx), dot(ei, by), dot(ei, bz) + offset }; // do coordinate system transformation into (x, y, f(x,y)) and apply PLIC pffsets
			number += 1;
		}
	}
	float M[25], x[5] = { 0.0f,0.0f,0.0f,0.0f,0.0f }, b[5] = { 0.0f,0.0f,0.0f,0.0f,0.0f };
	for (int i = 0; i < 25; i++) M[i] = 0.0f;
	for (int i = 0; i < number; i++) { // f(x,y)=A*x2+B*y2+C*x*y+H*x+I*y, x=(A,B,C,H,I), Q=(x2,y2,x*y,x,y), M*x=b, M=Q*Q^T, b=Q*z
		const float x = p[i].x, y = p[i].y, z = p[i].z, x2 = x * x, y2 = y * y, x3 = x2 * x, y3 = y2 * y;
		/**/M[0] += x2 * x2; M[1] += x2 * y2; M[2] += x3 * y; M[3] += x3; M[4] += x2 * y; b[0] += x2 * z;
		/*M[ 5]+=x2*y2;*/ M[6] += y2 * y2; M[7] += x * y3; M[8] += x * y2; M[9] += y3; b[1] += y2 * z;
		/*M[10]+=x3*y ; M[11]+=x *y3;*/ M[12] += x2 * y2; M[13] += x2 * y; M[14] += x * y2; b[2] += x * y * z;
		/*M[15]+=x3   ; M[16]+=x *y2; M[17]+=x2*y ;*/ M[18] += x2; M[19] += x * y; b[3] += x * z;
		/*M[20]+=x2*y ; M[21]+=   y3; M[22]+=x *y2; M[23]+=x *y ;*/ M[24] += y2; b[4] += y * z;
	}
	for (int i = 1; i < 5; i++) { // use symmetry of matrix to save arithmetic operations
		for (int j = 0; j < i; j++) M[i * 5 + j] = M[j * 5 + i];
	}
	// printf("number: %d\n", number);
	if (number >= 5) lu_solve(M, x, b, 5, 5);
	else lu_solve(M, x, b, 5, number); // cannot do loop unrolling here -> slower -> extra if-else to avoid slowdown

	const float A = x[0], B = x[1], C = x[2], H = x[3], I = x[4];
	const float K = (A * (I * I + 1.0f) + B * (H * H + 1.0f) - C * H * I) * cb(rsqrt_(H * H + I * I + 1.0f)); // mean curvature of Monge patch (x, y, f(x, y))
	return clamp(K, -1.0f, 1.0f);
}



inline MLFUNC_TYPE void mrUtilFuncGpu3D::mlGetPIAfterCollision(
	float R, float U, float V, float W, float Fx, float Fy, float Fz, float omega, /*MlLatticeNodeD3Q27 node_in_out,*/
	float& pixx_t45,
	float& pixy_t90,
	float& pixz_t90,
	float& piyy_t45,
	float& piyz_t90,
	float& pizz_t45
)
{


	REAL pixx_part = (2 * pixx_t45 - piyy_t45 - pizz_t45) / 3.0;//(2 * f1) / 3 + (2 * f2) / 3 - f3 / 3 - f4 / 3 - f5 / 3 - f6 / 3 + f7 / 3 + f8 / 3 + f9 / 3 + f10 / 3 - (2 * f11) / 3 - (2 * f12) / 3 + f13 / 3 + f14 / 3 + f15 / 3 + f16 / 3 - (2 * f17) / 3 - (2 * f18) / 3;
	REAL piyy_part = (2 * piyy_t45 - pixx_t45 - pizz_t45) / 3.0;// -f1 / 3 - f2 / 3 + (2 * f3) / 3 + (2 * f4) / 3 - f5 / 3 - f6 / 3 + f7 / 3 + f8 / 3 - (2 * f9) / 3 - (2 * f10) / 3 + f11 / 3 + f12 / 3 + f13 / 3 + f14 / 3 - (2 * f15) / 3 - (2 * f16) / 3 + f17 / 3 + f18 / 3;
	REAL pizz_part = (2 * pizz_t45 - pixx_t45 - piyy_t45) / 3.0;//-f1 / 3 - f2 / 3 - f3 / 3 - f4 / 3 + (2 * f5) / 3 + (2 * f6) / 3 - (2 * f7) / 3 - (2 * f8) / 3 + f9 / 3 + f10 / 3 + f11 / 3 + f12 / 3 - (2 * f13) / 3 - (2 * f14) / 3 + f15 / 3 + f16 / 3 + f17 / 3 + f18 / 3;
	REAL RU2 = R * U * U;
	REAL RV2 = R * V * V;
	REAL RW2 = R * W * W;
	REAL RUVW2 = (1 * RU2) / 3 + (1 * RV2) / 3 + (1 * RW2) / 3;
	pixx_t45 =
		R / 3
		+ pixx_part * (1 - omega)
		+RUVW2
		+ (2 * RU2 * omega) / 3
		- (1 * RV2 * omega) / 3
		- (1 * RW2 * omega) / 3 + Fx * U;


	piyy_t45 =
		R / 3
		+ piyy_part * (1 - omega)
		+RUVW2
		- (1 * RU2 * omega) / 3
		+ (2 * RV2 * omega) / 3
		- (1 * RW2 * omega) / 3 + Fy * V;

	pizz_t45 =
		R / 3
		+ pizz_part * (1 - omega)
		+RUVW2
		- (1 * RU2 * omega) / 3
		- (1 * RV2 * omega) / 3
		+ (2 * RW2 * omega) / 3 + Fz * W;

	pixy_t90 = pixy_t90 - pixy_t90 * omega + U * V * R * omega + (Fy * U) / 2 + (Fx * V) / 2;
	pixz_t90 = pixz_t90 - pixz_t90 * omega + U * W * R * omega + (Fz * U) / 2 + (Fx * W) / 2;
	piyz_t90 = piyz_t90 - piyz_t90 * omega + V * W * R * omega + (Fz * V) / 2 + (Fy * W) / 2;
}


inline MLFUNC_TYPE void mrUtilFuncGpu3D::mlConvertCmrMoment_d3q7(float ux, float uy, float uz, float * node_in_out)
{
	float  node[7];// = node_in_out;
	for (int i = 0; i < 7; i++)
	{
		node[i] = node_in_out[i];
		node_in_out[i] = 0;
	}
	for (int k = 0; k < 7; k++)
	{
		float CX = (ex3d_gpu[k]) - ux;
		float CY = (ey3d_gpu[k]) - uy;
		float CZ = (ez3d_gpu[k]) - uz;
		float ftemp = node[k];
		node_in_out[0] += ftemp;
		node_in_out[1] += ftemp * CX;
		node_in_out[2] += ftemp * CY;
		node_in_out[3] += ftemp * CZ;
		node_in_out[4] += ftemp * (CX * CX - CY * CY);
		node_in_out[5] += ftemp * (CX * CX - CZ * CZ);
		node_in_out[6] += ftemp * (CX * CX + CY * CY + CZ * CZ);
	}
}


inline MLFUNC_TYPE void mrUtilFuncGpu3D::mlConvertCmrF_d3q7(float U, float V, float W, float* node_in_out)
{
	float k0, k1, k2, k3, k4, k5, k6;
	k0 = node_in_out[0];
	k1 = node_in_out[1];
	k2 = node_in_out[2];
	k3 = node_in_out[3];
	k4 = node_in_out[4];
	k5 = node_in_out[5];
	k6 = node_in_out[6];


	node_in_out[0] = -k0 * U * U - 2 * k1 * U - k0 * V * V - 2 * k2 * V - k0 * W * W - 2 * k3 * W + k0 - k6;
	node_in_out[1] = k1 / 2 + k4 / 6 + k5 / 6 + k6 / 6 + (U * k0) / 2 + U * k1 + (U * U * k0) / 2;
	node_in_out[2] = k4 / 6 - k1 / 2 + k5 / 6 + k6 / 6 - (U * k0) / 2 + U * k1 + (U * U * k0) / 2;
	node_in_out[3] = k2 / 2 - k4 / 3 + k5 / 6 + k6 / 6 + (V * k0) / 2 + V * k2 + (V * V * k0) / 2;
	node_in_out[4] = k5 / 6 - k4 / 3 - k2 / 2 + k6 / 6 - (V * k0) / 2 + V * k2 + (V * V * k0) / 2;
	node_in_out[5] = k3 / 2 + k4 / 6 - k5 / 3 + k6 / 6 + (W * k0) / 2 + W * k3 + (W * W * k0) / 2;
	node_in_out[6] = k4 / 6 - k3 / 2 - k5 / 3 + k6 / 6 - (W * k0) / 2 + W * k3 + (W * W * k0) / 2;
}



#endif // !_MRUTILFUNCGU3DH_
