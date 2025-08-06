#pragma once
#ifndef _MRUTILFUNCGU2DH_
#define _MRUTILFUNCGU2DH_

#include "cuda_runtime.h"
//#include "../../../lw_core_win/mlCoreWinHeader.h"
#include "../../../common/mlCoreWin.h"
#include "../../../common/mlLatticeNode.h"
#include "mrLbmSolverGpu2D.h"
#include "mrConstantParamsGpu2D.h"


class mrUtilFuncGpu2D
{
public:
	MLFUNC_TYPE void calculate_f_eq(const float rho, float ux, float uy, float  uz, float* feq);
	MLFUNC_TYPE void calculate_g_eq(const float rho, float ux, float uy, float  uz, float* feq);
	MLFUNC_TYPE float calculate_phi(const float rhon, const float massn, const unsigned char flagsn);
	MLFUNC_TYPE float calculate_curvature(const float* phit);
	MLFUNC_TYPE float3 calculate_normal(const float* phit);
	MLFUNC_TYPE void mlCalDistributionD2Q9AtIndex(REAL rho, REAL ux, REAL uy, REAL pixx, REAL pixy, REAL piyy, int i, REAL& f_out);
	MLFUNC_TYPE void mlGetPIAfterCollision(REAL R, REAL U, REAL V, REAL Fx, REAL Fy, REAL omega, REAL& pixx, REAL& piyy, REAL& pixy);
	MLFUNC_TYPE float plic_cube(const float V0, const float3 n);
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
	const float n = s / length(v);
	return { v.x * n, v.y * n, v.z * n
	};
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


inline MLFUNC_TYPE float plic_cube_reduced(const float V, const float n1, const float n2, const float n3) { // optimized solution from SZ and Kawano, source: https://doi.org/10.3390/computation10020021
	const float n12 = n1 + n2, n3V = n3 * V;
	if (n12 <= 2.0f * n3V) return n3V + 0.5f * n12; // case (5)
	const float sqn1 = sq(n1), n26 = 6.0f * n2, v1 = sqn1 / n26; // after case (5) check n2>0 is true
	if (v1 <= n3V && n3V < v1 + 0.5f * (n2 - n1)) return 0.5f * (n1 + sqrt(sqn1 + 8.0f * n2 * (n3V - v1))); // case (2)
	const float V6 = n1 * n26 * n3V;
	if (n3V < v1) return cbrt(V6); // case (1)
	const float v3 = n3 < n12 ? (sq(n3) * (3.0f * n12 - n3) + sqn1 * (n1 - 3.0f * n3) + sq(n2) * (n2 - 3.0f * n3)) / (n1 * n26) : 0.5f * n12; // after case (2) check n1>0 is true
	const float sqn12 = sqn1 + sq(n2), V6cbn12 = V6 - cb(n1) - cb(n2);
	const bool case34 = n3V < v3; // true: case (3), false: case (4)
	const float a = case34 ? V6cbn12 : 0.5f * (V6cbn12 - cb(n3));
	const float b = case34 ? sqn12 : 0.5f * (sqn12 + sq(n3));
	const float c = case34 ? n12 : 0.5f;
	const float t = sqrt(sq(c) - b);
	return c - 2.0f * t * sinf(0.33333334f * asinf((cb(c) - 0.5f * a - 1.5f * b * c) / cb(t)));
}

inline MLFUNC_TYPE float mrUtilFuncGpu2D::plic_cube(const float V0, const float3 n) { // unit cube - plane intersection: volume V0 in [0,1], normal vector n -> plane offset d0
	const float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z), V = 0.5f - fabsf(V0 - 0.5f), l = ax + ay + az; // eliminate symmetry cases, normalize n using L1 norm
	const float n1 = fmin(fmin(ax, ay), az) / l;
	const float n3 = fmax(fmax(ax, ay), az) / l;
	const float n2 = fdim(1.0f, n1 + n3); // ensure n2>=0
	const float d = plic_cube_reduced(V, n1, n2, n3); // calculate PLIC with reduced symmetry
	return l * copysignf(0.5f - d, V0 - 0.5f); // rescale result and apply symmetry for V0>0.5
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


inline MLFUNC_TYPE void mrUtilFuncGpu2D::mlCalDistributionD2Q9AtIndex(REAL rho, REAL ux, REAL uy, REAL pixx, REAL pixy, REAL piyy, int j, REAL& f_out)
{
	int inv_map2crt2[9] = {0,1,3,2,4,5,7,8,6};
	int i = inv_map2crt2[j];

	REAL A0 = rho;
	REAL Ax = ux * A0;
	REAL Ay = uy * A0;

	REAL Axx = rho * pixx;
	REAL Ayy = rho * piyy;
	REAL Axy = rho * pixy;

	REAL Axx_0 = ux * Ax;
	REAL Ayy_0 = uy * Ay;
	REAL Axy_0 = Ax * uy;


	REAL Axxx_0 = Axx_0 * ux;
	REAL Ayyy_0 = uy * Ayy_0;
	REAL Axxy_0 = Axx_0 * uy;
	REAL Axyy_0 = ux * Ayy_0;

	REAL Axx_1 = Axx - Axx_0;
	REAL Ayy_1 = Ayy - Ayy_0;
	REAL Axy_1 = Axy - Axy_0;

	REAL Axxy_1 = 2 * ux * (Axy_1)+uy * (Axx_1);
	REAL Axyy_1 = 2 * uy * (Axy_1)+ux * (Ayy_1);


	REAL Axxy = -2 * rho * uy * ux * ux + 2 * Axy * ux + Axx * uy;
	REAL Axyy = -2 * rho * ux * uy * uy + 2 * Axy * uy + Ayy * ux;

	REAL Axxyy_0 = Axxy_0 * uy;
	REAL Axxyy_1 = Axxy_1 * uy + ux * ux * Ayy_1 + 2 * ux * uy * Axy_1;


	REAL Axxyy = 0;// Axxyy_0 + Axxyy_1;


	switch (i)
	{
	case 0:
		f_out = w2d_gpu[i] * (A0 - (3 * Axx) / 2 + (9 * Axxyy) / 4 - (3 * Ayy) / 2);
		break;
	case 1:
		f_out = w2d_gpu[i] * (A0 + 3 * Ax + 3 * Axx - (9 * Axyy) / 2 - (9 * Axxyy) / 2 - (3 * Ayy) / 2);
		break;

	case 2:
		f_out = w2d_gpu[i] * (A0 - (3 * Axx) / 2 - (9 * Axxy) / 2 + 3 * Ay - (9 * Axxyy) / 2 + 3 * Ayy);
		break;

	case 3:
		f_out = w2d_gpu[i] * (A0 - 3 * Ax + 3 * Axx + (9 * Axyy) / 2 - (9 * Axxyy) / 2 - (3 * Ayy) / 2);
		break;

	case 4:
		f_out = w2d_gpu[i] * (A0 - (3 * Axx) / 2 + (9 * Axxy) / 2 - 3 * Ay - (9 * Axxyy) / 2 + 3 * Ayy);
		break;

	case 5:
		f_out = w2d_gpu[i] * (A0 + 3 * Ax + 3 * Axx + 9 * Axy + 9 * Axxy + 9 * Axyy + 3 * Ay + 9 * Axxyy + 3 * Ayy);
		break;

	case 6:
		f_out = w2d_gpu[i] * (A0 - 3 * Ax + 3 * Axx - 9 * Axy + 9 * Axxy - 9 * Axyy + 3 * Ay + 9 * Axxyy + 3 * Ayy);
		break;

	case 7:
		f_out = w2d_gpu[i] * (A0 - 3 * Ax + 3 * Axx + 9 * Axy - 9 * Axxy - 9 * Axyy - 3 * Ay + 9 * Axxyy + 3 * Ayy);
		break;

	case 8:
		f_out = w2d_gpu[i] * (A0 + 3 * Ax + 3 * Axx - 9 * Axy - 9 * Axxy + 9 * Axyy - 3 * Ay + 9 * Axxyy + 3 * Ayy);
		break;
	}
}



inline MLFUNC_TYPE void mrUtilFuncGpu2D::mlGetPIAfterCollision(REAL R, REAL U, REAL V, REAL Fx, REAL Fy, REAL omega, REAL& pixx, REAL& piyy, REAL& pixy)
{
	REAL pixx_part = (pixx - piyy) / 2;
	REAL piyy_part = (piyy - pixx) / 2;
	REAL RU2 = R * U * U;
	REAL RV2 = R * V * V;
	pixx = R / 3 + pixx_part * (1 - omega)
		+ RU2 / 2
		+ RV2 / 2
		+ RU2 * omega / 2
		- RV2 * omega / 2 + Fx * U + 0.5f * (1 - omega) * (Fx * U - Fy * V);
	piyy = R / 3 + piyy_part * (1 - omega)
		+ RU2 / 2
		+ RV2 / 2
		- RU2 * omega / 2
		+ RV2 * omega / 2 + Fy * V + 0.5f * (1 - omega) * (Fy * V - Fx * U);
	pixy = pixy - pixy * omega + U * V * R * omega + (1 - 0.5f * omega) * (Fy * U + Fx * V);
}


inline MLFUNC_TYPE void mrUtilFuncGpu2D::calculate_f_eq(const float rho, float ux, float uy, float  uz, float* feq)
{
	const float c3 = -3.0f * (sq(ux) + sq(uy) + sq(uz));
	const float	rhom1 = rho - 1.0f; // c3 = -2*sq(u)/(2*sq(c)), rhom1 is arithmetic optimization to minimize digit extinction
	ux *= 3.0f;
	uy *= 3.0f;
	uz *= 3.0f;
	feq[0] = def_w0 * fma(rho, 0.5f * c3, rhom1);
	const float u0 = ux + uy, u1 = ux - uy; // these pre-calculations make manual unrolling require less FLOPs
	const float rhos = def_ws * rho, rhoe = def_we * rho, rhom1s = def_ws * rhom1, rhom1e = def_we * rhom1;
	feq[1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[5] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[6] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	feq[7] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[8] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +-0 -+0

}

inline MLFUNC_TYPE void mrUtilFuncGpu2D::calculate_g_eq(const float rho, float ux, float uy, float  uz, float* feq)
{
	// g is updated by the D2Q5
	const float	rhom1 = rho; // c3 = -2*sq(u)/(2*sq(c)), rhom1 is arithmetic optimization to minimize digit extinction
	ux *= 3.0f;
	uy *= 3.0f;
	uz *= 3.0f;
	feq[0] = d2q5_w[0] *  rhom1 ;

	const float rhos = d2q5_w[1] * rho, rhom1s = d2q5_w[1] * rhom1;
	feq[1] = d2q5_w[1] * (1 + (float)ex2d_gpu[1] * ux + (float)ey2d_gpu[1] * uy) * rhom1 ;
	feq[2] = d2q5_w[2] * (1 + (float)ex2d_gpu[2] * ux + (float)ey2d_gpu[2] * uy) * rhom1 ;
	feq[3] = d2q5_w[3] * (1 + (float)ex2d_gpu[3] * ux + (float)ey2d_gpu[3] * uy) * rhom1 ;
	feq[4] = d2q5_w[4] * (1 + (float)ex2d_gpu[4] * ux + (float)ey2d_gpu[4] * uy) * rhom1 ;
	//need to change
}


inline MLFUNC_TYPE float mrUtilFuncGpu2D::calculate_phi(const float rhon, const float massn, const unsigned char flagsn)
{
	return flagsn & TYPE_F ? 1.0f : flagsn & TYPE_I ? rhon > 0.0f ? clamp(massn / rhon, 0.0f, 1.0f) : 0.5f : 0.0f;
}


inline MLFUNC_TYPE float3 mrUtilFuncGpu2D::calculate_normal(const float* phij)
{
	float phit[9];
	for (int ik=0;ik<9;ik++)
		phit[ik] = phij[index2dInv_gpu[ik]];
	float3 by;
	by.x = 2.0f * (phit[2] - phit[1]) + phit[6] - phit[5] + phit[8] - phit[7];
	by.y = 2.0f * (phit[4] - phit[3]) + phit[6] - phit[5] + phit[7] - phit[8];
	by.z = 0.0f;
	by = normalize(by);
	return by;
}

inline MLFUNC_TYPE float mrUtilFuncGpu2D::calculate_curvature(const float* phij)
{
	float phit[9];
	for (int ik=0;ik<9;ik++)
		phit[ik] = phij[index2dInv_gpu[ik]];


	float3 by;
	by.x = 2.0f * (phit[2] - phit[1]) + phit[6] - phit[5] + phit[8] - phit[7];
	by.y = 2.0f * (phit[4] - phit[3]) + phit[6] - phit[5] + phit[7] - phit[8];
	by.z = 0.0f;


	by = normalize(by);
	
	const float3 z{ 0.0f, 0.0f, 1.0f};
	const float3 bx = cross(by, z); // normalize() is necessary here because bz and rn are not perpendicular
	int number = 0; // number of neighboring interface points
	float2 p[6]; // number of neighboring interface points is less or equal than than 8 minus 1 gas and minus 1 fluid point = 6
	const float center_offset = plic_cube(phit[0], by); // calculate z-offset PLIC of center point only once
	for (int i = 1; i < 9; i++) { // iterate over neighbors, no loop unrolling here (50% better perfoemance without loop unrolling)
		if (phit[i] > 0.0f && phit[i] < 1.0f) { // limit neighbors to interface cells
			
			const float3 ei{ (float)ex2d_gpu[i], (float)ey2d_gpu[i], 0.0f }; // assume neighbor normal vector is the same as center normal vector
			const float offset = plic_cube(phit[i], by) - center_offset;
			p[number].x = dot(ei, bx);
			p[number].y = dot(ei, by) + offset; // do coordinate system transformation into (x, f(x)) and apply PLIC pffsets
			number+=1;
		}
	}
	float M[4] = { 0.0f,0.0f,0.0f,0.0f }, x[2] = { 0.0f,0.0f }, b[2] = { 0.0f,0.0f };
	for (int i = 0; i < number; i++) { // f(x,y)=A*x2+H*x, x=(A,H), Q=(x2,x), M*x=b, M=Q*Q^T, b=Q*z
		const float x = p[i].x, y = p[i].y, x2 = x * x, x3 = x2 * x;
		/**/M[0] += x2 * x2; M[1] += x3; b[0] += x2 * y;
		/*M[2]+=x3   ;*/ M[3] += x2; b[1] += x * y;
	}
	M[2] = M[1]; // use symmetry of matrix to save arithmetic operations
	if (number >= 2) lu_solve(M, x, b, 2, 2);
	else lu_solve(M, x, b, 2, number); // cannot do loop unrolling here -> slower -> extra if-else to avoid slowdown
	const float A = x[0], H = x[1];
	const float K = 2.0f * A * cb(rsqrt_(H * H + 1.0f)); // mean curvature of Monge patch (x, y, f(x, y))

	return clamp(K,-1.f,1.f);
}


#endif // !_MRUTILFUNCGU3DH_
