#pragma once
#ifndef _MRUTILFUNCGU2DH_
#define _MRUTILFUNCGU2DH_

#include "cuda_runtime.h"
//#include "../../../lw_core_win/mlCoreWinHeader.h"
#include "../../../common/mlCoreWin.h"
#include "../../../common/mlLatticeNode.h"
#include "mrLbmSolverGpu2D.h"
#include "mrConstantParamsGpu2D.h"

class RigidFuncGpu2D
{
	public:
	MLFUNC_TYPE bool intersect(float2 point, float2 pos, float2 scale, float angle);
	MLFUNC_TYPE void lineSegmentDiamondIntersections(float2 lineStart, float2 lineEnd, float2 diamondPos, float2 diamondScale, float diamondAngle, float2& intersection);
	MLFUNC_TYPE float2 calculatePointVelocity(float2 point, float2 diamondCenter, float3 translationVelocity, float angularVelocity);
	MLFUNC_TYPE float pointToRhombusDistance(float2 point, float2 pos, float2 scale, float angle, float3 translationVelocity, float angularVelocity, float2& velocityAtClosestPoint);

};


inline MLFUNC_TYPE bool RigidFuncGpu2D::intersect(float2 point, float2 pos, float2 scale, float angle) {
	// ����ƽ�Ƶ�����������Ϊԭ�������ϵ
	float translatedX = point.x - pos.x;
	float translatedY = point.y - pos.y;

	// ������ת�Ƕȵ����Һ�����
	float cosTheta = cos(angle);
	float sinTheta = sin(angle);

	// ��ʱ����ת���Ե������ε���ת
	float rotatedX = translatedX * cosTheta + translatedY * sinTheta;
	float rotatedY = -translatedX * sinTheta + translatedY * cosTheta;

	// ʹ�÷���ת���ε��������ж�
	float dx = scale.x / 2.0f;
	float dy = scale.y / 2.0f;
	return (abs(rotatedX / dx) + abs(rotatedY / dy)) <= 1.0f;
}

// ���ڴ洢����Ŀ�ѡ����
inline MLFUNC_TYPE float2 computeIntersection(float2 p1, float2 p2, float2 p3, float2 p4) {
	// �����߶εķ�������
	float2 d1 = { p2.x - p1.x, p2.y - p1.y };
	float2 d2 = { p4.x - p3.x, p4.y - p3.y };

	// ��������ʽ
	float det = d1.x * d2.y - d1.y * d2.x;

	// �������ʽΪ�㣬�߶�ƽ�л���
	if (abs(det) < 1e-6) {
		return {-1,-1};
	}

	// ʹ�ÿ���ķ������㽻��
	float t = ((p3.x - p1.x) * d2.y - (p3.y - p1.y) * d2.x) / det;
	float u = ((p3.x - p1.x) * d1.y - (p3.y - p1.y) * d1.x) / det;

	// ��� t �� u �Ƿ��� [0, 1] ��Χ��
	if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
		return float2{ p1.x + t * d1.x, p1.y + t * d1.y };
	}
	return { -1,-1 };
}

inline MLFUNC_TYPE float2 RigidFuncGpu2D::calculatePointVelocity(float2 point, float2 diamondCenter, float3 translationVelocity, float angularVelocity) {
	// ������������ĵ�����
	float x = point.x - diamondCenter.x;
	float y = point.y - diamondCenter.y;

	// ������ת�ٶ�
	float2 rotationalVelocity = { -angularVelocity * y, angularVelocity * x };

	// �������ٶ�
	float2 totalVelocity = {
		translationVelocity.x + rotationalVelocity.x,
		translationVelocity.y + rotationalVelocity.y
	};

	return totalVelocity;
}

inline MLFUNC_TYPE void calculateDiamondVertices(float2 pos, float2 scale, float angle, float2* vertices) {
	float dx = scale.x / 2.0f;
	float dy = scale.y / 2.0f;

	float cosTheta = cos(angle);
	float sinTheta = sin(angle);

	vertices[0] = { pos.x + dx * cosTheta, pos.y + dx * sinTheta };
	vertices[1] = { pos.x - dy * sinTheta, pos.y + dy * cosTheta };
	vertices[2] = { pos.x - dx * cosTheta, pos.y - dx * sinTheta };
	vertices[3] = { pos.x + dy * sinTheta, pos.y - dy * cosTheta };
}

inline MLFUNC_TYPE void RigidFuncGpu2D::lineSegmentDiamondIntersections(float2 lineStart, float2 lineEnd, float2 diamondPos, float2 diamondScale, float diamondAngle, float2& intersection) {
	float2 vertices[4];
	calculateDiamondVertices(diamondPos, diamondScale, diamondAngle, vertices);

	int count = 0;
	float2 intersection_;
	for (int i = 0; i < 4; ++i) {
		intersection_ = computeIntersection(lineStart, lineEnd, vertices[i], vertices[(i + 1) % 4]);
		if (intersection_.x != -1)
		{
			intersection = intersection_;
		}
	}
}

inline MLFUNC_TYPE float pointToLineDistance(float2 p, float2 a, float2 b, float2& closestPoint) {
	float A = p.x - a.x;
	float B = p.y - a.y;
	float C = b.x - a.x;
	float D = b.y - a.y;
	float dot = A * C + B * D;
	float len_sq = C * C + D * D;
	float param = (len_sq != 0) ? dot / len_sq : -1;
	if (param < 0) {
		closestPoint = a;
	}
	else if (param > 1) {
		closestPoint = b;
	}
	else {
		closestPoint = { a.x + param * C, a.y + param * D };
	}
	float dx = p.x - closestPoint.x;
	float dy = p.y - closestPoint.y;
	return sqrt(dx * dx + dy * dy);
}

inline MLFUNC_TYPE float RigidFuncGpu2D::pointToRhombusDistance(float2 point, float2 pos, float2 scale, float angle, float3 translationVelocity, float angularVelocity, float2& velocityAtClosestPoint) {
	float2 vertices[4];
	calculateDiamondVertices(pos, scale, angle, vertices);
	float minDistance = 1000.f;
	float2 closestPoint;

	for (int i = 0; i < 4; ++i) {
		float2 tempClosestPoint;
		float distance = pointToLineDistance(point, vertices[i], vertices[(i + 1) % 4], tempClosestPoint);
		if (distance < minDistance) {
			minDistance = distance;
			closestPoint = tempClosestPoint;
		}
	}

	velocityAtClosestPoint = calculatePointVelocity(closestPoint, pos, translationVelocity, angularVelocity);
	return minDistance;
}

class mrUtilFuncGpu2D
{
public:
	MLFUNC_TYPE void calculate_rho_u(REAL* pop, REAL& rho, REAL& ux, REAL& uy, REAL& uz);
	MLFUNC_TYPE void calculate_forcing_terms(REAL ux, REAL uy, REAL uz, REAL fx, REAL fy, REAL fz, REAL* Fin);
	MLFUNC_TYPE void calculate_f_eq(const float rho, float ux, float uy, float  uz, float* feq);
	MLFUNC_TYPE void calculate_g_eq(const float rho, float ux, float uy, float  uz, float* feq);
	MLFUNC_TYPE float calculate_phi(const float rhon, const float massn, const unsigned char flagsn);
	MLFUNC_TYPE float calculate_curvature(const float* phit, int print);
	MLFUNC_TYPE void ComputeCentralMomentK(REAL ux, REAL uy, REAL uz, float* node, float* node_in_out);
	MLFUNC_TYPE void CentralProFvalue(REAL ux, REAL uy, REAL uz, float* node_in_out);
	MLFUNC_TYPE void ComputeCentralEQ(REAL rho, float* node_in_out);
	MLFUNC_TYPE void ComputeCentralForcing(REAL fx, REAL fy, REAL fz, float* node_in_out);
	MLFUNC_TYPE float3 calculate_normal(const float* phit);
	MLFUNC_TYPE void mlCalDistributionD2Q9AtIndex(REAL rho, REAL ux, REAL uy, REAL pixx, REAL pixy, REAL piyy, int i, REAL& f_out);
	MLFUNC_TYPE void mlGetPIAfterCollision(REAL R, REAL U, REAL V, REAL Fx, REAL Fy, REAL omega, REAL& pixx, REAL& piyy, REAL& pixy);
	//MLFUNC_TYPE void calculate_f_eq_index(const float rho, float ux, float uy, int index, float& feq);
	MLFUNC_TYPE void calculate_f_eq2(const float rho, float ux, float uy, float  uz, float* feq);
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
	return c - 2.0f * t * sin(0.33333334f * asin((cb(c) - 0.5f * a - 1.5f * b * c) / cb(t)));
}

inline MLFUNC_TYPE float mrUtilFuncGpu2D::plic_cube(const float V0, const float3 n) { // unit cube - plane intersection: volume V0 in [0,1], normal vector n -> plane offset d0
	const float ax = fabs(n.x), ay = fabs(n.y), az = fabs(n.z), V = 0.5f - fabs(V0 - 0.5f), l = ax + ay + az; // eliminate symmetry cases, normalize n using L1 norm
	const float n1 = fmin(fmin(ax, ay), az) / l;
	const float n3 = fmax(fmax(ax, ay), az) / l;
	const float n2 = fdim(1.0f, n1 + n3); // ensure n2>=0
	const float d = plic_cube_reduced(V, n1, n2, n3); // calculate PLIC with reduced symmetry
	return l * copysign(0.5f - d, V0 - 0.5f); // rescale result and apply symmetry for V0>0.5
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

//inline MLFUNC_TYPE void mrUtilFuncGpu2D::calculate_f_eq_index(const float rho, float ux, float uy, int index, float& feq)
//{
//	index = inv_map2crt[index];
//	const float	rhom1 = rho;
//	float cu, U2;
//	U2 = ux * ux + uy * uy;
//	cu = ex2d_gpu[index] * ux + ey2d_gpu[index] * uy; // c k*u
//	feq = w2d_gpu[index] * rhom1 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U2) - w2d_gpu[index];
//
//}

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

inline MLFUNC_TYPE void mrUtilFuncGpu2D::ComputeCentralMomentK(REAL ux, REAL uy, REAL uz, float* node, float* node_in_out)
{
	for (int i = 0; i < 9; i++)
	{
		node_in_out[i] = 0;
	}

	for (int k = 0; k < 9; k++)
	{
		float CX = ex2d_gpu[k] - ux;
		float CY = ey2d_gpu[k] - uy;
		float ftemp = node[k];
		node_in_out[map2crt[0]] += ftemp;
		node_in_out[map2crt[1]] += ftemp * CX;
		node_in_out[map2crt[2]] += ftemp * CY;
		node_in_out[map2crt[3]] += ftemp * (CX * CX + CY * CY);
		node_in_out[map2crt[4]] += ftemp * (CX * CX - CY * CY);
		node_in_out[map2crt[5]] += ftemp * CX * CY;
		node_in_out[map2crt[6]] += ftemp * CX * CX * CY;
		node_in_out[map2crt[7]] += ftemp * CX * CY * CY;
		node_in_out[map2crt[8]] += ftemp * CX * CX * CY * CY;
	}
}


inline MLFUNC_TYPE void mrUtilFuncGpu2D::ComputeCentralEQ(REAL rho, float* node_in_out)
{
	for (int i = 0; i < 9; i++) {
		node_in_out[i] = 0;
	}
	float R = rho;
	node_in_out[map2crt[0]] = R;
	node_in_out[map2crt[3]] = 2 * R / 3.0f;
	node_in_out[map2crt[8]] = R / 9.0f;

}

inline MLFUNC_TYPE void mrUtilFuncGpu2D::calculate_f_eq2(const float rho, float ux, float uy, float  uz, float* feq)
{
	//// printf("rho: %f, ux: %f, uy: %f, uz: %f\n", rho, ux, uy, uz);
	//const float c3 = -3.0f * (sq(ux) + sq(uy) + sq(uz));
	//// printf("c3: %f\n", c3);
	//const float	rhom1 = rho - 1.0f; // c3 = -2*sq(u)/(2*sq(c)), rhom1 is arithmetic optimization to minimize digit extinction
	//ux *= 3.0f;
	//uy *= 3.0f;
	//uz *= 3.0f;
	//feq[0] = def_w0 * fma(rho, 0.5f * c3, rhom1);
	//const float u0 = ux + uy, u1 = ux - uy; // these pre-calculations make manual unrolling require less FLOPs
	//const float rhos = def_ws * rho, rhoe = def_we * rho, rhom1s = def_ws * rhom1, rhom1e = def_we * rhom1;
	//feq[1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[3] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	//feq[2] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	//feq[5] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	//feq[8] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[6] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +-0 -+0

	const float	rhom1 = rho;
	float cu, U2;
	U2 = ux * ux + uy * uy + uz * uz;
	for (int i = 0; i < 9; i++)
	{
		cu = ex2d_gpu[i] * ux + ey2d_gpu[i] * uy; // c k*u
		feq[i] = w2d_gpu[i] * rhom1 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U2) - w2d_gpu[i];
	}

}

inline MLFUNC_TYPE void mrUtilFuncGpu2D::ComputeCentralForcing(REAL fx, REAL fy, REAL fz, float* node_in_out)
{
	for (int i = 0; i < 9; i++) {
		node_in_out[i] = 0;
	}
	float cs = 1 / sqrtf(3);
	node_in_out[map2crt[1]] = fx;
	node_in_out[map2crt[2]] = fy;
	node_in_out[map2crt[6]] = fy * cs * cs;
	node_in_out[map2crt[7]] = fx * cs * cs;
}


inline MLFUNC_TYPE void mrUtilFuncGpu2D::CentralProFvalue(REAL ux, REAL uy, REAL uz, float* node_in_out)
{

	float U = ux;
	float V = uy;

	float k0, k1, k2, k3, k4, k5, k6, k7, k8;

	k0 = node_in_out[map2crt[0]];
	k1 = node_in_out[map2crt[1]];
	k2 = node_in_out[map2crt[2]];
	k3 = node_in_out[map2crt[3]];
	k4 = node_in_out[map2crt[4]];
	k5 = node_in_out[map2crt[5]];
	k6 = node_in_out[map2crt[6]];
	k7 = node_in_out[map2crt[7]];
	k8 = node_in_out[map2crt[8]];

	node_in_out[map2crt[0]] = k0 - k3 + k8 - 2 * k2 * V + 2 * k6 * V - k0 * U * U + (k3 * U * U) / 2 - (k4 * U * U) / 2 - k0 * V * V + (k3 * V * V) / 2 + (k4 * V * V) / 2 - 2 * k1 * U + 2 * k7 * U + k0 * U * U * V * V + 4 * k5 * U * V + 2 * k1 * U * V * V + 2 * k2 * U * U * V;
	node_in_out[map2crt[1]] = k1 / 2 + k3 / 4 + k4 / 4 - k7 / 2 - k8 / 2 - k5 * V - k6 * V + (k0 * U * U) / 2 - (k3 * U * U) / 4 + (k4 * U * U) / 4 - (k1 * V * V) / 2 - (k3 * V * V) / 4 - (k4 * V * V) / 4 + (k0 * U) / 2 + k1 * U - (k3 * U) / 4 + (k4 * U) / 4 - k7 * U - (k0 * U * U * V * V) / 2 - k2 * U * V - 2 * k5 * U * V - (k0 * U * V * V) / 2 - k1 * U * V * V - k2 * U * U * V;
	node_in_out[map2crt[2]] = k2 / 2 + k3 / 4 - k4 / 4 - k6 / 2 - k8 / 2 + (k0 * V) / 2 + k2 * V - (k3 * V) / 4 - (k4 * V) / 4 - k6 * V - (k2 * U * U) / 2 - (k3 * U * U) / 4 + (k4 * U * U) / 4 + (k0 * V * V) / 2 - (k3 * V * V) / 4 - (k4 * V * V) / 4 - k5 * U - k7 * U - (k0 * U * U * V * V) / 2 - k1 * U * V - 2 * k5 * U * V - (k0 * U * U * V) / 2 - k1 * U * V * V - k2 * U * U * V;
	node_in_out[map2crt[3]] = k3 / 4 - k1 / 2 + k4 / 4 + k7 / 2 - k8 / 2 + k5 * V - k6 * V + (k0 * U * U) / 2 - (k3 * U * U) / 4 + (k4 * U * U) / 4 + (k1 * V * V) / 2 - (k3 * V * V) / 4 - (k4 * V * V) / 4 - (k0 * U) / 2 + k1 * U + (k3 * U) / 4 - (k4 * U) / 4 - k7 * U - (k0 * U * U * V * V) / 2 + k2 * U * V - 2 * k5 * U * V + (k0 * U * V * V) / 2 - k1 * U * V * V - k2 * U * U * V;
	node_in_out[map2crt[4]] = k3 / 4 - k2 / 2 - k4 / 4 + k6 / 2 - k8 / 2 - (k0 * V) / 2 + k2 * V + (k3 * V) / 4 + (k4 * V) / 4 - k6 * V + (k2 * U * U) / 2 - (k3 * U * U) / 4 + (k4 * U * U) / 4 + (k0 * V * V) / 2 - (k3 * V * V) / 4 - (k4 * V * V) / 4 + k5 * U - k7 * U - (k0 * U * U * V * V) / 2 + k1 * U * V - 2 * k5 * U * V + (k0 * U * U * V) / 2 - k1 * U * V * V - k2 * U * U * V;
	node_in_out[map2crt[5]] = k5 / 4 + k6 / 4 + k7 / 4 + k8 / 4 + (k1 * V) / 4 + (k3 * V) / 8 + (k4 * V) / 8 + (k5 * V) / 2 + (k6 * V) / 2 + (k2 * U * U) / 4 + (k3 * U * U) / 8 - (k4 * U * U) / 8 + (k1 * V * V) / 4 + (k3 * V * V) / 8 + (k4 * V * V) / 8 + (k2 * U) / 4 + (k3 * U) / 8 - (k4 * U) / 8 + (k5 * U) / 2 + (k7 * U) / 2 + (k0 * U * U * V * V) / 4 + (k0 * U * V) / 4 + (k1 * U * V) / 2 + (k2 * U * V) / 2 + k5 * U * V + (k0 * U * V * V) / 4 + (k0 * U * U * V) / 4 + (k1 * U * V * V) / 2 + (k2 * U * U * V) / 2;
	node_in_out[map2crt[6]] = k6 / 4 - k5 / 4 - k7 / 4 + k8 / 4 - (k1 * V) / 4 + (k3 * V) / 8 + (k4 * V) / 8 - (k5 * V) / 2 + (k6 * V) / 2 + (k2 * U * U) / 4 + (k3 * U * U) / 8 - (k4 * U * U) / 8 - (k1 * V * V) / 4 + (k3 * V * V) / 8 + (k4 * V * V) / 8 - (k2 * U) / 4 - (k3 * U) / 8 + (k4 * U) / 8 + (k5 * U) / 2 + (k7 * U) / 2 + (k0 * U * U * V * V) / 4 - (k0 * U * V) / 4 + (k1 * U * V) / 2 - (k2 * U * V) / 2 + k5 * U * V - (k0 * U * V * V) / 4 + (k0 * U * U * V) / 4 + (k1 * U * V * V) / 2 + (k2 * U * U * V) / 2;
	node_in_out[map2crt[7]] = k5 / 4 - k6 / 4 - k7 / 4 + k8 / 4 + (k1 * V) / 4 - (k3 * V) / 8 - (k4 * V) / 8 - (k5 * V) / 2 + (k6 * V) / 2 - (k2 * U * U) / 4 + (k3 * U * U) / 8 - (k4 * U * U) / 8 - (k1 * V * V) / 4 + (k3 * V * V) / 8 + (k4 * V * V) / 8 + (k2 * U) / 4 - (k3 * U) / 8 + (k4 * U) / 8 - (k5 * U) / 2 + (k7 * U) / 2 + (k0 * U * U * V * V) / 4 + (k0 * U * V) / 4 - (k1 * U * V) / 2 - (k2 * U * V) / 2 + k5 * U * V - (k0 * U * V * V) / 4 - (k0 * U * U * V) / 4 + (k1 * U * V * V) / 2 + (k2 * U * U * V) / 2;
	node_in_out[map2crt[8]] = k7 / 4 - k6 / 4 - k5 / 4 + k8 / 4 - (k1 * V) / 4 - (k3 * V) / 8 - (k4 * V) / 8 + (k5 * V) / 2 + (k6 * V) / 2 - (k2 * U * U) / 4 + (k3 * U * U) / 8 - (k4 * U * U) / 8 + (k1 * V * V) / 4 + (k3 * V * V) / 8 + (k4 * V * V) / 8 - (k2 * U) / 4 + (k3 * U) / 8 - (k4 * U) / 8 - (k5 * U) / 2 + (k7 * U) / 2 + (k0 * U * U * V * V) / 4 - (k0 * U * V) / 4 - (k1 * U * V) / 2 + (k2 * U * V) / 2 + k5 * U * V + (k0 * U * V * V) / 4 - (k0 * U * U * V) / 4 + (k1 * U * V * V) / 2 + (k2 * U * U * V) / 2;

}






inline MLFUNC_TYPE void mrUtilFuncGpu2D::calculate_rho_u(REAL* pop, REAL& rho, REAL& ux, REAL& uy, REAL& uz)
{

	for (int i = 0; i < 9; i++)
		rho += pop[i];
	rho += 1.0f;
	for (int i = 0; i < 9; i++)
	{
		ux += pop[i] * ex2d_gpu[i];
		uy += pop[i] * ey2d_gpu[i];
		uz += pop[i] * ez2d_gpu[i];
	}
	ux = ux / rho;
	uy = uy / rho;
	uz = uz / rho;
}



// fix the f_eq as the original ones
inline MLFUNC_TYPE void mrUtilFuncGpu2D::calculate_f_eq(const float rho, float ux, float uy, float  uz, float* feq)
{
	// printf("rho: %f, ux: %f, uy: %f, uz: %f\n", rho, ux, uy, uz);
	const float c3 = -3.0f * (sq(ux) + sq(uy) + sq(uz));
	// printf("c3: %f\n", c3);
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

	// const float c3 = -3.0f * (sq(ux) + sq(uy) + sq(uz));
	// printf("c3: %f\n", c3);
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


inline MLFUNC_TYPE void mrUtilFuncGpu2D::calculate_forcing_terms(REAL ux, REAL uy, REAL uz, REAL fx, REAL fy, REAL fz, REAL* Fin)
{
	const float uF = -0.33333334f * fma(ux, fx, uy * fy);
	Fin[0] = 9.0f * def_w0 * uF; // 000 (identical for all velocity sets)
	for (int i = 1; i < 9; i++) { // loop is entirely unrolled by compiler, no unnecessary FLOPs are happening
		Fin[i] = 9.0f * w2d_gpu[i] * fma(ex2d_gpu[i] * fx + ey2d_gpu[i] * fy
			+ ez2d_gpu[i] * fz, ex2d_gpu[i] * ux
			+ ey2d_gpu[i] * uy +
			ez2d_gpu[i] * uz + 0.33333334f, uF);
	}
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

inline MLFUNC_TYPE float mrUtilFuncGpu2D::calculate_curvature(const float* phij,int print)
{
	//const float3 by = calculate_normal_py(phit); // new coordinate system: bz is normal to surface, bx and by are tangent to surface
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

	if (print>0)
	{
		printf("by %f %f %f \n", by.x, by.y, by.z);
		printf("bx %f %f %f \n", bx.x, bx.y, bx.z);
	}

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
			//if (number > 5)
			//	printf("before x y number %f %f %f %d \n",  dot(ei, bx), dot(ei, by), offset, number);
		}
	}
	if (number > 7)
		printf("number %d \n", number);
	float M[4] = { 0.0f,0.0f,0.0f,0.0f }, x[2] = { 0.0f,0.0f }, b[2] = { 0.0f,0.0f };
	for (int i = 0; i < number; i++) { // f(x,y)=A*x2+H*x, x=(A,H), Q=(x2,x), M*x=b, M=Q*Q^T, b=Q*z
		const float x = p[i].x, y = p[i].y, x2 = x * x, x3 = x2 * x;
		/**/M[0] += x2 * x2; M[1] += x3; b[0] += x2 * y;
		/*M[2]+=x3   ;*/ M[3] += x2; b[1] += x * y;
		//printf("after x y number %f %f %d \n",  x, y, i);
	}
	M[2] = M[1]; // use symmetry of matrix to save arithmetic operations

	
	if (number >= 2) lu_solve(M, x, b, 2, 2);
	else lu_solve(M, x, b, 2, number); // cannot do loop unrolling here -> slower -> extra if-else to avoid slowdown

	const float A = x[0], H = x[1];
	const float K = 2.0f * A * cb(rsqrt_(H * H + 1.0f)); // mean curvature of Monge patch (x, y, f(x, y))
	// if (K > 1.f || K<-1.f)
	// {
	// 	printf("A %f H %f K %f\n",A,H, K);
	// }



	return clamp(K,-1.f,1.f);
}


#endif // !_MRUTILFUNCGU3DH_
