#pragma once
#ifndef _MLPHYSCIALMAPPINGH_
#define _MLPHYSCIALMAPPINGH_

#include "mlFluidCommon.h"
#include "mlCuRunTime.h"
#include <fstream>
#include <iostream>
#include <vector>


class mlPhysicalParamMapping
{
public:
	mlPhysicalParamMapping()
	{}
	mlPhysicalParamMapping(mlVelocity3f _uop, float _labma, float _l0p, float _N, float _roup)
	{
		u0p = _uop;
		labma = _labma;
		l0p = _l0p;
		N = _N;
		roup = _roup;
	}

	float lp, tp, xp;
	float deltax, deltat;
	float t0p, l0p;
	float N;

	mlVelocity3f u0p;

	float viscosity_p, viscosity_k;
	float labma;
	float roup;
	void mlMappingTime_LaToPhys(float& tk, float& tp);
	void mlMappingX_LaToPhys(float& xk, float& xp);
	void mlMappingVelocity_LaToPhys(mlVelocity3f& uk, mlVelocity3f& up);
	void mlMappingVelocity_LaToPhys(mlVelocity3f& uin_out);
	void mlMappingForce_LaToPhys(mlVector3f& forcek, mlVector3f& forcep);
	void mlMappingForce_LaToPhys(mlVector3f& forcepin_out);

	void mlMappingViscosity_LaToPhys(float& viscosityk, float& viscosityp);
	void mlMappingViscosity_LaToPhys(float& viscosityin_out);

	void mlMappingVelocity_PhysToLa(mlVelocity3f& uk, mlVelocity3f& up);
	void mlMappingVelocity_PhysToLa(mlVelocity3f& uin_out);

	void mlMappingForce_PhysToLa(mlVector3f& forcek, mlVector3f& forcep);
	void mlMappingForce_PhysToLa(mlVector3f& forcein_out);

	void mlMappingViscosity_PhysToLa(float& viscosityk, float& viscosityp);
	void mlMappingViscosity_PhysToLa(float& viscosityin_out);

	void mlMappingPressure_LaToPhys();
	void mlsetInitVariable(mlVelocity3f u0p, float labma, float l0p, float N, float _roup);

	void mlCalGravity_phy(float& g, float length);
	void mlCalGravity_La(float& gL, float& gp);
	void mlCaltp(float& t);



	mlPhysicalParamMapping& operator= (const mlPhysicalParamMapping& mapping);
};
void mlPhysicalParamMapping::mlMappingTime_LaToPhys(float& tk, float& tp)
{
	tp = labma * l0p * tk / (u0p.uy * N);
}
void mlPhysicalParamMapping::mlMappingX_LaToPhys(float& xk, float& xp)
{
	xp = xk * l0p / N;
}

void mlPhysicalParamMapping::mlMappingVelocity_LaToPhys(mlVelocity3f& uk, mlVelocity3f& up)
{
	up.ux = u0p.ux / labma * uk.ux;
	up.uy = u0p.uy / labma * uk.uy;
	up.uz = u0p.uz / labma * uk.uz;
}

inline void mlPhysicalParamMapping::mlMappingVelocity_LaToPhys(mlVelocity3f& uin_out)
{
	mlVelocity3f uin = uin_out;
	uin_out.ux = u0p.ux / labma * uin.ux;
	uin_out.uy = u0p.uy / labma * uin.uy;
	uin_out.uz = u0p.uz / labma * uin.uz;
}

void mlPhysicalParamMapping::mlMappingForce_LaToPhys(mlVector3f& forcek, mlVector3f& forcep)
{
	forcep[0] = forcek[0] * N * (u0p.ux * u0p.ux) * roup / (l0p * labma * labma);
	forcep[1] = forcek[1] * N * (u0p.uy * u0p.uy) * roup / (l0p * labma * labma);
	forcep[2] = forcek[2] * N * (u0p.uz * u0p.uz) * roup / (l0p * labma * labma);


	//forcep.fx = forcek.fx*(u0p.ux*u0p.ux)*roup / (labma*labma) * ((l0p*l0p) / (N*N));
	//forcep.fy = forcek.fy*(u0p.uy*u0p.uy)*roup / (labma*labma) * ((l0p*l0p) / (N*N));
	//forcep.fz = forcek.fz*(u0p.uz*u0p.uz)*roup / (labma*labma) * ((l0p*l0p) / (N*N));

	//forcep.fx = forcek.fx*(u0p.ux)*roup / (labma);
	//forcep.fy = forcek.fy*(u0p.uy)*roup / (labma);
	//forcep.fz = forcek.fz*(u0p.uz)*roup / (labma);

}

void mlPhysicalParamMapping::mlMappingForce_LaToPhys(mlVector3f& forcepin_out)
{

	forcepin_out[0] = forcepin_out[0] * N * (u0p.ux * u0p.ux * roup) / (l0p * labma * labma);
	forcepin_out[1] = forcepin_out[1] * N * (u0p.uy * u0p.uy * roup) / (l0p * labma * labma);
	forcepin_out[2] = forcepin_out[2] * N * (u0p.uz * u0p.uz * roup) / (l0p * labma * labma);


	//forcepin_out.fx = forcepin_out.fx*(u0p.ux*u0p.ux*roup) / (labma*labma) * ((l0p*l0p) / (N*N));
	//forcepin_out.fy = forcepin_out.fy*(u0p.uy*u0p.uy*roup) / (labma*labma) * ((l0p*l0p) / (N*N));
	//forcepin_out.fz = forcepin_out.fz*(u0p.uz*u0p.uz*roup) / (labma*labma) * ((l0p*l0p) / (N*N));

	//forcepin_out.fx = forcepin_out.fx*(u0p.ux)*roup / (labma);
	//forcepin_out.fy = forcepin_out.fy*(u0p.uy)*roup / (labma);
	//forcepin_out.fz = forcepin_out.fz*(u0p.uz)*roup / (labma);
}

void mlPhysicalParamMapping::mlMappingViscosity_LaToPhys(float& viscosityk, float& viscosityp)
{
	viscosityp = u0p.uy * l0p * viscosityk / (labma * N);
}

void mlPhysicalParamMapping::mlMappingViscosity_LaToPhys(float& viscosityin_out)
{
	viscosityin_out = u0p.uy * l0p * viscosityin_out / (labma * N);
}




void mlPhysicalParamMapping::mlMappingVelocity_PhysToLa(mlVelocity3f& uk, mlVelocity3f& up)
{
	uk.ux = labma / (u0p.ux) * up.ux;
	uk.uy = labma / (u0p.uy) * up.uy;
	uk.uz = labma / (u0p.uz) * up.uz;
}

void mlPhysicalParamMapping::mlMappingVelocity_PhysToLa(mlVelocity3f& uin_out)
{
	uin_out.ux = labma / (u0p.ux) * uin_out.ux;
	uin_out.uy = labma / (u0p.uy) * uin_out.uy;
	uin_out.uz = labma / (u0p.uz) * uin_out.uz;
}

void mlPhysicalParamMapping::mlMappingForce_PhysToLa(mlVector3f& forcek, mlVector3f& forcep)
{
	forcek[0] = forcep[0] * (l0p * labma * labma) / (N * (u0p.ux * u0p.ux) * roup);
	forcek[1] = forcep[1] * (l0p * labma * labma) / (N * (u0p.uy * u0p.uy) * roup);
	forcek[2] = forcep[2] * (l0p * labma * labma) / (N * (u0p.uz * u0p.uz) * roup);

	/*forcek.fx = forcep.fx* (labma*labma) / ((u0p.ux*u0p.ux)*roup)*((N*N) / (l0p*l0p));
	forcek.fy = forcep.fy* (labma*labma) / ((u0p.uy*u0p.uy)*roup)*((N*N) / (l0p*l0p));
	forcek.fz = forcep.fz* (labma*labma) / ((u0p.uz*u0p.uz)*roup)*((N*N) / (l0p*l0p));*/

	/*forcek.fx = forcep.fx*(labma) / (u0p.ux*roup);
	forcek.fy = forcep.fy*(labma) / (u0p.ux*roup);
	forcek.fz = forcep.fz*(labma) / (u0p.ux*roup);*/

}

void mlPhysicalParamMapping::mlMappingForce_PhysToLa(mlVector3f& forcein_out)
{
	forcein_out[0] = forcein_out[0] * (l0p * labma * labma) / (N * (u0p.ux * u0p.ux) * roup);
	forcein_out[1] = forcein_out[1] * (l0p * labma * labma) / (N * (u0p.uy * u0p.uy) * roup);
	forcein_out[2] = forcein_out[2] * (l0p * labma * labma) / (N * (u0p.uz * u0p.uz) * roup);

	/*forcein_out.fx = forcein_out.fx* (labma*labma) / ((u0p.ux*u0p.ux)*roup) * ((N*N) / (l0p*l0p));
	forcein_out.fy = forcein_out.fy* (labma*labma) / ((u0p.uy*u0p.uy)*roup) * ((N*N) / (l0p*l0p));
	forcein_out.fz = forcein_out.fz* (labma*labma) / ((u0p.uz*u0p.uz)*roup) * ((N*N) / (l0p*l0p));*/

	//forcein_out.fx = forcein_out.fx*(labma) / (u0p.ux*roup);
	//forcein_out.fy = forcein_out.fy*(labma) / (u0p.ux*roup);
	//forcein_out.fz = forcein_out.fz*(labma) / (u0p.ux*roup);
}

void mlPhysicalParamMapping::mlMappingViscosity_PhysToLa(float& viscosityk, float& viscosityp)
{
	viscosityk = labma * N * viscosityp / (u0p.uy * l0p);
}

void mlPhysicalParamMapping::mlMappingViscosity_PhysToLa(float& viscosityin_out)
{
	viscosityin_out = labma * N * viscosityin_out / (u0p.uy * l0p);
}

void mlPhysicalParamMapping::mlMappingPressure_LaToPhys()
{


}

void mlPhysicalParamMapping::mlsetInitVariable
(
	mlVelocity3f _u0p,
	float _labma,
	float _l0p,
	float _N,
	float _roup
)
{
	u0p = _u0p;
	labma = _labma;
	l0p = _l0p;
	N = _N;
	roup = _roup;
}

void mlPhysicalParamMapping::mlCalGravity_phy(float& g, float length)
{
	g = (u0p.uy * u0p.uy) / (2 * length);
}

void mlPhysicalParamMapping::mlCalGravity_La(float& gL, float& gp)
{
	gL = (labma * labma * l0p) / (u0p.uy * u0p.uy * N) * gp;
}

void mlPhysicalParamMapping::mlCaltp(float& t)
{
	t = labma * l0p / (u0p.uy * N);
}

mlPhysicalParamMapping& mlPhysicalParamMapping::operator=(const mlPhysicalParamMapping& mapping)
{
	this->u0p = mapping.u0p;
	this->labma = mapping.labma;
	this->l0p = mapping.l0p;
	this->N = mapping.N;
	this->roup = mapping.roup;
	return (*this);
}

#endif // !_MLPHYSCIALMAPPINGH_
