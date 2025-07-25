#pragma once

#ifndef _GVLDATATYPE_
#define _GVLDATATYPE_

#include "mlCoreWin.h"
//#include "mlcudaComon.h"

//////////////////////////////position definition///////////////////////////////////
template<class T>
class GVLPos1D
{
public:
	T x;

	MLFUNC_TYPE GVLPos1D<T> & operator +=(const GVLPos1D<T> &p);
	MLFUNC_TYPE GVLPos1D<T> operator +(const GVLPos1D<T> &p);
	MLFUNC_TYPE GVLPos1D<T> & operator +=(const T &v);
	MLFUNC_TYPE GVLPos1D<T> operator +(const T &v);

	MLFUNC_TYPE GVLPos1D<T> & operator -=(const GVLPos1D<T> &p);
	MLFUNC_TYPE GVLPos1D<T> operator -(const GVLPos1D<T> &p);
	MLFUNC_TYPE GVLPos1D<T> & operator -=(const T &v);
	MLFUNC_TYPE GVLPos1D<T> operator -(const T &v);

	MLFUNC_TYPE GVLPos1D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLPos1D<T> operator *(const T &v);

	MLFUNC_TYPE GVLPos1D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLPos1D<T> operator /(const T &v);

	MLFUNC_TYPE GVLPos1D<T> & operator = (const GVLPos1D<T> &p);
	MLFUNC_TYPE bool operator == (const GVLPos1D<T> &p);
	MLFUNC_TYPE bool operator != (const GVLPos1D<T> &p);

	MLFUNC_TYPE GVLPos1D();
	MLFUNC_TYPE GVLPos1D(T x);
	MLFUNC_TYPE GVLPos1D(const GVLPos1D<T> &p);
};

typedef GVLPos1D<int> GVLPos1i;
typedef GVLPos1D<long> GVLPos1l;
typedef GVLPos1D<REAL> GVLPos1f;
typedef GVLPos1D<double> GVLPos1d;
/////////////////////////////////////////////////////////////////////////
template<class T>
MLFUNC_TYPE GVLPos1D<T>::GVLPos1D()
{
	x = 0;
}

template<class T>
MLFUNC_TYPE GVLPos1D<T>::GVLPos1D(T x)
{
	this->x = x;
}

template<class T>
MLFUNC_TYPE GVLPos1D<T>::GVLPos1D(const GVLPos1D<T> &p)
{
	x = p.x;
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> & GVLPos1D<T>::operator = (const GVLPos1D<T> &p)
{
	x = p.x;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLPos1D<T>::operator == (const GVLPos1D<T> &p)
{
	if (x == p.x)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLPos1D<T>::operator != (const GVLPos1D<T> &p)
{
	return !((*this) == p);
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> & GVLPos1D<T>::operator +=(const GVLPos1D<T> &p)
{
	x += p.x;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> GVLPos1D<T>::operator +(const GVLPos1D<T> &p)
{
	GVLPos1D<T> temp(*this);
	return temp += p;
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> & GVLPos1D<T>::operator +=(const T &v)
{
	x += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> GVLPos1D<T>::operator +(const T &v)
{
	GVLPos1D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> & GVLPos1D<T>::operator -=(const GVLPos1D<T> &p)
{
	x -= p.x;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> GVLPos1D<T>::operator -(const GVLPos1D<T> &p)
{
	GVLPos1D<T> temp(*this);
	return temp -= p;
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> & GVLPos1D<T>::operator -=(const T &v)
{
	x -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> GVLPos1D<T>::operator -(const T &v)
{
	GVLPos1D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> & GVLPos1D<T>::operator *=(const T &v)
{
	x *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> GVLPos1D<T>::operator *(const T &v)
{
	GVLPos1D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> & GVLPos1D<T>::operator /=(const T &v)
{
	x /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos1D<T> GVLPos1D<T>::operator /(const T &v)
{
	GVLPos1D<T> temp(*this);
	return temp /= v;
}
///////////////////////////////GVLPos2D/////////////////////////////////////
template<class T>
class GVLPos2D
{
public:
	T x, y;

	MLFUNC_TYPE GVLPos2D<T> & operator +=(const GVLPos2D<T> &p);
	MLFUNC_TYPE GVLPos2D<T> operator +(const GVLPos2D<T> &p);
	MLFUNC_TYPE GVLPos2D<T> & operator +=(const T &v);
	MLFUNC_TYPE GVLPos2D<T> operator +(const T &v);

	MLFUNC_TYPE GVLPos2D<T> & operator -=(const GVLPos2D<T> &p);
	MLFUNC_TYPE GVLPos2D<T> operator -(const GVLPos2D<T> &p);
	MLFUNC_TYPE GVLPos2D<T> & operator -=(const T &v);
	MLFUNC_TYPE GVLPos2D<T> operator -(const T &v);

	MLFUNC_TYPE GVLPos2D<T> & operator *=(const GVLPos2D<T> &p);
	MLFUNC_TYPE GVLPos2D<T> operator *(const GVLPos2D<T> &p);
	MLFUNC_TYPE GVLPos2D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLPos2D<T> operator *(const T &v);

	MLFUNC_TYPE GVLPos2D<T> & operator /=(const GVLPos2D<T> &p);
	MLFUNC_TYPE GVLPos2D<T> operator /(const GVLPos2D<T> &p);
	MLFUNC_TYPE GVLPos2D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLPos2D<T> operator /(const T &v);

	MLFUNC_TYPE GVLPos2D<T> & operator = (const GVLPos2D<T> &p);
	MLFUNC_TYPE bool operator == (const GVLPos2D<T> &p);
	MLFUNC_TYPE bool operator != (const GVLPos2D<T> &p);

	MLFUNC_TYPE GVLPos2D();
	MLFUNC_TYPE GVLPos2D(const T &xy);
	MLFUNC_TYPE GVLPos2D(const T &x, const T &y);
	MLFUNC_TYPE GVLPos2D(const GVLPos2D<T> &p);
};

typedef GVLPos2D<int> GVLPos2i;
typedef GVLPos2D<long> GVLPos2l;
typedef GVLPos2D<REAL> GVLPos2f;
typedef GVLPos2D<double> GVLPos2d;
////////////////////////////////////////////////////////////////////////
template<class T>
MLFUNC_TYPE GVLPos2D<T>::GVLPos2D()
{
	x = 0; y = 0;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T>::GVLPos2D(const T &xy)
{
	this->x = xy;
	this->y = xy;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T>::GVLPos2D(const T &x, const T &y)
{
	this->x = x;
	this->y = y;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T>::GVLPos2D(const GVLPos2D<T> &p)
{
	x = p.x; y = p.y;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> & GVLPos2D<T>::	operator = (const GVLPos2D<T> &p)
{
	x = p.x; y = p.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLPos2D<T>::operator == (const GVLPos2D<T> &p)
{
	if (x == p.x&&y == p.y)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLPos2D<T>::operator != (const GVLPos2D<T> &p)
{
	return !((*this) == p);
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> & GVLPos2D<T>::operator +=(const GVLPos2D<T> &p)
{
	x += p.x; y += p.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> GVLPos2D<T>::operator +(const GVLPos2D<T> &p)
{
	GVLPos2D<T> temp(*this);
	return temp += p;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> & GVLPos2D<T>::operator +=(const T &v)
{
	x += v; y += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> GVLPos2D<T>::operator +(const T &v)
{
	GVLPos2D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> & GVLPos2D<T>::operator -=(const GVLPos2D<T> &p)
{
	x -= p.x; y -= p.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> GVLPos2D<T>::operator -(const GVLPos2D<T> &p)
{
	GVLPos2D<T> temp(*this);
	return temp -= p;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> & GVLPos2D<T>::operator -=(const T &v)
{
	x -= v; y -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> GVLPos2D<T>::operator -(const T &v)
{
	GVLPos2D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> & GVLPos2D<T>::operator *=(const GVLPos2D<T> &p)
{
	x *= p.x; y *= p.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> GVLPos2D<T>::operator *(const GVLPos2D<T> &p)
{
	GVLPos2D<T> temp(*this);
	return temp *= p;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> & GVLPos2D<T>::operator *=(const T &v)
{
	x *= v; y *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> GVLPos2D<T>::operator *(const T &v)
{
	GVLPos2D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> & GVLPos2D<T>::operator /=(const GVLPos2D<T> &p)
{
	x /= p.x; y /= p.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> GVLPos2D<T>::operator /(const GVLPos2D<T> &p)
{
	GVLPos2D<T> temp(*this);
	return temp /= p;
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> & GVLPos2D<T>::operator /=(const T &v)
{
	x /= v; y /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos2D<T> GVLPos2D<T>::operator /(const T &v)
{
	GVLPos2D<T> temp(*this);
	return temp /= v;
}
////////////////////////////////GVLPos3D//////////////////////////////////////
template<class T>
class GVLPos3D
{
public:
	T x, y, z;

	MLFUNC_TYPE GVLPos3D<T> & operator +=(const GVLPos3D<T> &p);
	MLFUNC_TYPE GVLPos3D<T> operator +(const GVLPos3D<T> &p);
	MLFUNC_TYPE GVLPos3D<T> & operator +=(const T &v);
	MLFUNC_TYPE GVLPos3D<T> operator +(const T &v);

	MLFUNC_TYPE GVLPos3D<T> & operator -=(const GVLPos3D<T> &p);
	MLFUNC_TYPE GVLPos3D<T> operator -(const GVLPos3D<T> &p);
	MLFUNC_TYPE GVLPos3D<T> & operator -=(const T &v);
	MLFUNC_TYPE GVLPos3D<T> operator -(const T &v);

	MLFUNC_TYPE GVLPos3D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLPos3D<T> operator *(const T &v);

	MLFUNC_TYPE GVLPos3D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLPos3D<T> operator /(const T &v);

	MLFUNC_TYPE GVLPos3D<T> & operator = (const GVLPos3D<T> &p);
	MLFUNC_TYPE bool operator == (const GVLPos3D<T> &p);
	MLFUNC_TYPE bool operator != (const GVLPos3D<T> &p);

	MLFUNC_TYPE GVLPos3D();
	MLFUNC_TYPE GVLPos3D(T x, T y, T z);
	MLFUNC_TYPE GVLPos3D(const GVLPos3D<T> &p);
};

typedef GVLPos3D<int> GVLPos3i;
typedef GVLPos3D<long> GVLPos3l;
typedef GVLPos3D<REAL> GVLPos3f;
typedef GVLPos3D<double> GVLPos3d;

//////////////////////////////////////////////////////////////////
template<class T>
MLFUNC_TYPE GVLPos3D<T>::GVLPos3D()
{
	x = 0; y = 0; z = 0;
}

template<class T>
MLFUNC_TYPE GVLPos3D<T>::GVLPos3D(T x, T y, T z)
{
	this->x = x;
	this->y = y;
	this->z = z;
}

template<class T>
MLFUNC_TYPE GVLPos3D<T>::GVLPos3D(const GVLPos3D<T> &p)
{
	x = p.x; y = p.y; z = p.z;
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> & GVLPos3D<T>::	operator = (const GVLPos3D<T> &p)
{
	x = p.x; y = p.y; z = p.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLPos3D<T>::operator == (const GVLPos3D<T> &p)
{
	if (x == p.x&&y == p.y&&z == p.z)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLPos3D<T>::operator != (const GVLPos3D<T> &p)
{
	return !((*this) == p);
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> & GVLPos3D<T>::operator +=(const GVLPos3D<T> &p)
{
	x += p.x; y += p.y; z += p.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> GVLPos3D<T>::operator +(const GVLPos3D<T> &p)
{
	GVLPos3D<T> temp(*this);
	return temp += p;
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> & GVLPos3D<T>::operator +=(const T &v)
{
	x += v; y += v; z += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> GVLPos3D<T>::operator +(const T &v)
{
	GVLPos3D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> & GVLPos3D<T>::operator -=(const GVLPos3D<T> &p)
{
	x -= p.x; y -= p.y; z -= p.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> GVLPos3D<T>::operator -(const GVLPos3D<T> &p)
{
	GVLPos3D<T> temp(*this);
	return temp -= p;
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> & GVLPos3D<T>::operator -=(const T &v)
{
	x -= v; y -= v; z -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> GVLPos3D<T>::operator -(const T &v)
{
	GVLPos3D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> & GVLPos3D<T>::operator *=(const T &v)
{
	x *= v; y *= v; z *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> GVLPos3D<T>::operator *(const T &v)
{
	GVLPos3D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> & GVLPos3D<T>::operator /=(const T &v)
{
	x /= v; y /= v; z /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPos3D<T> GVLPos3D<T>::operator /(const T &v)
{
	GVLPos3D<T> temp(*this);
	return temp /= v;
}
//////////////////////////////point definition//////////////////////////
template<class T>
class GVLPoint1D
{
public:
	T x;

	MLFUNC_TYPE GVLPoint1D<T> & operator +=(const GVLPoint1D<T> &p);
	MLFUNC_TYPE GVLPoint1D<T> operator +(const GVLPoint1D<T> &p);
	MLFUNC_TYPE GVLPoint1D<T> & operator +=(const T &v);
	MLFUNC_TYPE GVLPoint1D<T> operator +(const T &v);

	MLFUNC_TYPE GVLPoint1D<T> & operator -=(const GVLPoint1D<T> &p);
	MLFUNC_TYPE GVLPoint1D<T> operator -(const GVLPoint1D<T> &p);
	MLFUNC_TYPE GVLPoint1D<T> & operator -=(const T &v);
	MLFUNC_TYPE GVLPoint1D<T> operator -(const T &v);

	MLFUNC_TYPE GVLPoint1D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLPoint1D<T> operator *(const T &v);

	MLFUNC_TYPE GVLPoint1D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLPoint1D<T> operator /(const T &v);

	MLFUNC_TYPE GVLPoint1D<T> & operator = (const GVLPoint1D<T> &p);
	MLFUNC_TYPE bool operator == (const GVLPoint1D<T> &p);
	MLFUNC_TYPE bool operator != (const GVLPoint1D<T> &p);

	MLFUNC_TYPE GVLPoint1D();
	MLFUNC_TYPE GVLPoint1D(T x);
	MLFUNC_TYPE GVLPoint1D(const GVLPoint1D<T> &p);
};

typedef GVLPoint1D<int> GVLPoint1i;
typedef GVLPoint1D<long> GVLPoint1l;
typedef GVLPoint1D<REAL> GVLPoint1f;
typedef GVLPoint1D<double> GVLPoint1d;

//////////////////////////////////////////////////////////////////////
template<class T>
MLFUNC_TYPE GVLPoint1D<T>::GVLPoint1D()
{
	x = 0;
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T>::GVLPoint1D(T x)
{
	this->x = x;
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T>::GVLPoint1D(const GVLPoint1D<T> &p)
{
	x = p.x;
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> & GVLPoint1D<T>::operator = (const GVLPoint1D<T> &p)
{
	x = p.x;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLPoint1D<T>::operator == (const GVLPoint1D<T> &p)
{
	if (x == p.x)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLPoint1D<T>::operator != (const GVLPoint1D<T> &p)
{
	return !((*this) == p);
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> & GVLPoint1D<T>::operator +=(const GVLPoint1D<T> &p)
{
	x += p.x;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> GVLPoint1D<T>::operator +(const GVLPoint1D<T> &p)
{
	GVLPoint1D<T> temp(*this);
	return temp += p;
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> & GVLPoint1D<T>::operator +=(const T &v)
{
	x += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> GVLPoint1D<T>::operator +(const T &v)
{
	GVLPoint1D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> & GVLPoint1D<T>::operator -=(const GVLPoint1D<T> &p)
{
	x -= p.x;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> GVLPoint1D<T>::operator -(const GVLPoint1D<T> &p)
{
	GVLPoint1D<T> temp(*this);
	return temp -= p;
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> & GVLPoint1D<T>::operator -=(const T &v)
{
	x -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> GVLPoint1D<T>::operator -(const T &v)
{
	GVLPoint1D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> & GVLPoint1D<T>::operator *=(const T &v)
{
	x *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> GVLPoint1D<T>::operator *(const T &v)
{
	GVLPoint1D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> & GVLPoint1D<T>::operator /=(const T &v)
{
	x /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint1D<T> GVLPoint1D<T>::operator /(const T &v)
{
	GVLPoint1D<T> temp(*this);
	return temp /= v;
}

//////////////////////////////////////////////////////////////////////
template<class T>
class GVLPoint2D
{
public:
	T x, y;

	MLFUNC_TYPE GVLPoint2D<T> & operator +=(const GVLPoint2D<T> &p);
	MLFUNC_TYPE GVLPoint2D<T> operator +(const GVLPoint2D<T> &p);
	MLFUNC_TYPE GVLPoint2D<T> & operator +=(const T &v);
	MLFUNC_TYPE GVLPoint2D<T> operator +(const T &v);

	MLFUNC_TYPE GVLPoint2D<T> & operator -=(const GVLPoint2D<T> &p);
	MLFUNC_TYPE GVLPoint2D<T> operator -(const GVLPoint2D<T> &p);
	MLFUNC_TYPE GVLPoint2D<T> & operator -=(const T &v);
	MLFUNC_TYPE GVLPoint2D<T> operator -(const T &v);

	MLFUNC_TYPE GVLPoint2D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLPoint2D<T> operator *(const T &v);

	MLFUNC_TYPE GVLPoint2D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLPoint2D<T> operator /(const T &v);

	MLFUNC_TYPE GVLPoint2D<T> & operator = (const GVLPoint2D<T> &p);
	MLFUNC_TYPE bool operator == (const GVLPoint2D<T> &p);
	MLFUNC_TYPE bool operator != (const GVLPoint2D<T> &p);

	MLFUNC_TYPE GVLPoint2D();
	MLFUNC_TYPE GVLPoint2D(T x, T y);
	MLFUNC_TYPE GVLPoint2D(const GVLPoint2D<T> &p);
};

typedef GVLPoint2D<int> GVLPoint2i;
typedef GVLPoint2D<long> GVLPoint2l;
typedef GVLPoint2D<REAL> GVLPoint2f;
typedef GVLPoint2D<double> GVLPoint2d;

/////////////////////////////////////////////////////////////////////
template<class T>
MLFUNC_TYPE GVLPoint2D<T>::GVLPoint2D()
{
	x = 0; y = 0;
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T>::GVLPoint2D(T x, T y)
{
	this->x = x;
	this->y = y;
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T>::GVLPoint2D(const GVLPoint2D<T> &p)
{
	x = p.x; y = p.y;
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> & GVLPoint2D<T>::	operator = (const GVLPoint2D<T> &p)
{
	x = p.x; y = p.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLPoint2D<T>::operator == (const GVLPoint2D<T> &p)
{
	if (x == p.x&&y == p.y)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLPoint2D<T>::operator != (const GVLPoint2D<T> &p)
{
	return !((*this) == p);
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> & GVLPoint2D<T>::operator +=(const GVLPoint2D<T> &p)
{
	x += p.x; y += p.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> GVLPoint2D<T>::operator +(const GVLPoint2D<T> &p)
{
	GVLPoint2D<T> temp(*this);
	return temp += p;
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> & GVLPoint2D<T>::operator +=(const T &v)
{
	x += v; y += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> GVLPoint2D<T>::operator +(const T &v)
{
	GVLPoint2D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> & GVLPoint2D<T>::operator -=(const GVLPoint2D<T> &p)
{
	x -= p.x; y -= p.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> GVLPoint2D<T>::operator -(const GVLPoint2D<T> &p)
{
	GVLPoint2D<T> temp(*this);
	return temp -= p;
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> & GVLPoint2D<T>::operator -=(const T &v)
{
	x -= v; y -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> GVLPoint2D<T>::operator -(const T &v)
{
	GVLPoint2D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> & GVLPoint2D<T>::operator *=(const T &v)
{
	x *= v; y *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> GVLPoint2D<T>::operator *(const T &v)
{
	GVLPoint2D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> & GVLPoint2D<T>::operator /=(const T &v)
{
	x /= v; y /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint2D<T> GVLPoint2D<T>::operator /(const T &v)
{
	GVLPoint2D<T> temp(*this);
	return temp /= v;
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
class GVLPoint3D
{
public:
	T x, y, z;

	MLFUNC_TYPE GVLPoint3D<T> & operator +=(const GVLPoint3D<T> &p);
	MLFUNC_TYPE GVLPoint3D<T> operator +(const GVLPoint3D<T> &p);
	MLFUNC_TYPE GVLPoint3D<T> & operator +=(const T &v);
	MLFUNC_TYPE GVLPoint3D<T> operator +(const T &v);

	MLFUNC_TYPE GVLPoint3D<T> & operator -=(const GVLPoint3D<T> &p);
	MLFUNC_TYPE GVLPoint3D<T> operator -(const GVLPoint3D<T> &p);
	MLFUNC_TYPE GVLPoint3D<T> & operator -=(const T &v);
	MLFUNC_TYPE GVLPoint3D<T> operator -(const T &v);

	MLFUNC_TYPE GVLPoint3D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLPoint3D<T> operator *(const T &v);

	MLFUNC_TYPE GVLPoint3D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLPoint3D<T> operator /(const T &v);

	MLFUNC_TYPE GVLPoint3D<T> & operator = (const GVLPoint3D<T> &p);
	MLFUNC_TYPE bool operator == (const GVLPoint3D<T> &p);
	MLFUNC_TYPE bool operator != (const GVLPoint3D<T> &p);

	MLFUNC_TYPE GVLPoint3D();
	MLFUNC_TYPE GVLPoint3D(T x, T y, T z);
	MLFUNC_TYPE GVLPoint3D(const GVLPoint3D<T> &p);
};

typedef GVLPoint3D<int> GVLPoint3i;
typedef GVLPoint3D<long> GVLPoint3l;
typedef GVLPoint3D<REAL> GVLPoint3f;
typedef GVLPoint3D<double> GVLPoint3d;

/////////////////////////////////////////////////////////////////////////
template<class T>
MLFUNC_TYPE GVLPoint3D<T>::GVLPoint3D()
{
	x = 0; y = 0; z = 0;
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T>::GVLPoint3D(T x, T y, T z)
{
	this->x = x;
	this->y = y;
	this->z = z;
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T>::GVLPoint3D(const GVLPoint3D<T> &p)
{
	x = p.x; y = p.y; z = p.z;
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> & GVLPoint3D<T>::	operator = (const GVLPoint3D<T> &p)
{
	x = p.x; y = p.y; z = p.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLPoint3D<T>::operator == (const GVLPoint3D<T> &p)
{
	if (x == p.x&&y == p.y&&z == p.z)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLPoint3D<T>::operator != (const GVLPoint3D<T> &p)
{
	return !((*this) == p);
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> & GVLPoint3D<T>::operator +=(const GVLPoint3D<T> &p)
{
	x += p.x; y += p.y; z += p.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> GVLPoint3D<T>::operator +(const GVLPoint3D<T> &p)
{
	GVLPoint3D<T> temp(*this);
	return temp += p;
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> & GVLPoint3D<T>::operator +=(const T &v)
{
	x += v; y += v; z += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> GVLPoint3D<T>::operator +(const T &v)
{
	GVLPoint3D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> & GVLPoint3D<T>::operator -=(const GVLPoint3D<T> &p)
{
	x -= p.x; y -= p.y; z -= p.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> GVLPoint3D<T>::operator -(const GVLPoint3D<T> &p)
{
	GVLPoint3D<T> temp(*this);
	return temp -= p;
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> & GVLPoint3D<T>::operator -=(const T &v)
{
	x -= v; y -= v; z -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> GVLPoint3D<T>::operator -(const T &v)
{
	GVLPoint3D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> & GVLPoint3D<T>::operator *=(const T &v)
{
	x *= v; y *= v; z *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> GVLPoint3D<T>::operator *(const T &v)
{
	GVLPoint3D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> & GVLPoint3D<T>::operator /=(const T &v)
{
	x /= v; y /= v; z /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint3D<T> GVLPoint3D<T>::operator /(const T &v)
{
	GVLPoint3D<T> temp(*this);
	return temp /= v;
}
//////////////////////////////////////////////////////////////////////
template<class T>
class GVLPoint4D
{
public:
	T x, y, z, w;

	MLFUNC_TYPE GVLPoint4D<T> & operator +=(const GVLPoint4D<T> &p);
	MLFUNC_TYPE GVLPoint4D<T> operator +(const GVLPoint4D<T> &p);
	MLFUNC_TYPE GVLPoint4D<T> & operator +=(const T &v);
	MLFUNC_TYPE GVLPoint4D<T> operator +(const T &v);

	MLFUNC_TYPE GVLPoint4D<T> & operator -=(const GVLPoint4D<T> &p);
	MLFUNC_TYPE GVLPoint4D<T> operator -(const GVLPoint4D<T> &p);
	MLFUNC_TYPE GVLPoint4D<T> & operator -=(const T &v);
	MLFUNC_TYPE GVLPoint4D<T> operator -(const T &v);

	MLFUNC_TYPE GVLPoint4D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLPoint4D<T> operator *(const T &v);

	MLFUNC_TYPE GVLPoint4D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLPoint4D<T> operator /(const T &v);

	MLFUNC_TYPE GVLPoint4D<T> & operator = (const GVLPoint4D<T> &p);
	MLFUNC_TYPE bool operator == (const GVLPoint4D<T> &p);
	MLFUNC_TYPE bool operator != (const GVLPoint4D<T> &p);

	MLFUNC_TYPE GVLPoint4D();
	MLFUNC_TYPE GVLPoint4D(T x, T y, T z, T w);
	MLFUNC_TYPE GVLPoint4D(const GVLPoint4D<T> &p);
};

typedef GVLPoint4D<int> GVLPoint4i;
typedef GVLPoint4D<long> GVLPoint4l;
typedef GVLPoint4D<REAL> GVLPoint4f;
typedef GVLPoint4D<double> GVLPoint4d;

////////////////////////////////////////////////////////////////////////////
template<class T>
MLFUNC_TYPE GVLPoint4D<T>::GVLPoint4D()
{
	x = 0; y = 0; z = 0; w = 0;
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T>::GVLPoint4D(T x, T y, T z, T w)
{
	this->x = x;
	this->y = y;
	this->z = z;
	this->w = w;
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T>::GVLPoint4D(const GVLPoint4D<T> &p)
{
	x = p.x; y = p.y; z = p.z; w = p.w;
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> & GVLPoint4D<T>::	operator = (const GVLPoint4D<T> &p)
{
	x = p.x; y = p.y; z = p.z; w = p.w;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLPoint4D<T>::operator == (const GVLPoint4D<T> &p)
{
	if (x == p.x&&y == p.y&&z == p.z&&w == p.w)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLPoint4D<T>::operator != (const GVLPoint4D<T> &p)
{
	return !((*this) == p);
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> & GVLPoint4D<T>::operator +=(const GVLPoint4D<T> &p)
{
	x += p.x; y += p.y; z += p.z; w += p.w;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> GVLPoint4D<T>::operator +(const GVLPoint4D<T> &p)
{
	GVLPoint4D<T> temp(*this);
	return temp += p;
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> & GVLPoint4D<T>::operator +=(const T &v)
{
	x += v; y += v; z += v; w += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> GVLPoint4D<T>::operator +(const T &v)
{
	GVLPoint4D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> & GVLPoint4D<T>::operator -=(const GVLPoint4D<T> &p)
{
	x -= p.x; y -= p.y; z -= p.z; w -= p.w;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> GVLPoint4D<T>::operator -(const GVLPoint4D<T> &p)
{
	GVLPoint4D<T> temp(*this);
	return temp -= p;
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> & GVLPoint4D<T>::operator -=(const T &v)
{
	x -= v; y -= v; z -= v; w -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> GVLPoint4D<T>::operator -(const T &v)
{
	GVLPoint4D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> & GVLPoint4D<T>::operator *=(const T &v)
{
	x *= v; y *= v; z *= v; w *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> GVLPoint4D<T>::operator *(const T &v)
{
	GVLPoint4D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> & GVLPoint4D<T>::operator /=(const T &v)
{
	x /= v; y /= v; z /= v; w /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLPoint4D<T> GVLPoint4D<T>::operator /(const T &v)
{
	GVLPoint4D<T> temp(*this);
	return temp /= v;
}
///////////////////point pair////////////////////

template<class T>
struct GVLPointPair1D
{
	GVLPoint1D<T> pt_src;
	GVLPoint1D<T> pt_des;
};

typedef GVLPointPair1D<int> GVLPointPair1i;
typedef GVLPointPair1D<long> GVLPointPair1l;
typedef GVLPointPair1D<REAL> GVLPointPair1f;
typedef GVLPointPair1D<double> GVLPointPair1d;

template<class T>
struct GVLPointPair2D
{
	GVLPoint2D<T> pt_src;
	GVLPoint2D<T> pt_des;
};

typedef GVLPointPair2D<int> GVLPointPair2i;
typedef GVLPointPair2D<long> GVLPointPair2l;
typedef GVLPointPair2D<REAL> GVLPointPair2f;
typedef GVLPointPair2D<double> GVLPointPair2d;

template<class T>
struct GVLPointPair3D
{
	GVLPoint3D<T> pt_src;
	GVLPoint3D<T> pt_des;
};

typedef GVLPointPair3D<int> GVLPointPair3i;
typedef GVLPointPair3D<long> GVLPointPair3l;
typedef GVLPointPair3D<REAL> GVLPointPair3f;
typedef GVLPointPair3D<double> GVLPointPair3d;

template<class T>
struct GVLPointPair4D
{
	GVLPoint4D<T> pt_src;
	GVLPoint4D<T> pt_des;
};

typedef GVLPointPair4D<int> GVLPointPair4i;
typedef GVLPointPair4D<long> GVLPointPair4l;
typedef GVLPointPair4D<REAL> GVLPointPair4f;
typedef GVLPointPair4D<double> GVLPointPair4d;

////////////////////////////////rectangle definition///////////////////////////
template<class T>
class GVLRect2D
{
public:
	T x, y, width, height;

	MLFUNC_TYPE GVLRect2D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLRect2D<T> operator *(const T &v);
	MLFUNC_TYPE GVLRect2D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLRect2D<T> operator /(const T &v);
	MLFUNC_TYPE GVLRect2D<T> & operator = (const GVLRect2D<T> &rect);
	MLFUNC_TYPE bool operator == (const GVLRect2D<T> &rect);
	MLFUNC_TYPE bool operator != (const GVLRect2D<T> &rect);

	MLFUNC_TYPE bool IsPointIn(const GVLPoint2D<T> &pt);
	MLFUNC_TYPE bool IsPosIn(const GVLPos2D<T> &pos);

	MLFUNC_TYPE GVLRect2D();
	MLFUNC_TYPE GVLRect2D(T x, T y, T width, T height);
	MLFUNC_TYPE GVLRect2D(const GVLRect2D<T> &rect);
};

typedef GVLRect2D<int> GVLRect2i;
typedef GVLRect2D<long> GVLRect2l;
typedef GVLRect2D<REAL> GVLRect2f;
typedef GVLRect2D<double> GVLRect2d;

//////////////////implementation of GVLRect2D//////////////////////////
template<class T>
MLFUNC_TYPE GVLRect2D<T>::GVLRect2D()
{
	x = 0; y = 0; width = 0; height = 0;
}

template<class T>
MLFUNC_TYPE GVLRect2D<T>::GVLRect2D(T x, T y, T width, T height)
{
	this->x = x; this->y = y;
	this->width = width; this->height = height;
}

template<class T>
MLFUNC_TYPE GVLRect2D<T>::GVLRect2D(const GVLRect2D<T> &rect)
{
	x = rect.x; y = rect.y; width = rect.width; height = rect.height;
}

template<class T>
MLFUNC_TYPE GVLRect2D<T> & GVLRect2D<T>::operator = (const GVLRect2D<T> &rect)
{
	x = rect.x; y = rect.y; width = rect.width; height = rect.height;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLRect2D<T>::operator == (const GVLRect2D<T> &rect)
{
	if (x == rect.x&&y == rect.y&&width == rect.width&&height == rect.height)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLRect2D<T>::operator != (const GVLRect2D<T> &rect)
{
	return !((*this) == rect);
}

template<class T>
MLFUNC_TYPE GVLRect2D<T> & GVLRect2D<T>::operator *=(const T &v)
{
	width *= v; height *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLRect2D<T> GVLRect2D<T>::operator *(const T &v)
{
	GVLRect2D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLRect2D<T> & GVLRect2D<T>::operator /=(const T &v)
{
	width /= v; height /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLRect2D<T> GVLRect2D<T>::operator /(const T &v)
{
	GVLRect2D<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE bool GVLRect2D<T>::IsPointIn(const GVLPoint2D<T> &pt)
{
	if (pt.x >= x&&pt.x <= x + width&&pt.y >= y&&pt.y <= y + height)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLRect2D<T>::IsPosIn(const GVLPos2D<T> &pos)
{
	if (pos.x >= x&&pos.x <= x + width&&pos.y >= y&&pos.y <= y + height)
		return true;
	else
		return false;
}

/////////////////////////////////////////////////////////////////
template<class T>
class GVLRect3D
{
public:
	T x, y, z, width, height, depth;

	MLFUNC_TYPE GVLRect3D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLRect3D<T> operator *(const T &v);
	MLFUNC_TYPE GVLRect3D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLRect3D<T> operator /(const T &v);
	MLFUNC_TYPE bool operator == (const GVLRect3D<T> &rect);
	MLFUNC_TYPE bool operator != (const GVLRect3D<T> &rect);
	MLFUNC_TYPE GVLRect3D<T> & operator = (const GVLRect3D<T> &rect);

	MLFUNC_TYPE GVLRect3D();
	MLFUNC_TYPE GVLRect3D(T x, T y, T z, T width, T height, T depth);
	MLFUNC_TYPE GVLRect3D(const GVLRect3D<T> &rect);
};

typedef GVLRect3D<int> GVLRect3i;
typedef GVLRect3D<long> GVLRect3l;
typedef GVLRect3D<REAL> GVLRect3f;
typedef GVLRect3D<double> GVLRect3d;

//////////////////implementation of GVLRect3D//////////////////////////
template<class T>
MLFUNC_TYPE GVLRect3D<T>::GVLRect3D()
{
	x = 0; y = 0; z = 0; width = 0; height = 0; depth = 0;
}

template<class T>
MLFUNC_TYPE GVLRect3D<T>::GVLRect3D(T x, T y, T z, T width, T height, T depth)
{
	this->x = x; this->y = y; this->z = z;
	this->width = width; this->height = height; this->depth = depth;
}

template<class T>
MLFUNC_TYPE GVLRect3D<T>::GVLRect3D(const GVLRect3D<T> &rect)
{
	x = rect.x; y = rect.y; z = rect.z; width = rect.width; height = rect.height; depth = rect.depth;
}

template<class T>
MLFUNC_TYPE GVLRect3D<T> & GVLRect3D<T>::operator = (const GVLRect3D<T> &rect)
{
	x = rect.x; y = rect.y; z = rect.z; width = rect.width; height = rect.height; depth = rect.depth;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLRect3D<T>::operator == (const GVLRect3D<T> &rect)
{
	if (x == rect.x&&y == rect.y&&z == rect.z&&width == rect.width&&height == rect.height&&depth == rect.depth)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLRect3D<T>::operator != (const GVLRect3D<T> &rect)
{
	return !((*this) == rect);
}

template<class T>
MLFUNC_TYPE GVLRect3D<T> & GVLRect3D<T>::operator *=(const T &v)
{
	width *= v; height *= v; depth *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLRect3D<T> GVLRect3D<T>::operator *(const T &v)
{
	GVLRect3D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLRect3D<T> & GVLRect3D<T>::operator /=(const T &v)
{
	width /= v; height /= v; depth /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLRect3D<T> GVLRect3D<T>::operator /(const T &v)
{
	GVLRect3D<T> temp(*this);
	return temp /= v;
}

////////////////////////////////////////////////////////////////////////
template<class T> //T is limited to int and long
class GVLSize2D
{
public:
	T x, y;

	MLFUNC_TYPE GVLSize2D<T> & operator = (const GVLSize2D<T> &size);

	MLFUNC_TYPE bool operator == (const GVLSize2D<T> &size);
	MLFUNC_TYPE bool operator != (const GVLSize2D<T> &size);

	MLFUNC_TYPE GVLSize2D<T> & operator +=(const T &v);
	MLFUNC_TYPE GVLSize2D<T>   operator + (const T &v);
	MLFUNC_TYPE GVLSize2D<T> & operator -=(const T &v);
	MLFUNC_TYPE GVLSize2D<T>   operator - (const T &v);
	MLFUNC_TYPE GVLSize2D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLSize2D<T>   operator * (const T &v);
	MLFUNC_TYPE GVLSize2D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLSize2D<T>   operator / (const T &v);

	MLFUNC_TYPE GVLSize2D();
	MLFUNC_TYPE GVLSize2D(T x, T y);
	MLFUNC_TYPE GVLSize2D(const GVLSize2D<T> &size);
};

typedef GVLSize2D<int> GVLSize2i;
typedef GVLSize2D<long> GVLSize2l;
typedef GVLSize2D<REAL> GVLSize2f;
typedef GVLSize2D<double> GVLSize2d;

//////////////////////////////

template<class T>
MLFUNC_TYPE GVLSize2D<T>::GVLSize2D()
{
	x = y = 0;
}

template<class T>
MLFUNC_TYPE GVLSize2D<T>::GVLSize2D(T x, T y)
{
	this->x = x;
	this->y = y;
}

template<class T>
MLFUNC_TYPE GVLSize2D<T>::GVLSize2D(const GVLSize2D<T> &size)
{
	x = size.x;
	y = size.y;
}

template<class T>
MLFUNC_TYPE GVLSize2D<T> & GVLSize2D<T>::operator = (const GVLSize2D<T> &size)
{
	x = size.x;
	y = size.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLSize2D<T>::operator == (const GVLSize2D<T> &size)
{
	if (x == size.x&&y == size.y)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLSize2D<T>::operator != (const GVLSize2D<T> &size)
{
	return !((*this) == size);
}

template<class T>
MLFUNC_TYPE GVLSize2D<T> & GVLSize2D<T>::operator +=(const T &v)
{
	x += v; y += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLSize2D<T>   GVLSize2D<T>::operator + (const T &v)
{
	GVLSize2D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE GVLSize2D<T> & GVLSize2D<T>::operator -=(const T &v)
{
	x -= v; y -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLSize2D<T>   GVLSize2D<T>::operator - (const T &v)
{
	GVLSize2D<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE GVLSize2D<T> & GVLSize2D<T>::operator *=(const T &v)
{
	x *= v; y *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLSize2D<T>   GVLSize2D<T>::operator * (const T &v)
{
	GVLSize2D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLSize2D<T> & GVLSize2D<T>::operator /=(const T &v)
{
	x /= v; y /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLSize2D<T>   GVLSize2D<T>::operator / (const T &v)
{
	GVLSize2D<T> temp(*this);
	return temp /= v;
}

//////////////////////////////

template<class T>
class GVLSize3D
{
public:
	T x, y, z;

	MLFUNC_TYPE GVLSize3D<T> & operator = (const GVLSize3D<T> &size);

	MLFUNC_TYPE bool operator == (const GVLSize3D<T> &size);
	MLFUNC_TYPE bool operator != (const GVLSize3D<T> &size);

	MLFUNC_TYPE GVLSize3D<T> & operator +=(const T &v);
	MLFUNC_TYPE GVLSize3D<T>   operator + (const T &v);
	MLFUNC_TYPE GVLSize3D<T> & operator -=(const T &v);
	MLFUNC_TYPE GVLSize3D<T>   operator - (const T &v);
	MLFUNC_TYPE GVLSize3D<T> & operator *=(const T &v);
	MLFUNC_TYPE GVLSize3D<T>   operator * (const T &v);
	MLFUNC_TYPE GVLSize3D<T> & operator /=(const T &v);
	MLFUNC_TYPE GVLSize3D<T>   operator / (const T &v);

	MLFUNC_TYPE GVLSize3D();
	MLFUNC_TYPE GVLSize3D(T x, T y, T z);
	MLFUNC_TYPE GVLSize3D(const GVLSize3D<T> &size);
};

typedef GVLSize3D<int> GVLSize3i;
typedef GVLSize3D<long> GVLSize3l;
typedef GVLSize3D<REAL> GVLSize3f;
typedef GVLSize3D<double> GVLSize3d;

////////////////////////////

template<class T>
MLFUNC_TYPE GVLSize3D<T>::GVLSize3D()
{
	x = y = z = 0;
}

template<class T>
MLFUNC_TYPE GVLSize3D<T>::GVLSize3D(T width, T height, T depth)
{
	this->x = x;
	this->y = y;
	this->z = z;
}

template<class T>
MLFUNC_TYPE GVLSize3D<T>::GVLSize3D(const GVLSize3D<T> &size)
{
	x = size.x;
	y = size.y;
	z = size.z;
}

template<class T>
MLFUNC_TYPE GVLSize3D<T> & GVLSize3D<T>::operator = (const GVLSize3D<T> &size)
{
	x = size.x;
	y = size.y;
	z = size.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool GVLSize3D<T>::operator == (const GVLSize3D<T> &size)
{
	if (x == size.x&&y == size.y&&z == size.z)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool GVLSize3D<T>::operator != (const GVLSize3D<T> &size)
{
	return !((*this) == size);
}

template<class T>
MLFUNC_TYPE GVLSize3D<T> & GVLSize3D<T>::operator +=(const T &v)
{
	x += v; y += v; z += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLSize3D<T>   GVLSize3D<T>::operator + (const T &v)
{
	GVLSize3D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE GVLSize3D<T> & GVLSize3D<T>::operator -=(const T &v)
{
	x -= v; y -= v; z -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLSize3D<T>   GVLSize3D<T>::operator - (const T &v)
{
	GVLSize3D<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE GVLSize3D<T> & GVLSize3D<T>::operator *=(const T &v)
{
	x *= v; y *= v; z *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLSize3D<T>   GVLSize3D<T>::operator * (const T &v)
{
	GVLSize3D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE GVLSize3D<T> & GVLSize3D<T>::operator /=(const T &v)
{
	x /= v; y /= v; z /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE GVLSize3D<T>   GVLSize3D<T>::operator / (const T &v)
{
	GVLSize3D<T> temp(*this);
	return temp /= v;
}

/////////////////////////////////////////

template<class T>
class GVLPair
{
public:
	T i, j;

	MLFUNC_TYPE GVLPair<T> & operator = (const GVLPair<T> &pair);

	MLFUNC_TYPE GVLPair();
	MLFUNC_TYPE GVLPair(T i, T j);
	MLFUNC_TYPE GVLPair(const GVLPair &pair);
};

typedef GVLPair<int> GVLPairi;
typedef GVLPair<long> GVLPairl;
typedef GVLPair<REAL> GVLPairf;
typedef GVLPair<double> GVLPaird;

///////////////implementation////////////////

template<class T>
MLFUNC_TYPE GVLPair<T>::GVLPair()
{
	i = j = 0;
}

template<class T>
MLFUNC_TYPE GVLPair<T>::GVLPair(T i, T j)
{
	this->i = i;
	this->j = j;
}

template<class T>
MLFUNC_TYPE GVLPair<T>::GVLPair(const GVLPair &pair)
{
	i = pair.i;
	j = pair.j;
}

template<class T>
MLFUNC_TYPE GVLPair<T> & GVLPair<T>::operator = (const GVLPair<T> &pair)
{
	i = pair.i;
	j = pair.j;

	return (*this);
}



#pragma  warning (disable : 4251)

#endif //_GVLDATATYPE_