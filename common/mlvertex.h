#pragma once

#ifndef _MLVERTEX_
#define _MLVERTEX_

#include "mlvector.h"
#include "mlCoreWin.h"
//#include "mlMatrix.h"


template<class T>
class mlVertex
{
public:
	T x, y, z, w;

	MLFUNC_TYPE bool operator != (const mlVertex<T> &v);
	MLFUNC_TYPE bool operator == (const mlVertex<T> &v);

	MLFUNC_TYPE mlVertex<T> operator / (T value);
	MLFUNC_TYPE mlVertex<T> & operator /= (T value);

	MLFUNC_TYPE mlVertex<T> operator * (T value);
	MLFUNC_TYPE mlVertex<T> & operator *= (T value);
	MLFUNC_TYPE mlVertex<T> operator + (mlVector4D<T> &v);
	MLFUNC_TYPE mlVertex<T> operator + (mlVertex<T> &v);
	MLFUNC_TYPE mlVertex<T> & operator += (mlVertex<T> &v);
	MLFUNC_TYPE mlVertex<T> & operator += (mlVector4D<T> &v);
	MLFUNC_TYPE mlVector4D<T> operator - (mlVertex<T> &v);
	MLFUNC_TYPE mlVertex<T> operator - (mlVector4D<T> &v);
	MLFUNC_TYPE mlVertex<T> & operator -= (mlVertex<T> &v);
	MLFUNC_TYPE mlVertex<T> & operator -= (mlVector4D<T> &v);
	MLFUNC_TYPE mlVertex<T> & operator = (const mlVertex<T> &v);
	MLFUNC_TYPE mlVertex<T> & operator = (const T &v);
	//MLFUNC_TYPE mlVertex<T> & Transform(mlMatrix<T, 4, 4> *m_t);
	MLFUNC_TYPE mlVertex();
	MLFUNC_TYPE mlVertex(T x, T y, T z, T w = 1);
	MLFUNC_TYPE mlVertex(const mlVector4D<T> &v);
	MLFUNC_TYPE mlVertex(const mlVertex<T> &v);
};

typedef mlVertex<int> mlVertexi;
typedef mlVertex<long> mlVertexl;
typedef mlVertex<REAL> mlVertexf;
typedef mlVertex<double> mlVertexd;

/////////////////implementation of vertex//////////////////////
template<class T>
MLFUNC_TYPE mlVertex<T>::mlVertex()
{
	x = y = z = 0; w = -1;
}

template<class T>
MLFUNC_TYPE mlVertex<T>::mlVertex(T x, T y, T z, T w)
{
	this->x = x; this->y = y; this->z = z; this->w = w;
}

template<class T>
MLFUNC_TYPE mlVertex<T>::mlVertex(const mlVertex &v)
{
	x = v.x; y = v.y; z = v.z; w = v.w;
}

template<class T>
MLFUNC_TYPE mlVertex<T>::mlVertex(const mlVector4D<T> &v)
{
	x = v[0]; y = v[1]; z = v[2]; w = v[3];
}

template<class T>
MLFUNC_TYPE mlVertex<T> & mlVertex<T>:: operator = (const mlVertex<T> &v)
{
	x = v.x; y = v.y; z = v.z; w = v.w;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex<T> & mlVertex<T>:: operator = (const T &v)
{
	x = y = z = v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex<T> & mlVertex<T>::operator +=(mlVertex<T> &v)
{
	x += v.x; y += v.y; z += v.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex<T> & mlVertex<T>:: operator += (mlVector4D<T> &v)
{
	x += v[0]; y += v[1]; z += v[2];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex<T> mlVertex<T>:: operator + (mlVector4D<T> &v)
{
	mlVertex<T> temp(*this);
	return (temp += v);
}

template<class T>
MLFUNC_TYPE mlVertex<T> mlVertex<T>:: operator + (mlVertex<T> &v)
{
	mlVertex<T> temp(*this);
	return (temp += v);
}

template<class T>
MLFUNC_TYPE mlVertex<T> & mlVertex<T>::operator -=(mlVertex<T> &v)
{
	x -= v.x; y -= v.y; z -= v.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex<T> & mlVertex<T>::operator -=(mlVector4D<T> &v)
{
	x -= v[0]; y -= v[1]; z -= v[2];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector4D<T> mlVertex<T>:: operator - (mlVertex<T> &v)
{
	return mlVector4D<T>(x - v.x, y - v.y, z - v.z, 0);
}

template<class T>
MLFUNC_TYPE mlVertex<T> mlVertex<T>:: operator - (mlVector4D<T> &v)
{
	mlVertex<T> temp(*this);
	return (temp -= v);
}

template<class T>
MLFUNC_TYPE mlVertex<T> mlVertex<T>:: operator * (T value)
{
	mlVertex<T> temp(*this);
	return (temp *= value);
}

template<class T>
MLFUNC_TYPE mlVertex<T> & mlVertex<T>:: operator *= (T value)
{
	x *= value; y *= value; z *= value;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex<T> mlVertex<T>:: operator / (T value)
{
	mlVertex<T> temp(*this);
	return (temp /= value);
}

template<class T>
MLFUNC_TYPE mlVertex<T> & mlVertex<T>:: operator /= (T value)
{
	x /= value; y /= value; z /= value;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool mlVertex<T>::operator == (const mlVertex<T> &v)
{
	if (x == v.x&&y == v.y&&z == v.z&&w == v.w)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool mlVertex<T>::operator != (const mlVertex<T> &v)
{
	return !((*this) == v);
}

//template<class T>
//MLFUNC_TYPE mlVertex<T> & mlVertex<T>::Transform(mlMatrix<T, 4, 4> *m_t)
//{
//	T x = this->x, y = this->y, z = this->z, w = this->w;
//
//	this->x = (*m_t)(0, 0)*x + (*m_t)(0, 1)*y + (*m_t)(0, 2)*z + (*m_t)(0, 3)*w;
//	this->y = (*m_t)(1, 0)*x + (*m_t)(1, 1)*y + (*m_t)(1, 2)*z + (*m_t)(1, 3)*w;
//	this->z = (*m_t)(2, 0)*x + (*m_t)(2, 1)*y + (*m_t)(2, 2)*z + (*m_t)(2, 3)*w;
//	this->w = (*m_t)(3, 0)*x + (*m_t)(3, 1)*y + (*m_t)(3, 2)*z + (*m_t)(3, 3)*w;
//
//	return (*this);
//}

///////////////////////////////////////////////////////////////////////

template<class T>
class mlVertex2D
{
public:
	T x, y;

	MLFUNC_TYPE bool operator != (const mlVertex2D<T> &v);
	MLFUNC_TYPE bool operator == (const mlVertex2D<T> &v);
	MLFUNC_TYPE mlVertex2D<T> operator / (T value);
	MLFUNC_TYPE mlVertex2D<T> & operator /= (T value);
	MLFUNC_TYPE mlVertex2D<T> operator * (T value);
	MLFUNC_TYPE mlVertex2D<T> & operator *= (T value);
	MLFUNC_TYPE mlVertex2D<T> operator + (mlVector2D<T> &v);
	MLFUNC_TYPE mlVertex2D<T> operator + (mlVertex2D<T> &v);
	MLFUNC_TYPE mlVertex2D<T> & operator += (mlVertex2D<T> &v);
	MLFUNC_TYPE mlVertex2D<T> operator - (mlVector2D<T> &v);
	MLFUNC_TYPE mlVertex2D<T> operator - (mlVertex2D<T> &v);
	MLFUNC_TYPE mlVertex2D<T> & operator -= (mlVertex2D<T> &v);
	MLFUNC_TYPE mlVertex2D<T> & operator = (const mlVertex2D<T> &v);
	MLFUNC_TYPE mlVertex2D<T> & operator = (const T &v);
	//MLFUNC_TYPE mlVertex2D<T> & Transform(mlMatrix<T, 2, 2> *m_t);
	MLFUNC_TYPE T GetNorm();
	MLFUNC_TYPE mlVertex2D();
	MLFUNC_TYPE mlVertex2D(T val);
	MLFUNC_TYPE mlVertex2D(T x, T y);
	MLFUNC_TYPE mlVertex2D(const mlVertex2D<T> &v);
};

typedef mlVertex2D<int> mlVertex2i;
typedef mlVertex2D<long> mlVertex2l;
typedef mlVertex2D<REAL> mlVertex2f;
typedef mlVertex2D<double> mlVertex2d;

/////////////////implementation of vertex2D//////////////////////
template<class T>
MLFUNC_TYPE mlVertex2D<T>::mlVertex2D()
{
	x = y = 0;
}

template<class T>
inline mlVertex2D<T>::mlVertex2D(T val)
{
	x = y = val;
}

template<class T>
MLFUNC_TYPE mlVertex2D<T>::mlVertex2D(T x, T y)
{
	this->x = x; this->y = y;
}

template<class T>
MLFUNC_TYPE mlVertex2D<T>::mlVertex2D(const mlVertex2D &v)
{
	x = v.x; y = v.y;;
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> & mlVertex2D<T>:: operator = (const mlVertex2D<T> &v)
{
	x = v.x; y = v.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> & mlVertex2D<T>:: operator = (const T &v)
{
	x = y = v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> & mlVertex2D<T>::operator -=(mlVertex2D<T> &v)
{
	x -= v.x; y -= v.y;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> mlVertex2D<T>:: operator - (mlVector2D<T> &v)
{
	mlVertex2D<T> temp(*this);
	return (temp -= v);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> mlVertex2D<T>:: operator - (mlVertex2D<T> &v)
{
	mlVertex2D<T> temp(*this);
	return (temp -= v);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> & mlVertex2D<T>::operator +=(mlVertex2D<T> &v)
{
	x += v.x; y += v.y;;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> mlVertex2D<T>:: operator + (mlVector2D<T> &v)
{
	mlVertex2D<T> temp(*this);
	return (temp += v);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> mlVertex2D<T>:: operator + (mlVertex2D<T> &v)
{
	mlVertex2D<T> temp(*this);
	return (temp += v);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> mlVertex2D<T>:: operator * (T value)
{
	mlVertex2D<T> temp(*this);
	return (temp *= value);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> & mlVertex2D<T>:: operator *= (T value)
{
	x *= value; y *= value;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> mlVertex2D<T>:: operator / (T value)
{
	mlVertex2D<T> temp(*this);
	return (temp /= value);
}

template<class T>
MLFUNC_TYPE mlVertex2D<T> & mlVertex2D<T>:: operator /= (T value)
{
	x /= value; y /= value;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool mlVertex2D<T>::operator == (const mlVertex2D<T> &v)
{
	if (x == v.x&&y == v.y)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool mlVertex2D<T>::operator != (const mlVertex2D<T> &v)
{
	return !((*this) == v);
}

//template<class T>
//MLFUNC_TYPE mlVertex2D<T> & mlVertex2D<T>::Transform(mlMatrix<T, 2, 2> *m_t)
//{
//	T x = this->x, y = this->y;
//
//	this->x = (*m_t)(0, 0)*x + (*m_t)(0, 1)*y;
//	this->y = (*m_t)(1, 0)*x + (*m_t)(1, 1)*y;
//
//	return (*this);
//}

template<class T>
MLFUNC_TYPE T mlVertex2D<T>::GetNorm()
{
	return T(sqrt(double(x)*double(x) + double(y)*double(y)));
}

////////////////////////////////////////////////////////////////////////

template<class T>
class mlVertex3D
{
public:
	T x, y, z;

	MLFUNC_TYPE bool operator != (const mlVertex3D<T> &v);
	MLFUNC_TYPE bool operator == (const mlVertex3D<T> &v);
	MLFUNC_TYPE mlVertex3D<T> operator / (T value);
	MLFUNC_TYPE mlVertex3D<T> & operator /= (T value);
	MLFUNC_TYPE mlVertex3D<T> operator * (T value);
	MLFUNC_TYPE mlVertex3D<T> & operator *= (T value);
	MLFUNC_TYPE mlVertex3D<T> operator + (mlVector3D<T> &v);
	MLFUNC_TYPE mlVertex3D<T> operator + (mlVertex3D<T> &v);
	MLFUNC_TYPE mlVertex3D<T> & operator += (mlVertex3D<T> &v);
	MLFUNC_TYPE mlVertex3D<T> operator - (mlVector3D<T> &v);
	MLFUNC_TYPE mlVertex3D<T> operator - (mlVertex3D<T> &v);
	MLFUNC_TYPE mlVertex3D<T> & operator -= (mlVertex3D<T> &v);
	MLFUNC_TYPE mlVertex3D<T> & operator = (const mlVertex3D<T> &v);
	MLFUNC_TYPE mlVertex3D<T> & operator = (const T &v);
	//	MLFUNC_TYPE mlVertex3D<T> & Transform(mlMatrix<T, 3, 3> *m_t);
	MLFUNC_TYPE T GetNorm();
	MLFUNC_TYPE mlVertex3D();
	MLFUNC_TYPE mlVertex3D(T x, T y, T z);
	MLFUNC_TYPE mlVertex3D(const mlVertex3D<T> &v);
	MLFUNC_TYPE mlVertex3D<T> CrossProduct(mlVertex3D<T> &v);

	MLFUNC_TYPE T Dot(mlVertex3D<T>& v);


};

typedef mlVertex3D<int> mlVertex3i;
typedef mlVertex3D<long> mlVertex3l;
typedef mlVertex3D<REAL> mlVertex3f;
typedef mlVertex3D<double> mlVertex3d;

/////////////////implementation of vertex3D//////////////////////
template<class T>
MLFUNC_TYPE mlVertex3D<T>::mlVertex3D()
{
	x = y = z = 0;
}

template<class T>
MLFUNC_TYPE mlVertex3D<T>::mlVertex3D(T x, T y, T z)
{
	this->x = x; this->y = y; this->z = z;
}

template<class T>
MLFUNC_TYPE mlVertex3D<T>::mlVertex3D(const mlVertex3D &v)
{
	x = v.x; y = v.y; z = v.z;
}

template<class T>
inline MLFUNC_TYPE mlVertex3D<T> mlVertex3D<T>::CrossProduct(mlVertex3D<T>& v)
{
	T x = this->y * v.z - this->z * v.y;
	T y = this->z * v.x - this->x * v.z;
	T z = this->x * v.y - this->y * v.x;
	return  mlVertex3D<T>(x, y, z);
}

template<class T>
inline MLFUNC_TYPE T mlVertex3D<T>::Dot(mlVertex3D<T>& v)
{
	return this->x * v.x + this->y * v.y + this->z * v.z;
}


template<class T>
MLFUNC_TYPE mlVertex3D<T> & mlVertex3D<T>:: operator = (const mlVertex3D<T> &v)
{
	x = v.x; y = v.y; z = v.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> & mlVertex3D<T>:: operator = (const T &v)
{
	x = y = z = v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> & mlVertex3D<T>::operator -=(mlVertex3D<T> &v)
{
	x -= v.x; y -= v.y; z -= v.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> mlVertex3D<T>:: operator - (mlVector3D<T> &v)
{
	mlVertex3D<T> temp(*this);
	return (temp -= v);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> mlVertex3D<T>:: operator - (mlVertex3D<T> &v)
{
	mlVertex3D<T> temp(*this);
	return (temp -= v);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> & mlVertex3D<T>::operator +=(mlVertex3D<T> &v)
{
	x += v.x; y += v.y; z += v.z;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> mlVertex3D<T>:: operator + (mlVector3D<T> &v)
{
	mlVertex3D<T> temp(*this);
	return (temp += v);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> mlVertex3D<T>:: operator + (mlVertex3D<T> &v)
{
	mlVertex3D<T> temp(*this);
	return (temp += v);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> mlVertex3D<T>:: operator * (T value)
{
	mlVertex3D<T> temp(*this);
	return (temp *= value);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> & mlVertex3D<T>:: operator *= (T value)
{
	x *= value; y *= value; z *= value;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> mlVertex3D<T>:: operator / (T value)
{
	mlVertex3D<T> temp(*this);
	return (temp /= value);
}

template<class T>
MLFUNC_TYPE mlVertex3D<T> & mlVertex3D<T>:: operator /= (T value)
{
	x /= value; y /= value; z /= value;
	return (*this);
}

template<class T>
MLFUNC_TYPE bool mlVertex3D<T>::operator == (const mlVertex3D<T> &v)
{
	if (x == v.x&&y == v.y&&z == v.z)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool mlVertex3D<T>::operator != (const mlVertex3D<T> &v)
{
	return !((*this) == v);
}

//template<class T>
//MLFUNC_TYPE mlVertex3D<T> & mlVertex3D<T>::Transform(mlMatrix<T, 3, 3> *m_t)
//{
//	T x = this->x, y = this->y, z = this->z;
//
//	this->x = (*m_t)(0, 0)*x + (*m_t)(0, 1)*y + (*m_t)(0, 2)*z;
//	this->y = (*m_t)(1, 0)*x + (*m_t)(1, 1)*y + (*m_t)(1, 2)*z;
//	this->z = (*m_t)(2, 0)*x + (*m_t)(2, 1)*y + (*m_t)(2, 2)*z;
//
//	return (*this);
//}

template<class T>
MLFUNC_TYPE T mlVertex3D<T>::GetNorm()
{
	return T(sqrt(double(x)*double(x) + double(y)*double(y) + double(z)*double(z)));
}

///////////////global functions on computing vertex distance///////////////

template<class T>
MLFUNC_TYPE T GVLVertexDist(const mlVertex<T> &v1, const mlVertex<T> &v2)
{
	return (T)sqrt(double((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z)));
}

template<class T>
MLFUNC_TYPE T GVLVertexSqrtDist(const mlVertex<T> &v1, const mlVertex<T> &v2)
{
	return (T)sqrt(double((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z)));
}

template<class T>
MLFUNC_TYPE T GVLVertexDist(const mlVertex2D<T> &v1, const mlVertex2D<T> &v2)
{
	return (T)sqrt(double((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y)));
}

template<class T>
MLFUNC_TYPE T GVLVertexSqrtDist(const mlVertex2D<T> &v1, const mlVertex2D<T> &v2)
{
	return (T)sqrt(double((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y)));
}

//////////////////////////////////////////////////////////////////////////

template<class T>
MLFUNC_TYPE void GVLVertexRotate(mlVertex2D<T> &ref_vertex, mlVertex2D<T> &vertex_in_out, T angle) //angle in arc
{
	mlVertex2D<T> v(vertex_in_out.x - ref_vertex.x, vertex_in_out.y - ref_vertex.y);

	double c = cos(double(angle)), s = sin(double(angle));

	vertex_in_out.x = ref_vertex.x + T(c*v.x - s*v.y);
	vertex_in_out.y = ref_vertex.y + T(s*v.x + c*v.y);
}



#endif //_GVLVERTEX_