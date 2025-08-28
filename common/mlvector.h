#pragma once

#ifndef _mlVector_
#define _mlVector_

#include "math.h"
#include "mlCoreWin.h"
#include "memory.h"


#ifndef NULL
#define NULL 0
#endif


template<class T, long size>
class mlSVector
{
public:
	MLFUNC_TYPE  void Swap(long i, long j);
	MLFUNC_TYPE  void Reverse();

	MLFUNC_TYPE  mlSVector<T, size>& Normalize();

	MLFUNC_TYPE  T GetNorm(double p = 2.0);//the p-norm of the vector
	MLFUNC_TYPE  T GetEuclidianNorm();
	MLFUNC_TYPE  T GetManhattanNorm();

	MLFUNC_TYPE  T GetSum();
	MLFUNC_TYPE  T GetSquaredSum();

	MLFUNC_TYPE  void GenRand(T low = 0, T up = 1);
	MLFUNC_TYPE  void GenGaussian(T center_v, REAL u = 0.0f, REAL sigma = 1.0f);

	MLFUNC_TYPE  void Zero();

	MLFUNC_TYPE  T GetMaxValue();
	MLFUNC_TYPE  T GetMinValue();

	MLFUNC_TYPE  void SetConstValue(const T& v);

	MLFUNC_TYPE long GetCount();

	MLFUNC_TYPE mlSVector<T, size>& Add(const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>& Sub(const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>& Mul(const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>& Div(const mlSVector<T, size>& v);

	MLFUNC_TYPE bool operator == (const mlSVector<T, size>& v);
	MLFUNC_TYPE bool operator != (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>   operator /  (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>& operator /= (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>   operator /  (const T& v);
	MLFUNC_TYPE mlSVector<T, size>& operator /= (const T& v);
	MLFUNC_TYPE mlSVector<T, size>   operator *  (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>& operator *= (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>   operator *  (const T& v);
	MLFUNC_TYPE mlSVector<T, size>& operator *= (const T& v);
	MLFUNC_TYPE mlSVector<T, size>   operator -  (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>& operator -= (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>   operator -  (const T& v);
	MLFUNC_TYPE mlSVector<T, size>& operator -= (const T& v);
	MLFUNC_TYPE mlSVector<T, size>   operator +  (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>& operator += (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>   operator +  (const T& v);
	MLFUNC_TYPE mlSVector<T, size>& operator += (const T& v);

	MLFUNC_TYPE T& operator () (long index);
	MLFUNC_TYPE const T& operator () (long index)const;
	MLFUNC_TYPE T& operator [] (long index);
	MLFUNC_TYPE const T& operator [] (long index)const;

	MLFUNC_TYPE mlSVector<T, size>& operator = (const mlSVector<T, size>& v);
	MLFUNC_TYPE mlSVector<T, size>& operator = (T& v);

	MLFUNC_TYPE void* GetData();
	MLFUNC_TYPE T* GetTypedData();

	MLFUNC_TYPE mlSVector();
	MLFUNC_TYPE mlSVector(T* data);
	MLFUNC_TYPE mlSVector(const T& value);
	MLFUNC_TYPE mlSVector(const mlSVector<T, size>& v);
protected:
	T data[size];
};

////////////////////implementation of vector////////////////////////
template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>::mlSVector()
{
	memset(data, 0, sizeof(T) * size);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>::mlSVector(T* data)
{
	memcpy(this->data, data, sizeof(T) * size);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>::mlSVector(const T& value)
{
	for (long i = 0; i < size; i++)
		data[i] = value;
}


template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>::mlSVector(const mlSVector<T, size>& v)
{
	memcpy(data, v.data, sizeof(T) * size);
}

template<class T, long size>
MLFUNC_TYPE long mlSVector<T, size>::GetCount()
{
	return size;
}

template<class T, long size>
MLFUNC_TYPE T mlSVector<T, size>::GetMaxValue()
{
	T max_value = data[0];
	for (long i = 1; i < size; i++)
	{
		if (max_value < data[i])
			max_value = data[i];
	}

	return max_value;
}

template<class T, long size>
MLFUNC_TYPE T mlSVector<T, size>::GetMinValue()
{
	T min_value = data[0];
	for (long i = 1; i < size; i++)
	{
		if (min_value > data[i])
			min_value = data[i];
	}

	return min_value;
}

template<class T, long size>
MLFUNC_TYPE void mlSVector<T, size>::Swap(long i, long j)
{
	T temp = data[i];
	data[i] = data[j];
	data[j] = temp;
}

template<class T, long size>
MLFUNC_TYPE void mlSVector<T, size>::Reverse()
{
	long i = 0, j = size - 1;
	while (i < j)
	{
		Swap(i, j);
		i++; j--;
	}
}

template<class T, long size>
MLFUNC_TYPE void mlSVector<T, size>::Zero()
{
	memset(data, 0, sizeof(T) * size);
}

template<class T, long size>
MLFUNC_TYPE void mlSVector<T, size>::SetConstValue(const T& v)
{
	for (long i = 0; i < size; i++)
		data[i] = v;
}

template<class T, long size>
MLFUNC_TYPE void* mlSVector<T, size>::GetData()
{
	return (void*)data;
}

template<class T, long size>
MLFUNC_TYPE T* mlSVector<T, size>::GetTypedData()
{
	return data;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>::Normalize()
{
	T Norm = GetSum();
	if (Norm != 0)
	{
		for (long i = 0; i < size; i++)
			data[i] /= Norm;
	}
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE T mlSVector<T, size>::GetNorm(double p)
{
	double total = 0;
	for (long i = 0; i < size; i++)
		total += pow(fabs(double(data[i])), p);
	return (T)pow(total, 1.0 / p);
}

template<class T, long size>
MLFUNC_TYPE T mlSVector<T, size>::GetEuclidianNorm()
{
	double total = 0;
	for (long i = 0; i < size; i++)
	{
		total += double(data[i] * data[i]);
	}
	return T(sqrt(total));
}

template<class T, long size>
MLFUNC_TYPE T mlSVector<T, size>::GetManhattanNorm()
{
	T total = 0;
	for (long i = 0; i < size; i++)
	{
		total += T(fabs(double(data[i])));
	}
	return total;
}

template<class T, long size>
MLFUNC_TYPE T mlSVector<T, size>::GetSum()
{
	T total = 0;
	for (long i = 0; i < size; i++)
		total += data[i];
	return total;
}

template<class T, long size>
MLFUNC_TYPE T mlSVector<T, size>::GetSquaredSum()
{
	T total = 0;
	for (long i = 0; i < size; i++)
		total += data[i] * data[i];
	return total;
}

template<class T, long size>
MLFUNC_TYPE void mlSVector<T, size>::GenRand(T low/* =0 */, T up/* =1 */)
{
	for (long i = 0; i < size; i++)
		data[i] = 0;//GVL_Rand<T>(low, up);
}

template<class T, long size>
MLFUNC_TYPE void mlSVector<T, size>::GenGaussian(T center_v, REAL u, REAL sigma)
{
	long shift = size / 2;
	for (long i = 0; i < size; i++)
	{
		REAL x = REAL(i - shift);
		T v = T(REAL(center_v) * exp(-(x - u) * (x - u) / (2 * sigma * sigma)));
		data[i] = v;
	}
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>::Add(const mlSVector<T, size>& v)
{
	return (*this) += v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>::Sub(const mlSVector<T, size>& v)
{
	return (*this) -= v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>::Mul(const mlSVector<T, size>& v)
{
	return (*this) *= v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>::Div(const mlSVector<T, size>& v)
{
	return (*this) /= v;
}

template<class T, long size>
MLFUNC_TYPE bool mlSVector<T, size>:: operator ==  (const mlSVector<T, size>& v)
{
	return memcmp(data, v.data, sizeof(T) * size) == 0;
}

template<class T, long size>
MLFUNC_TYPE bool mlSVector<T, size>:: operator !=  (const mlSVector<T, size>& v)
{
	return !((*this) == v);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>  mlSVector<T, size>:: operator /  (const mlSVector<T, size>& v)
{
	mlSVector<T, size> temp(*this);
	return temp /= v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>:: operator /= (const mlSVector<T, size>& v)
{
	for (long i = 0; i < size; i++)
		data[i] /= v.data[i];
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>  mlSVector<T, size>:: operator /  (const T& v)
{
	mlSVector<T, size> temp(*this);
	return temp /= v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>:: operator /= (const T& v)
{
	for (long i = 0; i < size; i++)
		data[i] /= v;
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>  mlSVector<T, size>:: operator *  (const mlSVector<T, size>& v)
{
	mlSVector<T, size> temp(*this);
	return temp *= v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>:: operator *= (const mlSVector<T, size>& v)
{
	for (long i = 0; i < size; i++)
		data[i] *= v.data[i];
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>  mlSVector<T, size>:: operator *  (const T& v)
{
	mlSVector<T, size> temp(*this);
	return temp *= v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>:: operator *= (const T& v)
{
	for (long i = 0; i < size; i++)
		data[i] *= v;
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>  mlSVector<T, size>:: operator -  (const mlSVector<T, size>& v)
{
	mlSVector<T, size> temp(*this);
	return temp -= v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>:: operator -= (const mlSVector<T, size>& v)
{
	for (long i = 0; i < size; i++)
		data[i] -= v.data[i];
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>  mlSVector<T, size>:: operator -  (const T& v)
{
	mlSVector<T, size> temp(*this);
	return temp -= v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>:: operator -= (const T& v)
{
	for (long i = 0; i < size; i++)
		data[i] -= v;
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>  mlSVector<T, size>:: operator +  (const mlSVector<T, size>& v)
{
	mlSVector<T, size> temp(*this);
	return temp += v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>:: operator += (const mlSVector<T, size>& v)
{
	for (long i = 0; i < size; i++)
		data[i] += v.data[i];
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>  mlSVector<T, size>:: operator +  (const T& v)
{
	mlSVector<T, size> temp(*this);
	return temp += v;
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>:: operator += (const T& v)
{
	for (long i = 0; i < size; i++)
		data[i] += v;
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE T& mlSVector<T, size>:: operator () (long index)
{
	return data[index];
}

template<class T, long size>
MLFUNC_TYPE const T& mlSVector<T, size>:: operator () (long index)const
{
	return data[index];
}

template<class T, long size>
MLFUNC_TYPE T& mlSVector<T, size>:: operator [] (long index)
{
	return data[index];
}

template<class T, long size>
MLFUNC_TYPE const T& mlSVector<T, size>:: operator [] (long index)const
{
	return data[index];
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>:: operator = (const mlSVector<T, size>& v)
{
	memcpy(data, v.data, sizeof(T) * size);
	return (*this);
}

template<class T, long size>
MLFUNC_TYPE mlSVector<T, size>& mlSVector<T, size>::operator = (T& v)
{
	for (long i = 0; i < size; i++)
		data[i] = v;
	return (*this);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
class mlVector2D : public mlSVector<T, 2>
{
public:
	MLFUNC_TYPE T GetNorm();
	MLFUNC_TYPE mlVector2D<T>& Normalize();
	MLFUNC_TYPE T GetSum();
	MLFUNC_TYPE T GetSquaredSum();

	MLFUNC_TYPE T* GetTypedData();

	MLFUNC_TYPE T& operator () (long index);
	MLFUNC_TYPE const T& operator () (long index)const;
	MLFUNC_TYPE T& operator [] (long index);
	MLFUNC_TYPE const T& operator [] (long index)const;

	MLFUNC_TYPE bool operator == (const mlVector2D<T>& v);
	MLFUNC_TYPE bool operator != (const mlVector2D<T>& v);
	MLFUNC_TYPE mlVector2D<T>   operator /  (const mlVector2D<T>& v);
	MLFUNC_TYPE mlVector2D<T>& operator /= (const mlVector2D<T>& v);
	MLFUNC_TYPE mlVector2D<T>   operator /  (const T& v);
	MLFUNC_TYPE mlVector2D<T>& operator /= (const T& v);
	MLFUNC_TYPE mlVector2D<T>   operator *  (const mlVector2D<T>& v);
	MLFUNC_TYPE mlVector2D<T>& operator *= (const mlVector2D<T>& v);
	MLFUNC_TYPE mlVector2D<T>   operator *  (const T& v);
	MLFUNC_TYPE mlVector2D<T>& operator *= (const T& v);
	MLFUNC_TYPE mlVector2D<T>   operator -  (const mlVector2D<T>& v);
	MLFUNC_TYPE mlVector2D<T>& operator -= (const mlVector2D<T>& v);
	MLFUNC_TYPE mlVector2D<T>   operator -  (const T& v);
	MLFUNC_TYPE mlVector2D<T>& operator -= (const T& v);
	MLFUNC_TYPE mlVector2D<T>   operator +  (const mlVector2D<T>& v);
	MLFUNC_TYPE mlVector2D<T>& operator += (const mlVector2D<T>& v);
	MLFUNC_TYPE mlVector2D<T>   operator +  (const T& v);
	MLFUNC_TYPE mlVector2D<T>& operator += (const T& v);

	MLFUNC_TYPE mlVector2D<T>& operator = (const mlVector2D<T>& v);

	MLFUNC_TYPE mlVector2D();
	MLFUNC_TYPE mlVector2D(T x, T y);
	MLFUNC_TYPE mlVector2D(T v);
	MLFUNC_TYPE mlVector2D(const mlVector2D<T>& v);

	MLFUNC_TYPE T dot(mlVector2D<T>& v);
	MLFUNC_TYPE T CrossProduct(mlVector2D<T>& u);

};

typedef mlVector2D<int> mlVector2i;
typedef mlVector2D<long> mlVector2l;
typedef mlVector2D<REAL> mlVector2f;
typedef mlVector2D<double> mlVector2d;

//////////////////////////////implementation of mlVector2D//////////////////////////////////////////
template<class T>
MLFUNC_TYPE mlVector2D<T>::mlVector2D()
{
	memset(this->data, 0, sizeof(T) * 2);
}

template<class T>
MLFUNC_TYPE mlVector2D<T>::mlVector2D(T x, T y)
{
	this->data[0] = x; this->data[1] = y;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>::mlVector2D(T v)
{
	this->data[0] = v; this->data[1] = v;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>::mlVector2D(const mlVector2D<T>& v)
{
	memcpy(this->data, v.data, sizeof(T) * 2);
}

template<class T>
inline MLFUNC_TYPE T mlVector2D<T>::dot(mlVector2D<T>& v)
{
	return T(this->data[0] * v.data[0] +
		this->data[1] * v.data[1]);
}

template<class T>
inline MLFUNC_TYPE  T mlVector2D<T>::CrossProduct(mlVector2D<T>& u)
{
	T  x = this->data[0] * u[1] - this->data[1] * u[0];
	return x;
}

template<class T>
MLFUNC_TYPE T& mlVector2D<T>::operator () (long index)
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE const T& mlVector2D<T>::operator () (long index)const
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE T& mlVector2D<T>::operator [] (long index)
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE const T& mlVector2D<T>::operator [] (long index)const
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE T mlVector2D<T>::GetNorm()
{
	return (T)sqrt(double(this->data[0] * this->data[0] + this->data[1] * this->data[1]));
}

template<class T>
MLFUNC_TYPE T mlVector2D<T>::GetSquaredSum()
{
	T total = 0;
	for (long i = 0; i < 2; i++)
		total += this->data[i];
	return total;
}

template<class T>
MLFUNC_TYPE T mlVector2D<T>::GetSum()
{
	T total = 0;
	for (long i = 0; i < 2; i++)
		total += this->data[i] * this->data[i];
	return total;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::Normalize()
{
	T Norm = GetNorm();
	if (Norm != 0)
	{
		for (long i = 0; i < 2; i++)
			this->data[i] /= Norm;
	}
	return (*this);
}

template<class T>
MLFUNC_TYPE T* mlVector2D<T>::GetTypedData()
{
	return this->data;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::operator = (const mlVector2D<T>& v)
{
	memcpy(this->data, v.data, sizeof(T) * 2);
	return (*this);
}

template<class T>
MLFUNC_TYPE bool mlVector2D<T>::operator == (const mlVector2D<T>& v)
{
	if (this->data[0] == v.data[0] && this->data[1] == v.data[1])
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool mlVector2D<T>::operator != (const mlVector2D<T>& v)
{
	return !((*this) == v);
}

template<class T>
MLFUNC_TYPE mlVector2D<T> mlVector2D<T>::operator /  (const mlVector2D<T>& v)
{
	mlVector2D<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::operator /= (const mlVector2D<T>& v)
{
	this->data[0] /= v.data[0];
	this->data[1] /= v.data[1];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector2D<T> mlVector2D<T>::operator /  (const T& v)
{
	mlVector2D<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::operator /= (const T& v)
{
	this->data[0] /= v;
	this->data[1] /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector2D<T> mlVector2D<T>::operator *  (const mlVector2D<T>& v)
{
	mlVector2D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::operator *= (const mlVector2D<T>& v)
{
	this->data[0] *= v.data[0];
	this->data[1] *= v.data[1];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector2D<T> mlVector2D<T>::operator *  (const T& v)
{
	mlVector2D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::operator *= (const T& v)
{
	this->data[0] *= v;
	this->data[1] *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector2D<T> mlVector2D<T>::operator -  (const mlVector2D<T>& v)
{
	mlVector2D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::operator -= (const mlVector2D<T>& v)
{
	this->data[0] -= v.data[0];
	this->data[1] -= v.data[1];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector2D<T> mlVector2D<T>::operator -  (const T& v)
{
	mlVector2D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::operator -= (const T& v)
{
	this->data[0] -= v;
	this->data[1] -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector2D<T> mlVector2D<T>::operator + (const mlVector2D<T>& v)
{
	mlVector2D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::operator += (const mlVector2D<T>& v)
{
	this->data[0] += v.data[0];
	this->data[1] += v.data[1];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector2D<T> mlVector2D<T>::operator +  (const T& v)
{
	mlVector2D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE mlVector2D<T>& mlVector2D<T>::operator += (const T& v)
{
	this->data[0] += v;
	this->data[1] += v;
	return (*this);
}

///////////////////////////////////////////////////////////////////////////
template<class T>
class mlVector3D : public mlSVector<T, 3>
{
public:
	MLFUNC_TYPE 	T GetNorm();
	MLFUNC_TYPE mlVector3D& Normalize();
	MLFUNC_TYPE 	T GetSum();
	MLFUNC_TYPE T GetSquaredSum();

	MLFUNC_TYPE 	T* GetTypedData();

	MLFUNC_TYPE T& operator () (long index);
	MLFUNC_TYPE 	const T& operator () (long index)const;
	MLFUNC_TYPE 	T& operator [] (long index);
	MLFUNC_TYPE const T& operator [] (long index)const;

	MLFUNC_TYPE bool operator == (const mlVector3D<T>& v);
	MLFUNC_TYPE bool operator != (const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T>   operator /  (const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T>& operator /= (const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T>   operator /  (const T& v);
	MLFUNC_TYPE mlVector3D<T>& operator /= (const T& v);
	MLFUNC_TYPE mlVector3D<T>   operator *  (const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T>& operator *= (const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T>   operator *  (const T& v);
	MLFUNC_TYPE mlVector3D<T>& operator *= (const T& v);
	MLFUNC_TYPE mlVector3D<T>   operator -  (const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T>& operator -= (const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T>   operator -  (const T& v);
	MLFUNC_TYPE mlVector3D<T>& operator -= (const T& v);
	MLFUNC_TYPE mlVector3D<T>   operator +  (const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T>& operator += (const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T>   operator +  (const T& v);
	MLFUNC_TYPE mlVector3D<T>& operator += (const T& v);

	MLFUNC_TYPE mlVector3D<T>& operator = (const mlVector3D<T>& v);

	MLFUNC_TYPE mlVector3D();
	MLFUNC_TYPE mlVector3D(T x, T y, T z);

	MLFUNC_TYPE mlVector3D(T v);
	MLFUNC_TYPE mlVector3D(const mlVector3D<T>& v);
	MLFUNC_TYPE mlVector3D<T> CrossProduct(mlVector3D<T>& v);
	MLFUNC_TYPE T dot(mlVector3D<T>& v);

};

typedef mlVector3D<int> mlVector3i;
typedef mlVector3D<long> mlVector3l;
typedef mlVector3D<REAL> mlVector3f;
typedef mlVector3D<double> mlVector3d;

//////////////////////////////implementation of mlVector3//////////////////////////////////////////
template<class T>
MLFUNC_TYPE mlVector3D<T>::mlVector3D()
{
	memset(this->data, 0, sizeof(T) * 3);
}

template<class T>
MLFUNC_TYPE mlVector3D<T>::mlVector3D(T x, T y, T z)
{
	this->data[0] = x; this->data[1] = y; this->data[2] = z;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>::mlVector3D(T v)
{
	this->data[0] = v; this->data[1] = v; this->data[2] = v;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>::mlVector3D(const mlVector3D<T>& v)
{
	memcpy(this->data, v.data, sizeof(T) * 3);
}

template<class T>
inline MLFUNC_TYPE mlVector3D<T> mlVector3D<T>::CrossProduct(mlVector3D<T>& v)
{
	T x = this->data[1] * v.data[2] - this->data[2] * v.data[1];
	T y = this->data[2] * v.data[0] - this->data[0] * v.data[2];
	T z = this->data[0] * v.data[1] - this->data[1] * v.data[0];
	return  mlVector3D<T>(x, y, z);
}

template<class T>
inline MLFUNC_TYPE T mlVector3D<T>::dot(mlVector3D<T>& v)
{
	return T(this->data[0] * v.data[0] +
		this->data[1] * v.data[1] +
		this->data[2] * v.data[2]);
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::operator = (const mlVector3D<T>& v)
{
	memcpy(this->data, v.data, sizeof(T) * 3);
	return (*this);
}

template<class T>
MLFUNC_TYPE T& mlVector3D<T>::operator () (long index)
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE const T& mlVector3D<T>::operator () (long index)const
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE T& mlVector3D<T>::operator [] (long index)
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE const T& mlVector3D<T>::operator [] (long index)const
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE T mlVector3D<T>::GetNorm()
{
	return (T)sqrt(double(this->data[0] * this->data[0] + this->data[1] * this->data[1] + this->data[2] * this->data[2]));
}

template<class T>
MLFUNC_TYPE T mlVector3D<T>::GetSum()
{
	T total = 0;
	for (long i = 0; i < 3; i++)
		total += this->data[i];
	return total;
}

template<class T>
MLFUNC_TYPE T mlVector3D<T>::GetSquaredSum()
{
	T total = 0;
	for (long i = 0; i < 3; i++)
		total += this->data[i] * this->data[i];
	return total;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::Normalize()
{
	T Norm = GetNorm();
	if (Norm != 0)
	{
		for (long i = 0; i < 3; i++)
			this->data[i] /= Norm;
	}
	return (*this);
}

template<class T>
MLFUNC_TYPE T* mlVector3D<T>::GetTypedData()
{
	return this->data;
}

template<class T>
MLFUNC_TYPE bool mlVector3D<T>::operator == (const mlVector3D<T>& v)
{
	if (this->data[0] == v.data[0] && this->data[1] == v.data[1] && this->data[2] == v.data[2])
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool mlVector3D<T>::operator != (const mlVector3D<T>& v)
{
	return !((*this) == v);
}

template<class T>
MLFUNC_TYPE mlVector3D<T> mlVector3D<T>::operator /  (const mlVector3D<T>& v)
{
	mlVector3D<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::operator /= (const mlVector3D<T>& v)
{
	this->data[0] /= v.data[0];
	this->data[1] /= v.data[1];
	this->data[2] /= v.data[2];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector3D<T> mlVector3D<T>::operator /  (const T& v)
{
	mlVector3D<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::operator /= (const T& v)
{
	this->data[0] /= v;
	this->data[1] /= v;
	this->data[2] /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector3D<T> mlVector3D<T>::operator *  (const mlVector3D<T>& v)
{
	mlVector3D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::operator *= (const mlVector3D<T>& v)
{
	this->data[0] *= v.data[0];
	this->data[1] *= v.data[1];
	this->data[2] *= v.data[2];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector3D<T> mlVector3D<T>::operator *  (const T& v)
{
	mlVector3D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::operator *= (const T& v)
{
	this->data[0] *= v;
	this->data[1] *= v;
	this->data[2] *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector3D<T> mlVector3D<T>::operator -  (const mlVector3D<T>& v)
{
	mlVector3D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::operator -= (const mlVector3D<T>& v)
{
	this->data[0] -= v.data[0];
	this->data[1] -= v.data[1];
	this->data[2] -= v.data[2];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector3D<T> mlVector3D<T>::operator -  (const T& v)
{
	mlVector3D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::operator -= (const T& v)
{
	this->data[0] -= v;
	this->data[1] -= v;
	this->data[2] -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector3D<T> mlVector3D<T>::operator + (const mlVector3D<T>& v)
{
	mlVector3D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::operator += (const mlVector3D<T>& v)
{
	this->data[0] += v.data[0];
	this->data[1] += v.data[1];
	this->data[2] += v.data[2];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector3D<T> mlVector3D<T>::operator +  (const T& v)
{
	mlVector3D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE mlVector3D<T>& mlVector3D<T>::operator += (const T& v)
{
	this->data[0] += v;
	this->data[1] += v;
	this->data[2] += v;
	return (*this);
}

///////////////////more functions//////////////////////
template<class T>
MLFUNC_TYPE inline mlVector3D<T> operator+(const mlVector3D<T>& v1, const mlVector3D<T>& v2) {
	return mlVector3D<T>(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}
template<class T>
MLFUNC_TYPE inline mlVector3D<T> operator-(const mlVector3D<T>& v1, const mlVector3D<T>& v2) {
	return vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

template<class T>
MLFUNC_TYPE inline mlVector3D<T> operator*(const mlVector3D<T>& v1, const mlVector3D<T>& v2) {
	return vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

template<class T>
MLFUNC_TYPE inline mlVector3D<T> operator/(const mlVector3D<T>& v1, const mlVector3D<T>& v2) {
	return vec3(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
}

template<class T>
MLFUNC_TYPE inline mlVector3D<T> operator*(REAL t, const mlVector3D<T>& v) {
	return mlVector3D<T>(t * v[0], t * v[1], t * v[2]);
}

template<class T>
MLFUNC_TYPE inline mlVector3D<T> operator/(mlVector3D<T> v, REAL t) {
	return mlVector3D<T>(v[0] / t, v[1] / t, v[2] / t);
}

template<class T>
MLFUNC_TYPE inline mlVector3D<T> operator*(const mlVector3D<T>& v, REAL t) {
	return mlVector3D<T>(t * v[0], t * v[1], t * v[2]);
}

template<class T>
MLFUNC_TYPE inline REAL dot(const mlVector3D<T>& v1, const mlVector3D<T>& v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template<class T>
MLFUNC_TYPE inline mlVector3D<T> cross(const mlVector3D<T>& v1, const mlVector3D<T>& v2) {
	return mlVector3D<T>((v1[1] * v2[2] - v1[2] * v2[1]),
		(-(v1[0] * v2[2] - v1[2] * v2[0])),
		(v1[0] * v2[1] - v1[1] * v2[0]));
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class mlVector4D : public mlSVector<T, 4>
{
public:
	MLFUNC_TYPE T GetNorm();
	MLFUNC_TYPE mlVector4D& Normalize();
	MLFUNC_TYPE T GetSum();
	MLFUNC_TYPE 	T GetSquaredSum();

	MLFUNC_TYPE 	T* GetTypedData();

	MLFUNC_TYPE 	T& operator () (long index);
	MLFUNC_TYPE const T& operator () (long index)const;
	MLFUNC_TYPE T& operator [] (long index);
	MLFUNC_TYPE 	const T& operator [] (long index)const;

	MLFUNC_TYPE bool operator == (const mlVector4D<T>& v);
	MLFUNC_TYPE bool operator != (const mlVector4D<T>& v);
	MLFUNC_TYPE mlVector4D<T>   operator /  (const mlVector4D<T>& v);
	MLFUNC_TYPE mlVector4D<T>& operator /= (const mlVector4D<T>& v);
	MLFUNC_TYPE mlVector4D<T>   operator /  (const T& v);
	MLFUNC_TYPE mlVector4D<T>& operator /= (const T& v);
	MLFUNC_TYPE mlVector4D<T>   operator *  (const mlVector4D<T>& v);
	MLFUNC_TYPE mlVector4D<T>& operator *= (const mlVector4D<T>& v);
	MLFUNC_TYPE mlVector4D<T>   operator *  (const T& v);
	MLFUNC_TYPE mlVector4D<T>& operator *= (const T& v);
	MLFUNC_TYPE mlVector4D<T>   operator -  (const mlVector4D<T>& v);
	MLFUNC_TYPE mlVector4D<T>& operator -= (const mlVector4D<T>& v);
	MLFUNC_TYPE mlVector4D<T>   operator -  (const T& v);
	MLFUNC_TYPE mlVector4D<T>& operator -= (const T& v);
	MLFUNC_TYPE mlVector4D<T>   operator +  (const mlVector4D<T>& v);
	MLFUNC_TYPE mlVector4D<T>& operator += (const mlVector4D<T>& v);
	MLFUNC_TYPE mlVector4D<T>   operator +  (const T& v);
	MLFUNC_TYPE mlVector4D<T>& operator += (const T& v);

	MLFUNC_TYPE mlVector4D& operator = (const mlVector4D& v);

	MLFUNC_TYPE mlVector4D();
	MLFUNC_TYPE mlVector4D(T x, T y, T z, T w = T(1));
	MLFUNC_TYPE mlVector4D(T v);
	MLFUNC_TYPE mlVector4D(const mlVector4D<T>& v);

};

typedef mlVector4D<int> mlVector4i;
typedef mlVector4D<long> mlVector4l;
typedef mlVector4D<REAL> mlVector4f;
typedef mlVector4D<double> mlVector4d;

//////////////////////////////implementation of mlVector4//////////////////////////////////////////
template<class T>
MLFUNC_TYPE mlVector4D<T>::mlVector4D()
{
	memset(this->data, 0, sizeof(T) * 4);
}

template<class T>
MLFUNC_TYPE mlVector4D<T>::mlVector4D(T x, T y, T z, T w)
{
	this->data[0] = x; this->data[1] = y; this->data[2] = z; this->data[3] = w;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>::mlVector4D(T v)
{
	this->data[0] = v; this->data[1] = v; this->data[2] = v; this->data[3] = v;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>::mlVector4D(const mlVector4D<T>& v)
{
	memcpy(this->data, v.data, sizeof(T) * 4);
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::operator = (const mlVector4D<T>& v)
{
	memcpy(this->data, v.data, sizeof(T) * 4);
	return (*this);
}

template<class T>
MLFUNC_TYPE T& mlVector4D<T>::operator () (long index)
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE const T& mlVector4D<T>::operator () (long index)const
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE T& mlVector4D<T>::operator [] (long index)
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE const T& mlVector4D<T>::operator [] (long index)const
{
	return this->data[index];
}

template<class T>
MLFUNC_TYPE T mlVector4D<T>::GetNorm()
{
	return (T)sqrt(double(this->data[0] * this->data[0] + this->data[1] * this->data[1] + this->data[2] * this->data[2] + this->data[3] * this->data[3]));
}

template<class T>
MLFUNC_TYPE T mlVector4D<T>::GetSum()
{
	T total = 0;
	for (long i = 0; i < 4; i++)
		total += this->data[i];
	return total;
}

template<class T>
MLFUNC_TYPE T mlVector4D<T>::GetSquaredSum()
{
	T total = 0;
	for (long i = 0; i < 4; i++)
		total += this->data[i] * this->data[i];
	return total;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::Normalize()
{
	T Norm = GetNorm();
	if (Norm != 0)
	{
		for (long i = 0; i < 4; i++)
			this->data[i] /= Norm;
	}
	return (*this);
}

template<class T>
MLFUNC_TYPE T* mlVector4D<T>::GetTypedData()
{
	return this->data;
}

template<class T>
MLFUNC_TYPE bool mlVector4D<T>::operator == (const mlVector4D<T>& v)
{
	if (this->data[0] == v.data[0] && this->data[1] == v.data[1] && this->data[2] == v.data[2] & this->data[3] == v.data[3])
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool mlVector4D<T>::operator != (const mlVector4D<T>& v)
{
	return !((*this) == v);
}

template<class T>
MLFUNC_TYPE mlVector4D<T> mlVector4D<T>::operator /  (const mlVector4D<T>& v)
{
	mlVector4D<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::operator /= (const mlVector4D<T>& v)
{
	this->data[0] /= v.data[0];
	this->data[1] /= v.data[1];
	this->data[2] /= v.data[2];
	this->data[3] /= v.data[3];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector4D<T> mlVector4D<T>::operator /  (const T& v)
{
	mlVector4D<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::operator /= (const T& v)
{
	this->data[0] /= v;
	this->data[1] /= v;
	this->data[2] /= v;
	this->data[3] /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector4D<T> mlVector4D<T>::operator *  (const mlVector4D<T>& v)
{
	mlVector4D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::operator *= (const mlVector4D<T>& v)
{
	this->data[0] *= v.data[0];
	this->data[1] *= v.data[1];
	this->data[2] *= v.data[2];
	this->data[3] *= v.data[3];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector4D<T> mlVector4D<T>::operator *  (const T& v)
{
	mlVector4D<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::operator *= (const T& v)
{
	this->data[0] *= v;
	this->data[1] *= v;
	this->data[2] *= v;
	this->data[3] *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector4D<T> mlVector4D<T>::operator -  (const mlVector4D<T>& v)
{
	mlVector4D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::operator -= (const mlVector4D<T>& v)
{
	this->data[0] -= v.data[0];
	this->data[1] -= v.data[1];
	this->data[2] -= v.data[2];
	this->data[3] -= v.data[3];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector4D<T> mlVector4D<T>::operator -  (const T& v)
{
	mlVector4D<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::operator -= (const T& v)
{
	this->data[0] -= v;
	this->data[1] -= v;
	this->data[2] -= v;
	this->data[3] -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector4D<T> mlVector4D<T>::operator + (const mlVector4D<T>& v)
{
	mlVector4D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::operator += (const mlVector4D<T>& v)
{
	this->data[0] += v.data[0];
	this->data[1] += v.data[1];
	this->data[2] += v.data[2];
	this->data[3] += v.data[3];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector4D<T> mlVector4D<T>::operator +  (const T& v)
{
	mlVector4D<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE mlVector4D<T>& mlVector4D<T>::operator += (const T& v)
{
	this->data[0] += v;
	this->data[1] += v;
	this->data[2] += v;
	this->data[3] += v;
	return (*this);
}

//////////////////////////////////////definition of mlVector  --XPLIU-- 2006-9-22/////////////////////////////////////////

template<class T>
class mlVector
{
public:
	MLFUNC_TYPE void AttachData(T* data, long count);
	MLFUNC_TYPE void DetachData();
	MLFUNC_TYPE void Swap(long i, long j);
	MLFUNC_TYPE void Reverse();
	MLFUNC_TYPE mlVector<T>& Normalize();
	MLFUNC_TYPE T GetNorm(double p = 2.0);//the p-norm of the vector
	MLFUNC_TYPE T GetEuclidianNorm();
	MLFUNC_TYPE T GetManhattanNorm();
	MLFUNC_TYPE T GetSum();
	MLFUNC_TYPE T GetSquaredSum();
	MLFUNC_TYPE bool Resize(long count);
	//MLFUNC_TYPE void GenRand(T low = 0, T up = 1);
	MLFUNC_TYPE void GenGaussian(T center_v, REAL u = 0.0f, REAL sigma = 1.0f);
	MLFUNC_TYPE void Zero();
	MLFUNC_TYPE void SetConstValue(const T& v);
	MLFUNC_TYPE void Clear();
	MLFUNC_TYPE T GetMaxValue();
	MLFUNC_TYPE T GetMinValue();
	MLFUNC_TYPE mlVector<T>& Add(const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>& Sub(const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>& Mul(const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>& Div(const mlVector<T>& v);
	MLFUNC_TYPE bool operator == (mlVector<T>& v);
	MLFUNC_TYPE bool operator != (mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>   operator /  (const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>& operator /= (const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>   operator /  (const T& v);
	MLFUNC_TYPE mlVector<T>& operator /= (const T& v);
	MLFUNC_TYPE mlVector<T>   operator *  (const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>& operator *= (const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>   operator *  (const T& v);
	MLFUNC_TYPE mlVector<T>& operator *= (const T& v);
	MLFUNC_TYPE mlVector<T>   operator -  (const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>& operator -= (const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>   operator -  (const T& v);
	MLFUNC_TYPE mlVector<T>& operator -= (const T& v);
	MLFUNC_TYPE mlVector<T>   operator +  (const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>& operator += (const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>   operator +  (const T& v);
	MLFUNC_TYPE mlVector<T>& operator += (const T& v);
	MLFUNC_TYPE T& operator () (long index);
	MLFUNC_TYPE const T& operator () (long index)const;
	MLFUNC_TYPE T& operator [] (long index);
	MLFUNC_TYPE const T& operator [] (long index)const;
	MLFUNC_TYPE mlVector<T>& operator = (const mlVector<T>& v);
	MLFUNC_TYPE mlVector<T>& operator = (const T& v); //if the vector is empty, this operator assigns the vector to be one component scalar
	MLFUNC_TYPE void* GetData();
	MLFUNC_TYPE T* GetTypedData();
	MLFUNC_TYPE long GetCount();
	MLFUNC_TYPE mlVector();
	MLFUNC_TYPE mlVector(long count);
	MLFUNC_TYPE mlVector(T* data, long count);
	MLFUNC_TYPE mlVector(const T& value, long count);
	MLFUNC_TYPE mlVector(const mlVector<T>& v);
	MLFUNC_TYPE ~mlVector();
protected:
	T* data;
	T* data_old;
	long count;
	long count_old;
};

typedef mlVector<int> mlVectori;
typedef mlVector<long> mlVectorl;
typedef mlVector<REAL> mlVectorf;
typedef mlVector<double> mlVectord;

////////////////////////////////////implementation of mlVector  --XPLIU-- 2006-9-22//////////////////////////////////////////////////
template<class T>
MLFUNC_TYPE mlVector<T>::mlVector()
{
	data = NULL;
	count = 0;
	data_old = data;
	count_old = count;
}

template<class T>
MLFUNC_TYPE mlVector<T>::mlVector(long count)
{
	data = new T[count];
	memset(data, 0, sizeof(T) * count);
	this->count = count;
	data_old = data;
	count_old = count;
}

template<class T>
MLFUNC_TYPE mlVector<T>::mlVector(T* data, long count)
{
	this->data = new T[count];
	if (this->data)
	{
		memcpy(this->data, data, sizeof(T) * count);
		this->count = count;
	}
	else
		this->count = 0;
	data_old = data;
	count_old = count;
}

template<class T>
MLFUNC_TYPE mlVector<T>::mlVector(const T& value, long count)
{
	data = new T[count];
	if (data)
	{
		for (long i = 0; i < count; i++)
			data[i] = value;
		this->count = count;
	}
	else
		this->count = 0;
	data_old = data;
	count_old = count;
}

template<class T>
MLFUNC_TYPE mlVector<T>::mlVector(const mlVector<T>& v)
{
	data = new T[v.count];
	if (data != NULL)
	{
		memcpy(data, v.data, sizeof(T) * v.count);
		count = v.count;
	}
	else
		count = 0;
	data_old = data;
	count_old = count;
}

template<class T>
MLFUNC_TYPE mlVector<T>::~mlVector()
{
	if (data == data_old && data)
	{
		delete[]data;
	}
	else if (data_old)
	{
		delete[]data_old;
	}
}

template<class T>
MLFUNC_TYPE void mlVector<T>::AttachData(T* data, long count)
{
	data_old = this->data;
	count_old = this->count;
	this->data = data;
	this->count = count;
}

template<class T>
MLFUNC_TYPE void mlVector<T>::DetachData()
{
	data = data_old;
	count = count_old;
}

template<class T>
MLFUNC_TYPE void mlVector<T>::Swap(long i, long j)
{
	T temp = data[i];
	data[i] = data[j];
	data[j] = temp;
}

template<class T>
MLFUNC_TYPE void mlVector<T>::Reverse()
{
	long i = 0, j = count - 1;
	while (i < j)
	{
		Swap(i, j);
		i++; j--;
	}
}

template<class T>
MLFUNC_TYPE T mlVector<T>::GetMaxValue()
{
	if (count <= 0) return 0;

	T max_value = data[0];
	for (long i = 1; i < count; i++)
	{
		if (max_value < data[i])
			max_value = data[i];
	}

	return max_value;
}

template<class T>
MLFUNC_TYPE T mlVector<T>::GetMinValue()
{
	if (count <= 0) return 0;

	T min_value = data[0];
	for (long i = 1; i < count; i++)
	{
		if (min_value > data[i])
			min_value = data[i];
	}

	return min_value;
}

template<class T>
MLFUNC_TYPE void mlVector<T>::Clear()
{
	if (data == data_old && data)
	{
		delete[]data;
		data = NULL;
		count = 0;
		data_old = NULL;
		count_old = 0;
	}
	else if (data_old)
	{
		delete[]data_old;
		data_old = NULL;
		count_old = 0;
	}
}

template<class T>
MLFUNC_TYPE void mlVector<T>::Zero()
{
	memset(data, 0, sizeof(T) * count);
}

template<class T>
MLFUNC_TYPE void mlVector<T>::SetConstValue(const T& v)
{
	for (long i = 0; i < count; i++)
		data[i] = v;
}

template<class T>
MLFUNC_TYPE void* mlVector<T>::GetData()
{
	return (void*)data;
}

template<class T>
MLFUNC_TYPE T* mlVector<T>::GetTypedData()
{
	return data;
}

template<class T>
MLFUNC_TYPE long mlVector<T>::GetCount()
{
	return count;
}

template<class T>
MLFUNC_TYPE bool mlVector<T>::Resize(long count)
{
	if (count < 0)
		return false;

	else if (count == 0)
	{
		if (data)
		{
			delete[]data;
			data = NULL;
		}
		this->count = 0;
	}
	else
	{
		if (this->count != count)
		{
			if (data)
				delete[]data;
			data = new T[count];
			if (data)
			{
				memset(data, 0, sizeof(T) * count);
				this->count = count;
			}
			else
			{
				this->count = 0;
				return false;
			}
		}
	}
	data_old = data;
	count_old = this->count;

	return true;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>::Normalize()
{
	T norm = GetSum();
	if (norm != 0)
	{
		for (long i = 0; i < count; i++)
			data[i] /= norm;
	}
	return (*this);
}

template<class T>
MLFUNC_TYPE T mlVector<T>::GetNorm(double p)
{
	double total = 0;
	for (long i = 0; i < count; i++)
		total += pow(double(data[i]), p);
	return (T)pow(double(total), 1.0 / p);
}

template<class T>
MLFUNC_TYPE T mlVector<T>::GetEuclidianNorm()
{
	T total = 0;
	for (long i = 0; i < count; i++)
		total += data[i] * data[i];
	return sqrt(double(total));
}

template<class T>
MLFUNC_TYPE T mlVector<T>::GetManhattanNorm()
{
	T total = 0;
	for (long i = 0; i < count; i++)
		total += fabs(data[i]);
	return total;
}

template<class T>
MLFUNC_TYPE T mlVector<T>::GetSum()
{
	T total = 0;
	for (long i = 0; i < count; i++)
		total += data[i];
	return total;
}

template<class T>
MLFUNC_TYPE T mlVector<T>::GetSquaredSum()
{
	T total = 0;
	for (long i = 0; i < count; i++)
		total += data[i] * data[i];
	return total;
}

//template<class T>
//MLFUNC_TYPE void mlVector<T>::GenRand(T low, T up)
//{
//	for (long i = 0; i < count; i++)
//		data[i] = GVL_Rand<T>(low, up);
//}

template<class T>
MLFUNC_TYPE void mlVector<T>::GenGaussian(T center_v, REAL u, REAL sigma)
{
	long shift = count / 2;
	for (long i = 0; i < count; i++)
	{
		REAL x = REAL(i - shift);
		T v = T(REAL(center_v) * exp(-(x - u) * (x - u) / (2 * sigma * sigma)));
		data[i] = v;
	}
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>::Add(const mlVector<T>& v)
{
	return (*this) += v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>::Sub(const mlVector<T>& v)
{
	return (*this) -= v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>::Mul(const mlVector<T>& v)
{
	return (*this) *= v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>::Div(const mlVector<T>& v)
{
	return (*this) /= v;
}

template<class T>
MLFUNC_TYPE bool mlVector<T>::operator == (mlVector<T>& v)
{
	if (this->GetCount() != v.GetCount())
		return false;
	return memcmp(data, v.data, sizeof(T) * GetCount()) == 0;
}

template<class T>
MLFUNC_TYPE bool mlVector<T>::operator != (mlVector<T>& v)
{
	return !((*this) == v);
}

template<class T>
MLFUNC_TYPE mlVector<T>  mlVector<T>:: operator /  (const mlVector<T>& v)
{
	mlVector<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator /= (const mlVector<T>& v)
{
	for (long i = 0; i < count; i++)
		data[i] /= v.data[i];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector<T>  mlVector<T>:: operator /  (const T& v)
{
	mlVector<T> temp(*this);
	return temp /= v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator /= (const T& v)
{
	for (long i = 0; i < count; i++)
		data[i] /= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector<T>  mlVector<T>:: operator *  (const mlVector<T>& v)
{
	mlVector<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator *= (const mlVector<T>& v)
{
	for (long i = 0; i < count; i++)
		data[i] *= v.data[i];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector<T>  mlVector<T>:: operator *  (const T& v)
{
	mlVector<T> temp(*this);
	return temp *= v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator *= (const T& v)
{
	for (long i = 0; i < count; i++)
		data[i] *= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector<T>  mlVector<T>:: operator -  (const mlVector<T>& v)
{
	mlVector<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator -= (const mlVector<T>& v)
{
	for (long i = 0; i < count; i++)
		data[i] -= v.data[i];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector<T>  mlVector<T>:: operator -  (const T& v)
{
	mlVector<T> temp(*this);
	return temp -= v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator -= (const T& v)
{
	for (long i = 0; i < count; i++)
		data[i] -= v;
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector<T>  mlVector<T>:: operator +  (const mlVector<T>& v)
{
	mlVector<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator += (const mlVector<T>& v)
{
	for (long i = 0; i < count; i++)
		data[i] += v.data[i];
	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector<T>  mlVector<T>:: operator +  (const T& v)
{
	mlVector<T> temp(*this);
	return temp += v;
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator += (const T& v)
{
	for (long i = 0; i < count; i++)
		data[i] += v;
	return (*this);
}

template<class T>
MLFUNC_TYPE T& mlVector<T>:: operator () (long index)
{
	return data[index];
}

template<class T>
MLFUNC_TYPE const T& mlVector<T>:: operator () (long index)const
{
	return data[index];
}

template<class T>
MLFUNC_TYPE T& mlVector<T>:: operator [] (long index)
{
	return data[index];
}

template<class T>
MLFUNC_TYPE const T& mlVector<T>:: operator [] (long index)const
{
	return data[index];
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator = (const mlVector<T>& v)
{
	if (v.count == 0)
	{
		if (data != NULL)
			delete[]data;

		data = NULL;
		count = 0;
	}
	else
	{
		if (data) delete[]data;
		data = new T[v.count];
		if (data != NULL)
		{
			memcpy(data, v.data, sizeof(T) * v.count);
			count = v.count;
		}
		else
			count = 0;
	}

	data_old = data;
	count_old = count;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVector<T>& mlVector<T>:: operator = (const T& v)
{
	if (data != NULL)
	{
		for (long i = 0; i < count; i++)
			data[i] = v;
	}
	else
	{
		data = new T[1];
		if (data != NULL)
		{
			data[0] = v;
			count = 1;
		}
		else
		{
			data = NULL;
			count = 0;
		}
	}

	return (*this);
}

/////////////////////////////////Vector Domain  --XPLIU-- 2006.9.27///////////////////////////////////



#pragma  warning (disable : 4251)

#endif //_mlVector_