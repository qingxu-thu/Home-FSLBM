#pragma once
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <omp.h>
//#include <objbase.h>
static const double INFTY = 1.0e32;

#define SMALLNUMBER	1.0e-5
#define HUGENUMBER	1.0e10
#define Sqr(x)		((x) * (x))

#define NumOfNeig 8 //map neighbor
#define NumOfbasis 5
#define Mach 340
#define QV 27
#define CS 5e-3  //Smagorinsky constants
#define Epsilon 20
#define M_PI 3.1415926*1
#define MLPI 3.1415926
#define  boundaryNum 4
#if (defined(WIN32) || defined(_WIN32) || defined(WINCE) || defined(__CYGWIN__))
#   define PACKED(__declare__) __pragma(pack(push,1)) __declare__ __pragma(pack(pop)) 
#else
#   define PACKED(__declare__) __declare__ __attribute__((__packed__))
#endif
// -----------------------------------------------------------------------------
// Assertion with message
// -----------------------------------------------------------------------------
#ifndef __FUNCTION_NAME__
#if defined(_WIN32) || defined(__WIN32__)
#define __FUNCTION_NAME__ __FUNCTION__
#else
#define __FUNCTION_NAME__ __func__
#endif
#endif

#define Assertion(PREDICATE, MSG) \
do {\
if (!(PREDICATE)) {	\
	std::cerr << "Asssertion \"" \
	<< #PREDICATE << "\" failed in " << __FILE__ \
	<< " line " << __LINE__ \
	<< " in function \"" << (__FUNCTION_NAME__) << "\"" \
	<< " : " << (MSG) << std::endl; \
	std::abort(); \
} \
} while (false)

//#pragma comment(lib,"3rdparty/opengl/glfw-3.2.1.bin.WIN64/lib-vc2015/glfw3.lib")
//#pragma comment(lib,"3rdparty/opengl/glew-1.13.0/lib/Release/x64/glew32.lib")


//#pragma comment(lib,"3rdParty/SDK/x64/cudart.lib")
//#pragma comment(lib,"3rdParty/SDK/x64/cuda.lib")
//#pragma comment(lib,"3rdParty/SDK/x64/cusparse.lib")
//#pragma comment(lib,"3rdParty/SDK/x64/cusolver.lib")
//#pragma comment(lib,"3rdParty/SDK/x64/cublas.lib")


//parameter

const double LID_VELOCITY = 0.05; // lid velocity in lattice units


/* ------------------------------ MEMORY SIZE ------------------------------ */
#define BLOCK_NX 4
#define BLOCK_NY 4
#define BLOCK_NZ 4

#define BLOCK_X 8
#define BLOCK_Y 4
#define BLOCK_Z 4

#define BLOCK_LBM_SIZE (BLOCK_NX * BLOCK_NY * BLOCK_NZ)

#define QF  9         // number of velocities on each face