#pragma once
#ifndef MLCOREWIN_H
#define MLCOREWIN_H

// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the GVLCORE_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// GVLCORE_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef GVLCORE_EXPORTS
#define GVLCORE_API __declspec(dllexport)
#else
#define GVLCORE_API __declspec(dllimport)
#endif




#ifndef MLFUNC_TYPE
#ifdef MLCUDA_DEVICE	
#define MLFUNC_TYPE __host__ __device__
#else
#define MLFUNC_TYPE
#endif//MLCUDA_DEVICE
#endif//MLFUNC_TYPE

#define REAL float

#endif //MLCOREWIN_H