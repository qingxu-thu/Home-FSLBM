cmake_minimum_required ( VERSION 3.10 )
project(lbm_flow_proj)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")
# On Linux (and Mac OS) we use conan for package management. Therefore, we include ${CMAKE_BINARY_DIR}
# in CMAKE_MODULE_PATH
if(NOT WIN32)
	list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
endif()

# Set a default build configuration (Release)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified.")
	set (CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
	"MinSizeRel" "RelWithDebInfo")
endif()
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW) # New-style link interface
endif()
option(FLR_WITH_CUDA  "Build GPU simulation on CUDA" ON)
option(FLR_ENABLE_LTO "Enable Link Time Optimization (LTO)?" OFF)

set(CMAKE_CUDA_FLAGS_DEBUG "-g")


set(CUDA_ARCHITECTURES 86) 
if (FLR_WITH_CUDA)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
	enable_language(CUDA)
	add_definitions(-DFLR_WITH_CUDA)
	find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
  endif()
	# include(check_cuda)
	# check_cuda_sm()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
endif()
FIND_PACKAGE( OpenMP REQUIRED)
    if(OPENMP_FOUND)
        message("OPENMP FOUND")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()


set(INC_DIR "${CMAKE_SOURCE_DIR}")
add_executable(${PROJECT_NAME}
    ${INC_DIR}/src/testMrLBM3D_bubble.cpp
    ${INC_DIR}/src/testMrLBM2D.cpp

)


set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/eigen)
target_include_directories(${PROJECT_NAME}
    PRIVATE 
    ./3rdParty/
    ${INC_DIR}/common
    ${INC_DIR}/inc/2D/cpu
    ${INC_DIR}/inc/2D/gpu
    ${INC_DIR}/inc/3D/cpu
    ${INC_DIR}/inc/3D/gpu
   
)



target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${PROJECT_NAME}-cuda
        ${PROJECT_NAME}-cpu
)

target_compile_features ( ${PROJECT_NAME} 
	PUBLIC 
		# cuda_std_14
		cxx_std_14
)

# ------- CPU library ------ SHARED
add_library(${PROJECT_NAME}-cpu STATIC
     ${INC_DIR}/common/colorramp.cpp      ${INC_DIR}/common/colorramp.h 
     )

target_include_directories(${PROJECT_NAME}-cpu
    PRIVATE 
        /usr/local/cuda/samples/common/inc/
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
         CUDA::cudart 
        ${PROJECT_NAME}-cuda
        ${OpenCV_LIBS}
)

# ------- CUDA library ------
add_library(${PROJECT_NAME}-cuda STATIC
     ${INC_DIR}/common/mlCuRunTime.cpp      ${INC_DIR}/common/mlCuRunTime.h 
     ${INC_DIR}/inc/3D/gpu/mrLbmSolverGpu3D.cu      ${INC_DIR}/inc/3D/gpu/mrLbmSolverGpu3D.h 
     ${INC_DIR}/inc/2D/gpu/mrLbmSolverGpu2D.cu      ${INC_DIR}/inc/2D/gpu/mrLbmSolverGpu2D.h 
     ${INC_DIR}/inc/2D/gpu/CCL.cu      ${INC_DIR}/inc/2D/gpu/CCL.cuh
     ${INC_DIR}/inc/3D/gpu/tDCCL.cu      ${INC_DIR}/inc/3D/gpu/tDCCL.cuh
     ${INC_DIR}/inc/3D/gpu/cuda_types3.hpp
     ${INC_DIR}/inc/2D/gpu/reduction.cuh
     )

 
# set_property(TARGET ${PROJECT_NAME}-cuda PROPERTY CUDA_ARCHITECTURES ${FLR_CUDA_ARCH})
set_property(TARGET ${PROJECT_NAME}-cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME}-cuda PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME}-cuda PROPERTY CUDA_STANDARD 14)

target_include_directories(${PROJECT_NAME}-cuda
    PRIVATE 
        # ${CMAKE_SOURCE_DIR}/ext/cuda_helper
        # $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/flare>
)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_options(${PROJECT_NAME}-cuda
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        # -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr
)

target_compile_definitions(${PROJECT_NAME}  PRIVATE $<$<CONFIG:Debug>:DEBUG>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()