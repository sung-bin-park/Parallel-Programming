# Start Parallel_Programming using CUDA & Visual Studio 2019 for C++

1.You have to install CUDA
->https://developer.nvidia.com/cuda-downloads

2.Set Visual Studio
build dependency -> custom build -> check the box CUDA (version)

3.Set file extension to using cu,cuh
tool -> file extension -> add cu, cuh 

4.set CUDA_PATH on Additional include directory
project property -> C/C++ -> general -> addtional include directory(add (CUDA_PATH\include))

5.set Addtional library directory
project property -> linker -> general -> CUDA_PATH\lib\x64(or 32bit)
and also set platform 32bit or 64bit same as your environment

6.set additional dependency
project property -> linker -> input -> add cuda.lib,cufft.lib,cublas.lib

7.Important!
To using CUDA functions on C++, you have to add below.

CGPUACC.cu

#include "CGPUACC.cuh"  
#include "cuda.h"
#include <iostream>
#include <cufft.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
CGPUACC::CGPUACC(void)
{
}
CGPUACC::~CGPUACC(void)
{
}

CGPUACC.cuh

#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus 
extern "C" {
#endif
	class CGPUACC
	{
	public:
		CGPUACC(void);
		virtual ~CGPUACC(void);
	};
#ifdef __cplusplus 
}
#endif

When you added above code, then you can using CUDA functions on C++.
