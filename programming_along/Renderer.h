#pragma once

#include "cuda_runtime.h"
#include "cuda.h"

#include "optix_types.h"

#include "gdt/math/vec.h"

class Renderer
{
public:
	Renderer();
	~Renderer();

private:
	/**
	* Does a few basic CUDA and OptiX operations to check that everything is available
	*/
	void InitOptix();

	/**
	* Creates the CUDA Device Context and the OptiX Device Context
	*/
	void CreateContext();

	/**
	* Creates the OptiX Module with the given PTX code
	*/
	void CreateModule();

public:

private:
	cudaStream_t CudaStream;
	CUcontext CudaContext;

	OptixDeviceContext OptixContext;
	OptixModule OptixModuleInstance;
};