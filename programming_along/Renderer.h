#pragma once

#include <vector>

#include "cuda_runtime.h"
#include "cuda.h"
#include "util/CUDABuffer.h"

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

	/**
	* Creates the ray generation program records
	*/
	void CreateRaygenPrograms();

	/**
	* Creates the miss program records
	*/
	void CreateMissPrograms();

	/**
	* Creates the hit group program records
	*/
	void CreateHitgroupPrograms();

	/**
	* Creates the OptiX Pipeline
	*/
	void CreatePipeline();

	/**
	* Builds the shader binding table
	*/
	void BuildShaderBindingTable();


public:

private:
	cudaStream_t CudaStream;
	CUcontext CudaContext;

	OptixDeviceContext OptixContext;
	OptixModule OptixModuleInstance;
	OptixPipeline Pipeline;

	OptixModuleCompileOptions ModuleOptions;
	OptixPipelineCompileOptions PipelineCompileOptions;
	OptixPipelineLinkOptions PipelineLinkOptions;

	OptixShaderBindingTable ShaderBindingTable;
	std::vector<OptixProgramGroup> RaygenProgramGroups;
	CUDABuffer RaygenRecordsBuffer;
	std::vector<OptixProgramGroup> MissProgramGroups;
	CUDABuffer MissRecordsBuffer;
	std::vector<OptixProgramGroup> HitgroupProgramGroups;
	CUDABuffer HitgroupRecordsBuffer;
};