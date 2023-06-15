#pragma once

#include <vector>

#include "cuda_runtime.h"
#include "cuda.h"

#include "util/CUDABuffer.h"
#include "util/Camera.h"

#include "optix_types.h"

#include "gdt/math/vec.h"

#include "LaunchParams.h"

class Renderer
{
public:
	Renderer();
	~Renderer();

	/**
	* Renders one frame
	*/
	void Render();

	/**
	* Sets the framebuffer size to the newly chosen size
	* @param size The newly chosen size
	*/
	void Resize(const vec2i& size);

	/**
	* Download the rendered color buffer from the device into a host array
	*/
	void DownloadPixels(uint32_t pixels[]);

	Camera* GetCameraPtr();

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

protected:
	/** Basic setup of size */
	LaunchParams Params;
	/** Contents of the basic setup transferred to GPU */
	CUDABuffer ParamsBuffer;
	/** Framebuffer contents */
	CUDABuffer ColorBuffer;

	/** Scene */
	Camera SceneCamera;

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