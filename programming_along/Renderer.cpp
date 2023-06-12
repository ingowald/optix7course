#pragma once

#include "Renderer.h"

#include "optix_function_table_definition.h"
#include "optix_stubs.h"

#include "SbtStructs/SbtStructs.h"

Renderer::Renderer()
{
	InitOptix();
	CreateContext();
	CreateModule();
}

Renderer::~Renderer()
{
	optixDeviceContextDestroy(OptixContext);
	cudaStreamDestroy(CudaStream);
}

void Renderer::InitOptix()
{
	// check that CUDA works and a CUDA capable device is found
	cudaFree(0);
	int32_t numDevices = 0;
	cudaGetDeviceCount(&numDevices);
	if (numDevices <= 0)
	{
		throw std::runtime_error("No CUDA Device available!");
	}

	std::cout << "Found " << std::to_string(numDevices) << " CUDA capable devices!" << std::endl;

	OptixResult result = optixInit();
	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not initialize OptiX!");
	}

	std::cout << "OptiX initialized!" << std::endl;
}

void Renderer::CreateContext()
{
	// create the CUDA and OptiX context
	// simply assume the first device is the correct one
	cudaError_t result = cudaSetDevice(0);
	if (result != cudaSuccess)
	{
		throw std::runtime_error("Could not set CUDA device!");
	}

	cudaStreamCreate(&CudaStream);
	if (result != cudaSuccess)
	{
		throw std::runtime_error("Could not set CUDA device!");
	}

	CUresult cuResult = cuCtxGetCurrent(&CudaContext);
	if (cuResult != CUDA_SUCCESS)
	{
		throw std::runtime_error("Could not get CUDA context!");
	}

	OptixDeviceContextOptions optixOptions;

	OptixResult opResult = optixDeviceContextCreate(CudaContext, &optixOptions, &OptixContext);
	if (opResult != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create OptiX context!");
	}
}

extern "C" char embedded_ptx_code[];
void Renderer::CreateModule()
{
	// setup the compile options for the module and the pipeline
	OptixModuleCompileOptions moduleOptions;
	OptixPipelineCompileOptions pipelineOptions;
	{
		// everything copied mindlessly from example 2
		moduleOptions.maxRegisterCount = 50;
		moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

		pipelineOptions = {};
		pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipelineOptions.usesMotionBlur = false;
		pipelineOptions.numPayloadValues = 2;
		pipelineOptions.numAttributeValues = 2;
		pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		pipelineOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
		
		// this is used in a different location on example 2, but written here
		OptixPipelineLinkOptions pipelineLinkOptions;
		pipelineLinkOptions.maxTraceDepth = 2;
	}// end of option setup

	// convert the device code to a string for ease of use
	const std::string deviceCode = embedded_ptx_code;

	// use a basic log string
	char log[2048];
	size_t logSize = static_cast<size_t>(sizeof(log));

	optixModuleCreateFromPTX(OptixContext, &moduleOptions, &pipelineOptions, deviceCode.c_str(), deviceCode.size(),
		log, &logSize, &OptixModuleInstance);
}