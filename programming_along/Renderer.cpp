#pragma once

#include "Renderer.h"

#include "optix_function_table_definition.h"
#include "optix_stubs.h"

#include "util/SbtStructs.h"

Renderer::Renderer()
{
	InitOptix();
	CreateContext();
	CreateModule();

	CreateRaygenPrograms();
	CreateMissPrograms();
	CreateHitgroupPrograms();

	CreatePipeline();

	BuildShaderBindingTable();

	ParamsBuffer.Alloc(sizeof(LaunchParams));
}

Renderer::~Renderer()
{
	optixPipelineDestroy(Pipeline);

	for (OptixProgramGroup pg : HitgroupProgramGroups)
	{
		optixProgramGroupDestroy(pg);
	}
	HitgroupProgramGroups.clear();
	
	for (OptixProgramGroup pg : MissProgramGroups)
	{
		optixProgramGroupDestroy(pg);
	}
	MissProgramGroups.clear();

	for (OptixProgramGroup pg : RaygenProgramGroups)
	{
		optixProgramGroupDestroy(pg);
	}
	RaygenProgramGroups.clear();

	optixModuleDestroy(OptixModuleInstance);
	optixDeviceContextDestroy(OptixContext);
	cudaStreamDestroy(CudaStream);
}

void Renderer::Render()
{
	// make sure the framebuffer is setup correctly
	if (Params.FramebufferSize.x == 0)
	{
		return;
	}

	// upload the launch params and increment frame ID
	ParamsBuffer.Upload(&Params, 1);
	Params.FrameID++;

	OptixResult result = optixLaunch(
		/* Pipeline to launch */
		Pipeline, CudaStream,
		/* parameters and shader binding table*/
		ParamsBuffer.CudaPtr(), ParamsBuffer.Size_bytes, &ShaderBindingTable,
		/* dimensions of the launch */
		Params.FramebufferSize.x, Params.FramebufferSize.y, 1);

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not execute optixLaunch!");
	}

	// make sure the frame is rendered before it is downloaded (only for this easy example!")
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		const std::string errorString(cudaGetErrorString(error));
		throw std::runtime_error("error synchronizing cuda: " + errorString);
	}
}

void Renderer::Resize(const vec2i& size)
{
	if (size.x == 0 || size.y == 0)
	{
		return;
	}
	Params.FramebufferSize = size;
	ColorBuffer.Resize(size.x * size.y * sizeof(uint32_t));
	Params.FramebufferData = reinterpret_cast<uint32_t*>(ColorBuffer.CudaPtr());
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
	ModuleOptions = {};
	PipelineCompileOptions = {};
	PipelineLinkOptions = {};
	{
		// everything copied mindlessly from example 2
		ModuleOptions.maxRegisterCount = 50;
		ModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		ModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

		PipelineCompileOptions = {};
		PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		PipelineCompileOptions.usesMotionBlur = false;
		PipelineCompileOptions.numPayloadValues = 2;
		PipelineCompileOptions.numAttributeValues = 2;
		PipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		PipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
				
		PipelineLinkOptions.maxTraceDepth = 2;
	}// end of option setup

	// convert the device code to a string for ease of use
	const std::string deviceCode = embedded_ptx_code;

	// use a basic log string
	char log[2048];
	size_t logSize = static_cast<size_t>(sizeof(log));

	optixModuleCreateFromPTX(OptixContext, &ModuleOptions, &PipelineCompileOptions, deviceCode.c_str(), deviceCode.size(),
		log, &logSize, &OptixModuleInstance);
}

void Renderer::CreateRaygenPrograms()
{
	RaygenProgramGroups.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDescr = {};
	pgDescr.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDescr.raygen.module = OptixModuleInstance;
	pgDescr.raygen.entryFunctionName = "__raygen__renderFrame";

	char log[2048];
	size_t logSize = sizeof(log);
	OptixResult result = optixProgramGroupCreate(OptixContext,
		&pgDescr, 1, &pgOptions, log, &logSize, RaygenProgramGroups.data());

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create raygen program!");
	}
}

void Renderer::CreateMissPrograms()
{
	MissProgramGroups.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDescr = {};
	pgDescr.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDescr.raygen.module = OptixModuleInstance;
	pgDescr.raygen.entryFunctionName = "__miss__radiance";

	char log[2048];
	size_t logSize = sizeof(log);
	OptixResult result = optixProgramGroupCreate(OptixContext,
		&pgDescr, 1, &pgOptions, log, &logSize, MissProgramGroups.data());

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create miss program!");
	}
}

void Renderer::CreateHitgroupPrograms()
{
	HitgroupProgramGroups.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDescr = {};
	pgDescr.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDescr.raygen.module = OptixModuleInstance;
	pgDescr.raygen.entryFunctionName = "__hitgroup__radiance";

	char log[2048];
	size_t logSize = sizeof(log);
	OptixResult result = optixProgramGroupCreate(OptixContext,
		&pgDescr, 1, &pgOptions, log, &logSize, HitgroupProgramGroups.data());

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create hitgroup program!");
	}
}

void Renderer::CreatePipeline()
{
	std::vector<OptixProgramGroup> programGroups;
	programGroups.reserve(RaygenProgramGroups.size() + MissProgramGroups.size() + HitgroupProgramGroups.size());
	programGroups.insert(programGroups.end(), RaygenProgramGroups.begin(), RaygenProgramGroups.end());
	programGroups.insert(programGroups.end(), MissProgramGroups.begin(), MissProgramGroups.end());
	programGroups.insert(programGroups.end(), HitgroupProgramGroups.begin(), HitgroupProgramGroups.end());

	char log[2048];
	size_t logSize = sizeof(log);
	OptixResult result = optixPipelineCreate(OptixContext, &PipelineCompileOptions,
		&PipelineLinkOptions, programGroups.data(), (int32_t)programGroups.size(), log, &logSize,
		&Pipeline);

	// write log before throwing error, as it may contain more useful information
	if (logSize > 1)
	{
		std::cout << log << std::endl;
	}

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create OptiX pipeline!");
	}
	
	result = optixPipelineSetStackSize(
		Pipeline,
		/* The direct stack size requirement for direct callables invoked from
		intersection or any hit*/
		2 * 1024,
		/* The direct stack size requirement for direct callables invoked from
		ray generation, miss or closest hit*/
		2 * 1024,
		/* the continuation stack requirement*/
		2 * 1024,
		/* the maximum depth of a traversable graph passed to trace*/
		1
	);
	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not set pipeline stack size!");
	}
}

void Renderer::BuildShaderBindingTable()
{
	ShaderBindingTable = {};

	// ray generation records
	std::vector<RaygenRecord> raygenRecords;
	for (size_t i = 0; i < RaygenProgramGroups.size(); i++)
	{
		RaygenRecord rec;
		OptixResult result = optixSbtRecordPackHeader(RaygenProgramGroups[i], &rec);
		if (result != OPTIX_SUCCESS)
		{
			throw std::runtime_error("Could not build raygen record!");
		}

		rec.Data = nullptr;
		raygenRecords.push_back(rec);
	}
	RaygenRecordsBuffer.AllocAndUpload(raygenRecords);
	ShaderBindingTable.raygenRecord = RaygenRecordsBuffer.CudaPtr();

	// miss generation records
	std::vector<MissRecord> missRecords;
	for (size_t i = 0; i < MissProgramGroups.size(); i++)
	{
		MissRecord rec;
		OptixResult result = optixSbtRecordPackHeader(MissProgramGroups[i], &rec);
		if (result != OPTIX_SUCCESS)
		{
			throw std::runtime_error("Could not build raygen record!");
		}

		rec.Data = nullptr;
		missRecords.push_back(rec);
	}
	MissRecordsBuffer.AllocAndUpload(missRecords);
	ShaderBindingTable.missRecordBase = MissRecordsBuffer.CudaPtr();
	ShaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
	ShaderBindingTable.missRecordCount = (int32_t)missRecords.size();

	// hit group generation records
	// TODO: nothing yet, add dummy record
	int32_t numObjects = 1;
	std::vector<HitgroupRecord> hitgroupRecords;
	for (size_t i = 0; i < numObjects; i++)
	{
		int32_t objectType = 0;
		HitgroupRecord rec;
		OptixResult result = optixSbtRecordPackHeader(HitgroupProgramGroups[objectType], &rec);
		if (result != OPTIX_SUCCESS)
		{
			throw std::runtime_error("Could not build raygen record!");
		}

		rec.ObjectId = i;
		hitgroupRecords.push_back(rec);
	}
	HitgroupRecordsBuffer.AllocAndUpload(hitgroupRecords);
	ShaderBindingTable.hitgroupRecordBase = HitgroupRecordsBuffer.CudaPtr();
	ShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	ShaderBindingTable.hitgroupRecordCount = (int32_t)hitgroupRecords.size();
}
