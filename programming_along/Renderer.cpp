#pragma once

#include "Renderer.h"

#include "optix_function_table_definition.h"
#include "optix_stubs.h"

#include "util/SbtStructs.h"
#include "scene/Model.h"
#include "scene/Mesh.h"

Renderer::Renderer()
{
	InitOptix();
	CreateContext();
	CreateModule();

	CreateRaygenPrograms();
	CreateMissPrograms();
	CreateHitgroupPrograms();

	CreatePipeline();

	SceneCamera = std::make_shared<Camera>();
	DynamicElements.push_back(SceneCamera);
}

Renderer::~Renderer()
{
	optixDenoiserDestroy(Denoiser);

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

void Renderer::OptixLogCallback(uint32_t level, const char* tag, const char* message, void* cbdata)
{
	std::string outStr = tag + std::string(" (") + std::to_string(level) + "): " + (message ? message : "no message");
	if (std::string(tag).compare("ERROR") == 0)
	{
		std::cerr << outStr << std::endl;
	}
	else
	{
		std::cout << outStr << std::endl;
	}
}

void Renderer::Init()
{
	// initialize everything which needed mesh or camera information,
	// as these may be set up from the outside (i.e. after ctor of Renderer)
	Params.Traversable = BuildAccelerationStructure();
	CreateTextures();
	BuildShaderBindingTable();

	ParamsBuffer.Alloc(sizeof(LaunchParams));

	IsInitialized = true;
}

void Renderer::Tick(const float& deltaTime_seconds)
{
	SceneCamera->Tick(deltaTime_seconds);

	for (std::shared_ptr<Model> model : ModelList)
	{
		model->Tick(deltaTime_seconds);
	}

	for (std::shared_ptr<Light> light : LightList)
	{
		light->Tick(deltaTime_seconds);
	}
}

void Renderer::Render()
{
	// make sure the framebuffer is setup correctly
	if (Params.FramebufferSize.x == 0)
	{
		return;
	}

	assert(IsInitialized && "Init has not been called. You should do this before rendering!");

	// update the camera values
	Params.Camera = SceneCamera->GetOptixCamera();

	// update light values
	std::shared_ptr<LightOptix> l = LightList[0]->GetOptixLight();
	Params.Light = *((QuadLightOptix*)l.get());

	// upload the launch params and increment frame ID
	if (!AccumulatedDenoiseImages || HasSceneChanged())
	{
		// prevent accumulation if disabled by "going back" to first frame
		Params.FrameID = 0;
	}
	ParamsBuffer.Upload(&Params, 1);
	Params.FrameID++;

	OptixResult result = optixLaunch(
		/* Pipeline to launch */
		Pipeline, CudaStream,
		/* parameters and shader binding table*/
		ParamsBuffer.CudaPtr(), ParamsBuffer.Size_bytes, &ShaderBindingTable,
		/* dimensions of the launch (i.e. of the threads in x/y/z) */
		Params.FramebufferSize.x, Params.FramebufferSize.y, 1);

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not execute optixLaunch!");
	}

	// framebuffer is written to, now denoise and write to denoised image buffer
	// this can be done before cuda synch (-> asynch with regards to the new image rendering)
	OptixDenoiserParams denoiserParams;
	denoiserParams.hdrIntensity = (CUdeviceptr)0;
	if (AccumulatedDenoiseImages)
	{
		denoiserParams.blendFactor = 1.f / Params.FrameID;
	}
	else
	{
		denoiserParams.blendFactor = 0.f;
	}

	OptixImage2D inputLayer;
	inputLayer.data = ColorBuffer.CudaPtr();
	inputLayer.width = Params.FramebufferSize.x;
	inputLayer.height = Params.FramebufferSize.y;
	inputLayer.rowStrideInBytes = Params.FramebufferSize.x * sizeof(float4);
	inputLayer.pixelStrideInBytes = sizeof(float4);
	inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

	OptixImage2D outputLayer;
	outputLayer.data = DenoisedBuffer.CudaPtr();
	outputLayer.width = Params.FramebufferSize.x;
	outputLayer.height = Params.FramebufferSize.y;
	outputLayer.rowStrideInBytes = Params.FramebufferSize.x * sizeof(float4);
	outputLayer.pixelStrideInBytes = sizeof(float4);
	outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

	if (DenoiserEnabled)
	{
		result = optixDenoiserInvoke(Denoiser,
			0, //stream
			&denoiserParams,
			DenoiserState.CudaPtr(),
			DenoiserState.Size_bytes,
			&inputLayer, 1,
			0, // input offset x
			0, // input offset y
			&outputLayer,
			DenoiserScratch.CudaPtr(),
			DenoiserScratch.Size_bytes
		);

		if (result != OPTIX_SUCCESS)
		{
			throw std::runtime_error("OptiX Denoiser invocation failed!");
		}
	}
	else
	{
		cudaMemcpy((void*)outputLayer.data, (void*)inputLayer.data,
			outputLayer.width * outputLayer.height * sizeof(float4),
			cudaMemcpyDeviceToDevice);
	}

	// make sure the frame is rendered before it is downloaded (only for this easy example!")
	SynchCuda("Error synchronizing CUDA after rendering!");
}

void Renderer::Resize(const vec2i& size)
{
	if (size.x == 0 || size.y == 0)
	{
		return;
	}

	if (Denoiser)
	{
		OptixResult result = optixDenoiserDestroy(Denoiser);

		if (result != OPTIX_SUCCESS)
		{
			throw std::runtime_error("Could not destroy OptiX denoiser!");
		}
	}



	//(re-)create the denoiser
	OptixDenoiserOptions denoiserOptions = {};
	denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB;

	OptixResult result = optixDenoiserCreate(OptixContext, &denoiserOptions, &Denoiser);
	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create OptiX denoiser!");
	}

	result = optixDenoiserSetModel(Denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, nullptr, 0);
	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not set OptiX denoiser model!");
	}

	OptixDenoiserSizes denoiserReturnSizes;
	result = optixDenoiserComputeMemoryResources(Denoiser, size.x, size.y, &denoiserReturnSizes);
	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not compute memory resource sizes for OptiX denoiser");
	}

	DenoiserScratch.Resize(
		std::max(
			denoiserReturnSizes.withOverlapScratchSizeInBytes,
			denoiserReturnSizes.withoutOverlapScratchSizeInBytes
		));
	DenoiserState.Resize(denoiserReturnSizes.stateSizeInBytes);

	// resize the CUDA framebuffer and denoised buffer
	const size_t bufferSize = size.x * size.y * sizeof(float4);
	Params.FramebufferSize = size;
	ColorBuffer.Resize(bufferSize);
	Params.FramebufferData = reinterpret_cast<float4*>(ColorBuffer.CudaPtr());
	SceneCamera->SetFramebufferSize(size);
	DenoisedBuffer.Resize(bufferSize);

	// finally setup the denoiser
	result = optixDenoiserSetup(Denoiser, 0,
		size.x, size.y,
		DenoiserState.CudaPtr(),
		DenoiserState.Size_bytes,
		DenoiserScratch.CudaPtr(),
		DenoiserScratch.Size_bytes);

}

void Renderer::DownloadPixels(vec4f pixels[])
{
	DenoisedBuffer.Download(pixels, Params.FramebufferSize.x * Params.FramebufferSize.y);
}

Camera* Renderer::GetCameraPtr()
{
	return SceneCamera.get();
}

void Renderer::InitializeCamera(const vec3f& eye, const vec3f& at, const vec3f& up)
{
	SceneCamera->SetEye(eye);
	SceneCamera->SetAt(at);
	SceneCamera->SetUp(up);

	SceneCamera->UpdateInitialEyeAtUp();

	Params.FrameID = 0;
}

void Renderer::AddMesh(std::shared_ptr<Mesh> mesh)
{
	AddModel(std::make_shared<Model>(mesh));
}

void Renderer::AddModel(std::shared_ptr<Model> model)
{
	ModelList.push_back(model);
}

void Renderer::AddLight(std::shared_ptr<Light> light)
{
	assert(LightList.size() == 0 && "Currently only one light source is supported!");
	LightList.push_back(light);

	if (light->IsDynamic())
	{
		DynamicElements.push_back(light);
	}
}

bool Renderer::GetDynamicLightsMovementsEnabled() const
{
	return DynamicLightsMovementsEnabled;
}

void Renderer::EnableDynamicLightsMovements(const bool& enabled)
{
	DynamicLightsMovementsEnabled = enabled;

	for (std::shared_ptr<Light> l : LightList)
	{
		IDynamicElement* dyn = dynamic_cast<IDynamicElement*>(l.get());
		if (dyn)
		{
			dyn->SetDynamicEnabled(enabled);
		}
	}
}

void Renderer::ToggleDynamicLightsMovement()
{
	DynamicLightsMovementsEnabled = !DynamicLightsMovementsEnabled;
	EnableDynamicLightsMovements(DynamicLightsMovementsEnabled);
}

bool Renderer::GetDenoiserEnabled() const
{
	return DenoiserEnabled;
}

void Renderer::SetDenoiserEnabled(const bool& enabled)
{
	DenoiserEnabled = enabled;
}

void Renderer::ToggleDenoiserEnabled()
{
	DenoiserEnabled = !DenoiserEnabled;
}

bool Renderer::GetAccumulationEnabled() const
{
	return AccumulatedDenoiseImages;
}

void Renderer::SetAccumulationEnabled(const bool& enabled)
{
	AccumulatedDenoiseImages = enabled;
}

void Renderer::ToggleAccumulationEnabled()
{
	AccumulatedDenoiseImages = !AccumulatedDenoiseImages;
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

	OptixDeviceContextOptions options = {};
#ifdef NDEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#else
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
	options.logCallbackFunction = &OptixLogCallback;
	options.logCallbackLevel = 4;
#endif

	OptixResult opResult = optixDeviceContextCreate(CudaContext, &options, &OptixContext);
	if (opResult != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create OptiX context!");
	}

	std::cout << "OptiX context created!" << std::endl;
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

	OptixResult result = optixModuleCreateFromPTX(OptixContext, &ModuleOptions, &PipelineCompileOptions, 
		deviceCode.c_str(), deviceCode.size(),
		log, &logSize, &OptixModuleInstance);

	if (result != OPTIX_SUCCESS)
	{
		std::cerr << log << std::endl;
		throw std::runtime_error("Could not create OptiX module!");
	}

	std::cout << "OptiX module created!" << std::endl;
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

	if (logSize > 1)
	{
		std::cout << log << std::endl;
	}

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create raygen program!");
	}

	std::cout << "Raygen program created!" << std::endl;
}

void Renderer::CreateMissPrograms()
{
	MissProgramGroups.resize(RAY_TYPE_COUNT);

	//---------------------
	//--- Radiance Rays ---
	//---------------------

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDescr = {};
	pgDescr.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDescr.raygen.module = OptixModuleInstance;
	pgDescr.raygen.entryFunctionName = "__miss__radiance";

	char log[2048];
	size_t logSize = sizeof(log);
	OptixResult result = optixProgramGroupCreate(OptixContext,
		&pgDescr, 1, &pgOptions, log, &logSize, &MissProgramGroups[RADIANCE_RAY_TYPE]);

	if (logSize > 1)
	{
		std::cout << log << std::endl;
	}

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create radiance miss program!");
	}

	std::cout << "Miss program for radiance rays created!" << std::endl;

	//-------------------
	//--- Shadow Rays ---
	//-------------------

	// we can reuse description and options of the radiance rays
	pgDescr.raygen.entryFunctionName = "__miss__shadow";

	result = optixProgramGroupCreate(OptixContext,
		&pgDescr, 1, &pgOptions, log, &logSize, &MissProgramGroups[SHADOW_RAY_TYPE]);

	if (logSize > 1)
	{
		std::cout << log << std::endl;
	}

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create shadow miss program!");
	}

	std::cout << "Miss program for shadow rays created!" << std::endl;

}

void Renderer::CreateHitgroupPrograms()
{
	HitgroupProgramGroups.resize(RAY_TYPE_COUNT);

	//---------------------
	//--- Radiance Rays ---
	//---------------------

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDescr = {};
	pgDescr.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDescr.hitgroup.moduleCH = OptixModuleInstance;
	pgDescr.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDescr.hitgroup.moduleAH = OptixModuleInstance;
	pgDescr.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	char log[2048];
	size_t logSize = sizeof(log);
	OptixResult result = optixProgramGroupCreate(OptixContext,
		&pgDescr, 1, &pgOptions, log, &logSize, &HitgroupProgramGroups[RADIANCE_RAY_TYPE]);

	if (logSize > 1)
	{
		std::cout << log << std::endl;
	}

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not radiance create hitgroup program!");
	}

	std::cout << "Hit group program for radiance created!" << std::endl;
	
	//-------------------
	//--- Shadow Rays ---
	//-------------------

	// we can reuse the previously created options
	// also: technically not needed since only the miss program is used
	pgDescr.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
	pgDescr.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

	result = optixProgramGroupCreate(OptixContext,
		&pgDescr, 1, &pgOptions, log, &logSize, &HitgroupProgramGroups[SHADOW_RAY_TYPE]);

	if (logSize > 1)
	{
		std::cout << log << std::endl;
	}

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not create shadow hitgroup program!");
	}

	std::cout << "Hit group program for shadow created!" << std::endl;
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

void Renderer::CreateTextures()
{
	int32_t numTextures = 0;
	for (std::shared_ptr<Model> model : ModelList)
	{
		numTextures += static_cast<int32_t>(model->GetTextureList().size());
	}

	TextureArrays.resize(numTextures);
	TextureObjects.resize(numTextures);

	for (std::shared_ptr<Model> model : ModelList)
	{
		for (size_t texId = 0; texId < model->GetTextureList().size(); texId++)
		{
			uint32_t textureIndex = GetTextureBufferIndex(model, static_cast<uint32_t>(texId));
			std::shared_ptr<Texture2D> texture = model->GetTextureList()[texId];

			cudaResourceDesc resourceDesc = {};
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
			const int32_t numComponents = 4;
			const int32_t pitch = texture->Resolution.x * numComponents * sizeof(uint8_t);

			cudaArray_t& pixelArray = TextureArrays[textureIndex];
			cudaError_t result = cudaMallocArray(&pixelArray, &channelDesc,
				texture->Resolution.x, texture->Resolution.y);

			if (result != cudaSuccess)
			{
				throw std::runtime_error("could allocate CUDA array for texture of model " + model->GetName() + "!");			}

			result = cudaMemcpy2DToArray(pixelArray,
				0, 0,	//wOffset, hOffset
				texture->Pixels,
				pitch, pitch, texture->Resolution.y,	// it's important to use width as "width in bytes"
														// i.e. here width_pixels * components * sizeof component
														// which is equal to pitch.
														// TODO: why is the pitch needed as an extra parameter then?
														//		seem to always be the same
				cudaMemcpyHostToDevice);

			if (result != cudaSuccess)
			{
				throw std::runtime_error("could not copy texture to CUDA array of model " + model->GetName() + "!");
			}

			resourceDesc.resType = cudaResourceTypeArray;
			resourceDesc.res.array.array = pixelArray;

			cudaTextureDesc texDesc = {};
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeNormalizedFloat;
			texDesc.normalizedCoords = 1;
			texDesc.maxAnisotropy = 1;
			texDesc.maxMipmapLevelClamp = 99;
			texDesc.minMipmapLevelClamp = 1;
			texDesc.mipmapFilterMode = cudaFilterModePoint;
			texDesc.borderColor[0] = 1.0f;
			texDesc.sRGB = 0;

			// create the actual CUDA texture object
			cudaTextureObject_t cudaTex = 0;
			result = cudaCreateTextureObject(&cudaTex, &resourceDesc, &texDesc, nullptr);

			if (result != cudaSuccess)
			{
				const std::string errorString(cudaGetErrorString(result));
				throw std::runtime_error("error creating CUDA texture object: " + errorString);
			}
			TextureObjects[textureIndex] = cudaTex;
		}
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
	// there are n meshes and m ray types, so n*m hitgroup records
	// currently there is only one ray type, so n hitgroup records
	std::vector<HitgroupRecord> hitgroupRecords;
	for (std::shared_ptr<Model> model : ModelList)
	{
		const std::vector<std::shared_ptr<Mesh>>& meshList = model->GetMeshList();
		for (size_t meshId = 0; meshId < meshList.size(); meshId++)
		{
			std::shared_ptr<Mesh> mesh = meshList[meshId];
			const uint32_t bufferIndex = GetMeshBufferIndex(model, static_cast<uint32_t>(meshId));

			// a hit group record per mesh and per ray type is needed
			for (size_t rayTypeId = 0; rayTypeId < RAY_TYPE_COUNT; rayTypeId++)
			{
				HitgroupRecord rec;

				// all the meshes use the same kernel(/shader), therefore all use the same hit group
				OptixResult result = optixSbtRecordPackHeader(HitgroupProgramGroups[0], &rec);
				if (result != OPTIX_SUCCESS)
				{
					throw std::runtime_error("Could not build raygen record!");
				}

				// TODO: only upload normals and texcoords, if they are available?
				//		what does CUDA do, if the uploaded array size is 0?
				rec.MeshData.DiffuseColor = mesh->DiffuseColor;
				rec.MeshData.Vertices = (vec3f*)VertexBufferList[bufferIndex].CudaPtr();
				rec.MeshData.Normals = (vec3f*)NormalBufferList[bufferIndex].CudaPtr();
				rec.MeshData.Indices = (vec3i*)IndexBufferList[bufferIndex].CudaPtr();
				rec.MeshData.TexCoords = (vec2f*)TexCoordsBufferList[bufferIndex].CudaPtr();

				// setup textures, if applicable
				if (model->GetTextureList().size() > 0)
				{
					if (mesh->DiffuseTextureId >= 0)
					{
						rec.MeshData.HasTexture = true;
						rec.MeshData.Texture = TextureObjects[mesh->DiffuseTextureId];
					}
				}

				hitgroupRecords.push_back(rec);
			}
		}
	}
	
	HitgroupRecordsBuffer.AllocAndUpload(hitgroupRecords);
	ShaderBindingTable.hitgroupRecordBase = HitgroupRecordsBuffer.CudaPtr();
	ShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	ShaderBindingTable.hitgroupRecordCount = (int32_t)hitgroupRecords.size();
}

OptixTraversableHandle Renderer::BuildAccelerationStructure()
{
	const uint32_t totalNumberMeshes = GetNumberMeshesFromScene();
	VertexBufferList.resize(totalNumberMeshes);
	NormalBufferList.resize(totalNumberMeshes);
	IndexBufferList.resize(totalNumberMeshes);
	TexCoordsBufferList.resize(totalNumberMeshes);

	OptixTraversableHandle handle = {};

	if (totalNumberMeshes == 0)
	{
		std::cout << "No meshes to build acceleration structure for!" << std::endl;
		return handle;
	}

	std::cout << "Building acceleration structure for " 
		<< std::to_string(totalNumberMeshes) 
		<< " meshes in "
		<< std::to_string(ModelList.size()) 
		<< " models." 
		<< std::endl;

	// triangle inputs
	//	-> literally triangle, i.e. only vertex and index data
	//		no normals, texcoords or anything else
	// 
	//		instead, the buffers for normals, texcoords, etc
	//		are simply allocated and uploaded, then pointed to 
	//		in the hitgroup record, but not otherwise handled in this
	//		triangle input data

	std::vector<OptixBuildInput> triangleInputList(totalNumberMeshes);
	std::vector<CUdeviceptr> cudaVertexBufferList(totalNumberMeshes);
	std::vector<CUdeviceptr> cudaIndexBufferList(totalNumberMeshes);
	std::vector<uint32_t> triangleInputFlagsList(totalNumberMeshes);

	// create a temporary model list, such that other models can be added to the scene
	std::vector<std::shared_ptr<Model>> tempModelList = ModelList;
	
	for (std::shared_ptr<Light> light : LightList)
	{
		if (light->GetShowProxyMesh())
		{
			//tempModelList.push_back(light->GetProxy());
		}
	}

	for (std::shared_ptr<Model> model : tempModelList)
	{
		std::vector<std::shared_ptr<Mesh>>& meshList = model->GetMeshList();
		for (size_t meshId = 0; meshId < meshList.size(); meshId++)
		{
			const uint32_t bufferIndex = GetMeshBufferIndex(model, static_cast<uint32_t>(meshId));
			std::shared_ptr<Mesh> mesh = meshList[meshId];
			VertexBufferList[bufferIndex].AllocAndUpload(mesh->Vertices);
			if (!mesh->Normals.empty())
			{
				NormalBufferList[bufferIndex].AllocAndUpload(mesh->Normals);
			}
			IndexBufferList[bufferIndex].AllocAndUpload(mesh->Indices);
			if (!mesh->TexCoords.empty())
			{
				TexCoordsBufferList[bufferIndex].AllocAndUpload(mesh->TexCoords);
			}

			// triangle inputs
			OptixBuildInput& triangleInput = triangleInputList[bufferIndex];
			triangleInput = {};
			triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			// local variables such that pointer to device pointers can be created
			cudaVertexBufferList[bufferIndex] = VertexBufferList[bufferIndex].CudaPtr();
			cudaIndexBufferList[bufferIndex] = IndexBufferList[bufferIndex].CudaPtr();

			triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput.triangleArray.vertexStrideInBytes = sizeof(vec3f);
			triangleInput.triangleArray.numVertices = (int32_t)meshList[meshId]->Vertices.size();
			triangleInput.triangleArray.vertexBuffers = &cudaVertexBufferList[bufferIndex];

			triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput.triangleArray.indexStrideInBytes = sizeof(vec3i);
			triangleInput.triangleArray.numIndexTriplets = (int32_t)meshList[meshId]->Indices.size();
			triangleInput.triangleArray.indexBuffer = cudaIndexBufferList[bufferIndex];

			triangleInputFlagsList[bufferIndex] = 0;

			// currently only shader binding table entry, no per-primitive materials
			triangleInput.triangleArray.flags = &triangleInputFlagsList[bufferIndex];
			triangleInput.triangleArray.numSbtRecords = 1;
			triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
			triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
			triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
		}
	}

	// setup bottom level acceleration structure (BLAS)
	OptixAccelBuildOptions accelOptions = { };
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OptixResult result = optixAccelComputeMemoryUsage(OptixContext,
		&accelOptions, triangleInputList.data(),
		totalNumberMeshes, // number of build inputs
		&blasBufferSizes);
	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("could not compute memory usage for acceleration structure!");
	}

	// prepare compaction
	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.Alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.CudaPtr();

	// execute build

	CUDABuffer tempBuffer;
	tempBuffer.Alloc(blasBufferSizes.tempSizeInBytes);

	CUDABuffer outputBuffer;
	outputBuffer.Alloc(blasBufferSizes.outputSizeInBytes);

	result = optixAccelBuild(OptixContext,
		0,	//stream
		&accelOptions, triangleInputList.data(),
		(int32_t)totalNumberMeshes,
		tempBuffer.CudaPtr(), tempBuffer.Size_bytes,
		outputBuffer.CudaPtr(), outputBuffer.Size_bytes,
		&handle,
		&emitDesc,
		1 // num emitted properties
	);

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not build acceleration structure!");
	}

	SynchCuda("Error synchronizing CUDA after building acceleration structure!");

	// perform compaction
	uint64_t compactedSize;
	compactedSizeBuffer.Download(&compactedSize, 1);

	AccelerationStructureBuffer.Alloc(compactedSize);
	result = optixAccelCompact(OptixContext,
		0, //stream
		handle,
		AccelerationStructureBuffer.CudaPtr(),
		AccelerationStructureBuffer.Size_bytes,
		&handle);

	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not compact acceleration structure!");
	}

	SynchCuda("Error synchronizing CUDA after compacting acceleration structure!");

	// clean up
	outputBuffer.Free();
	tempBuffer.Free();
	compactedSizeBuffer.Free();

	std::cout << "finished building acceleration structure" << std::endl;

	return handle;
}

bool Renderer::HasSceneChanged() const
{
	for (std::shared_ptr<IDynamicElement> e : DynamicElements)
	{
		if (e->IsMarkedDirty())
		{
			return true;
		}
	}

	return false;
}

void Renderer::SynchCuda(const std::string& errorMsg /* = "" */)
{
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		const std::string errorString(cudaGetErrorString(error));
		std::string additionalMessage = "";
		if (errorMsg.compare("") != 0)
		{
			additionalMessage = "\nDeveloper message: " + errorMsg;
		}
		throw std::runtime_error("error synchronizing cuda: " + errorString + additionalMessage);
	}
}

uint32_t Renderer::GetNumberMeshesFromScene(const bool& includeVisibleProxies /* = true*/) const
{
	uint32_t numMeshes = 0;
	for (std::shared_ptr<Model> model : ModelList)
	{
		numMeshes += static_cast<uint32_t>(model->GetMeshList().size());
	}

	if (includeVisibleProxies)
	{
		for (std::shared_ptr<Light> light : LightList)
		{
			if (light->GetShowProxyMesh())
			{
				numMeshes++;
			}
		}
	}

	return numMeshes;
}

uint32_t Renderer::GetModelIndex(std::shared_ptr<Model> model) const
{
	for (size_t i = 0; i < ModelList.size(); i++)
	{
		if (model == ModelList[i])
		{
			return static_cast<uint32_t>(i);
		}
	}

	return -1;
}

uint32_t Renderer::GetMeshBufferIndex(std::shared_ptr<Model> model, const uint32_t meshIndex) const
{
	size_t index = GetModelIndex(model);

	return GetMeshBufferIndex(static_cast<uint32_t>(index), meshIndex);
}

uint32_t Renderer::GetMeshBufferIndex(const uint32_t& modelIndex, const uint32_t meshIndex) const
{
	uint32_t index = 0;
	for (size_t modelId = 0; modelId < ModelList.size(); modelId++)
	{
		if (modelId == modelIndex)
		{
			index += meshIndex;
			break;
		}
		else
		{
			index += static_cast<uint32_t>(ModelList[modelId]->GetMeshList().size());
		}
	}

	return index;
}

uint32_t Renderer::GetNumberTexturesFromScene() const
{
	uint32_t numTexs = 0;
	for (std::shared_ptr<Model> model : ModelList)
	{
		numTexs += static_cast<uint32_t>(model->GetTextureList().size());
	}
	return numTexs;
}

uint32_t Renderer::GetTextureBufferIndex(std::shared_ptr<Model> model, const uint32_t textureIndex) const
{
	size_t index = GetModelIndex(model);

	return GetTextureBufferIndex(static_cast<uint32_t>(index), textureIndex);
}

uint32_t Renderer::GetTextureBufferIndex(const uint32_t& modelIndex, const uint32_t textureIndex) const
{
	uint32_t index = 0;
	for (size_t modelId = 0; modelId < ModelList.size(); modelId++)
	{
		if (modelId == modelIndex)
		{
			index += textureIndex;
			break;
		}
		else
		{
			index += static_cast<uint32_t>(ModelList[modelId]->GetTextureList().size());
		}
	}

	return index;
}
