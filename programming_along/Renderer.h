#pragma once

#include <vector>

#include "cuda_runtime.h"
#include "cuda.h"

#include "util/CUDABuffer.h"
#include "scene/Camera.h"

#include "optix_types.h"

#include "gdt/math/vec.h"

#include "LaunchParams.h"

class IDynamicElement;
class Model;
struct Mesh;
class Light;

class Renderer
{
public:
	Renderer();
	~Renderer();

	static void OptixLogCallback(uint32_t level, const char* tag, const char* message, void* cbdata);

	/**
	* Initializes anything that could not be done in the ctor,
	* e.g. building the shader binding table
	*/
	void Init();

	void Tick(const float& deltaTime_seconds);

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
	void DownloadPixels(vec4f pixels[]);

	Camera* GetCameraPtr();

	void InitializeCamera(const vec3f& eye, const vec3f& at, const vec3f& up);

	void AddMesh(std::shared_ptr<Mesh> mesh);

	void AddModel(std::shared_ptr<Model> model);

	void AddLight(std::shared_ptr<Light> light);

	bool GetDynamicLightsMovementsEnabled() const;
	void EnableDynamicLightsMovements(const bool& enabled);
	void ToggleDynamicLightsMovement();

	bool GetDenoiserEnabled() const;
	void SetDenoiserEnabled(const bool& enabled);
	void ToggleDenoiserEnabled();

	bool GetAccumulationEnabled() const;
	void SetAccumulationEnabled(const bool& enabled);
	void ToggleAccumulationEnabled();

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
	* Creates all necessary textures for all meshes from all models for OptiX
	*/
	void CreateTextures();

	/**
	* Builds the shader binding table
	*/
	void BuildShaderBindingTable();

	/**
	* Builds an acceleration structure based on the mesh list
	*/
	OptixTraversableHandle BuildAccelerationStructure();

	/**
	* Checks if any of the dynamic scene elements is marked as dirty.
	* @returns true if any element is marked dirty, false otherwise
	*/
	bool HasSceneChanged() const;

	void SynchCuda(const std::string& errorMsg = "");

	uint32_t GetNumberMeshesFromScene(const bool& includeVisibleProxies = true) const;

	/**
	* Finds the given model in the ModelList and returns the index
	*/
	uint32_t GetModelIndex(std::shared_ptr<Model> model) const;

	/**
	* Since there is a ModelList and a MeshList per Model,
	* but the CUDA Vertex and Index buffers are one dimensional,
	* we need to calculate the Index for a given mesh from a given model
	*/
	uint32_t GetMeshBufferIndex(std::shared_ptr<Model> model, const uint32_t meshIndex) const;

	/**
	* Since there is a ModelList and a MeshList per Model,
	* but the CUDA Vertex and Index buffers are one dimensional,
	* we need to calculate the Index for a given mesh from a given model
	*/
	uint32_t GetMeshBufferIndex(const uint32_t& modelIndex, const uint32_t meshIndex) const;

	uint32_t GetNumberTexturesFromScene() const;

	/**
	* Since there is a ModelList and a TextureList per Model,
	* but CUDA Texture buffers are one dimensional
	* we need to calculate the Index for a given texture from a given model
	*/
	uint32_t GetTextureBufferIndex(std::shared_ptr<Model> model, const uint32_t textureIndex) const;

	/**
	* Since there is a ModelList and a TextureList per Model,
	* but CUDA Texture buffers are one dimensional
	* we need to calculate the Index for a given texture from a given model
	*/
	uint32_t GetTextureBufferIndex(const uint32_t& modelIndex, const uint32_t textureIndex) const;

public:

protected:
	/** Basic setup of size */
	LaunchParams Params;
	/** Contents of the basic setup transferred to GPU */
	CUDABuffer ParamsBuffer;
	/** Framebuffer contents */
	CUDABuffer ColorBuffer;
	/** Denoised framebuffer contents */
	CUDABuffer DenoisedBuffer;

	/** Scene */
	std::shared_ptr<Camera> SceneCamera;

	std::vector<std::shared_ptr<Model>> ModelList;
	CUDABuffer AccelerationStructureBuffer;

	bool DynamicLightsMovementsEnabled = true;
	std::vector<std::shared_ptr<Light>> LightList;

	std::vector<std::shared_ptr<IDynamicElement>> DynamicElements;

	std::vector<CUDABuffer> VertexBufferList;
	std::vector<CUDABuffer> NormalBufferList;
	std::vector<CUDABuffer> IndexBufferList;
	std::vector<CUDABuffer> TexCoordsBufferList;

	std::vector<cudaArray_t> TextureArrays;
	std::vector<cudaTextureObject_t> TextureObjects;

	bool IsInitialized = false;
	bool AccumulatedDenoiseImages = false;
	bool DenoiserEnabled = true;

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

	// OptixDenoiser is a OptixDenoiser_t* (i.e. a pointer type)
	// therefore no need to make a (new) pointer out of it for the member
	OptixDenoiser Denoiser = nullptr;
	CUDABuffer DenoiserScratch;
	CUDABuffer DenoiserState;
};