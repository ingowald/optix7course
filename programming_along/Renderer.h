#pragma once

#include <vector>

#include "cuda_runtime.h"
#include "cuda.h"

#include "util/CUDABuffer.h"
#include "scene/Camera.h"

#include "optix_types.h"

#include "gdt/math/vec.h"

#include "LaunchParams.h"

class Model;
struct Mesh;

class Renderer
{
public:
	Renderer();
	~Renderer();

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
	void DownloadPixels(uint32_t pixels[]);

	Camera* GetCameraPtr();

	void InitializeCamera(const vec3f& eye, const vec3f& at, const vec3f& up);

	void AddMesh(const Mesh& mesh);

	void AddModel(const Model& model);

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

	void SynchCuda(const std::string& errorMsg = "");

	uint32_t GetNumberMeshesFromScene() const;

	/**
	* Since there is a ModelList and a MeshList per Model,
	* but the CUDA Vertex and Index buffers are one dimensional,
	* we need to calculate the Index for a given mesh from a given model
	*/
	uint32_t GetMeshBufferIndex(const Model& model, const uint32_t meshIndex) const;

	/**
	* Since there is a ModelList and a MeshList per Model,
	* but the CUDA Vertex and Index buffers are one dimensional,
	* we need to calculate the Index for a given mesh from a given model
	*/
	uint32_t GetMeshBufferIndex(const uint32_t& modelIndex, const uint32_t meshIndex) const;

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

	std::vector<Model> ModelList;
	CUDABuffer AccelerationStructureBuffer;

	std::vector<CUDABuffer> VertexBufferList;
	std::vector<CUDABuffer> NormalBufferList;
	std::vector<CUDABuffer> IndexBufferList;
	std::vector<CUDABuffer> TexCoordsBufferList;

	std::vector<cudaArray_t> TextureArrays;
	std::vector<cudaTextureObject_t> TextureObjects;

	bool IsInitialized = false;

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