#include "cuda_runtime.h"

#include <optix_device.h>

#include "gdt/math/vec.h"

#include "LaunchParams.h"
#include "util/Mesh.h"

extern "C" __constant__ LaunchParams launchParams;

// simple ray type
enum { 
	SURFACE_RAY_TYPE = 0,
	RAY_TYPE_COUNT
};

static __forceinline__ __device__
void packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{	
	// write the first half of the 64 bit value into i0, the second into i1 (both 32 bit)
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__
void* unpackPointer(uint32_t i0, uint32_t i1)
{
	// store i0 and i1 (both 32 bit) in the same 64 bit variable
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	// return the new pointer as a void pointer
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

template<typename T>
static __forceinline__ __device__ T* getPerRayData()
{
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

extern "C" __global__ void __raygen__renderFrame()
{
	const int32_t ix = optixGetLaunchIndex().x;
	const int32_t iy = optixGetLaunchIndex().y;

	const CameraOptix& camera = launchParams.Camera;

	// per ray data
	vec3f pixelColorPrd = vec3f(0.f);

	// values we store in the per ray data.
	// due to CUDA/OptiX restraints, we store two 32 bit values in a 64 bit pointer
	uint32_t u0, u1;
	packPointer(&pixelColorPrd, u0, u1);

	// get normalized coords (i.e. in [0, 1]^2)
	const vec2f normalizedScreenCoords(vec2f(ix + .5f, iy + .5f)
		/ vec2f(launchParams.FramebufferSize));


	// generate ray direction
	vec3f dir = camera.LookingDirection;
	vec3f rayDir = normalize(dir
		+ (normalizedScreenCoords.x - 0.5f) * camera.Horizontal
		+ (normalizedScreenCoords.y - 0.5f) * camera.Vertical);

	//bool newLine = false;
	//if ((ix == (launchParams.FramebufferSize.x*0.5))
	//	&& (iy == (launchParams.FramebufferSize.y*0.5)))
	//{
	//	printf("ix: %i, iy: %i\n"
	//		"framebuffer size : (%i, %i)\n"
	//		"normSreenCoord: (%.2f, %.2f)\n"
	//		"eye: (%.2f, %.2f, %.2f)\n"
	//		"at:  (%.2f, %.2f, %.2f)\n"
	//		"rayDir: (% .2f, % .2f, % .2f)\n\n", 
	//		ix, iy, 
	//		launchParams.FramebufferSize.x, launchParams.FramebufferSize.y, 
	//		normalizedScreenCoords.x, normalizedScreenCoords.y, 
	//		camera.Eye.x, camera.Eye.y, camera.Eye.z,
	//		camera.At.x, camera.At.y, camera.At.z,
	//		rayDir.x, rayDir.y, rayDir.z);
	//	newLine = true;
	//}

	optixTrace(launchParams.Traversable,
		camera.Position,
		rayDir,
		0.f,	// tmin -> the earliest a hit will be detected, similar to near clipping plane
		1e20f,	// tmax -> the latest a hit will be detected, similar to far clipping plane
		0.0f,	// ray time (has to be enabled in pipeline compile options, otherwise ignored)
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		SURFACE_RAY_TYPE,
		RAY_TYPE_COUNT,
		SURFACE_RAY_TYPE,
		u0, u1 // payload p0, p1 (, up to p7? up to 31?)
	);

	// optixTrace starts the actual ray trace. 
	// depending on if something is hit, either the closest hit program
	// or the miss program is executed (anyhit is disabled in this example)
	// since u0, u1 is the packed per ray data and set as a payload to trace
	// and both the closest hit and the miss program write to the per ray data,
	// we can assume that pixelColorPrd has gotten values through the payload
	// writing of the two programs
	// i.e:
	// the ray was traced and returned a value here
	// so we can write the result to the frame buffer

	const int32_t r = int32_t(255.99 * pixelColorPrd.x);
	const int32_t g = int32_t(255.99 * pixelColorPrd.y);
	const int32_t b = int32_t(255.99 * pixelColorPrd.z);

	// convert to 32-bit rgba value, with alpha explicitly 0xff
	const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

	const uint32_t fbIndex = ix + iy * launchParams.FramebufferSize.x;

	// finally write to frame buffer
	launchParams.FramebufferData[fbIndex] = rgba;
}


extern "C" __global__ void __miss__radiance() 
{
	vec3f& perRayData = *(vec3f*)getPerRayData<vec3f>();

	// set to constant white as background colour
	perRayData = vec3f(1.f);
}

extern "C" __global__ void __closesthit__radiance() 
{
	const MeshSbtData& meshData = *(const MeshSbtData*)optixGetSbtDataPointer();

	// compute a normal
	// TODO: this should be done offline when creating the mesh and put into a buffer
	const int32_t primitiveId = optixGetPrimitiveIndex();
	const vec3i index = meshData.Indices[primitiveId];
	const vec3f& v0 = meshData.Vertices[index.x];
	const vec3f& v1 = meshData.Vertices[index.y];
	const vec3f& v2 = meshData.Vertices[index.z];
	const vec3f& normal = normalize(cross(v1 - v0, v2 - v0));

	const vec3f rayDir = optixGetWorldRayDirection();
	
	// shade the model based on ray / triangle angle (i.e. abs(dot(rayDir, normal)) )
	const float cosAlpha = 0.2f + .8f * fabsf(dot(rayDir, normal));

	vec3f& perRayData = *(vec3f*)getPerRayData<vec3f>();
	perRayData = cosAlpha * meshData.Color;
}

// dummy functions for OptiX pipeline
extern "C" __global__ void __anyhit__radiance() {}