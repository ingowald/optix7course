#include "cuda_runtime.h"

#include <optix_device.h>

#include "LaunchParams.h"

extern "C" __constant__ LaunchParams launchParams;

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
	vec3f rayDir = normalize(camera.At
		+ (normalizedScreenCoords.x - 0.5f) * camera.Width
		+ (normalizedScreenCoords.y - 0.5f) * camera.Height);

	const uint32_t fbIndex = ix + iy * launchParams.FramebufferSize.x;

	// finally write to frame buffer
	launchParams.FramebufferData[fbIndex] = 1;
}

// dummy functions for OptiX pipeline
extern "C" __global__ void __miss__radiance() {}
extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}