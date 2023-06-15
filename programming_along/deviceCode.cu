#include "cuda_runtime.h"

#include "LaunchParams.h"

extern "C" __constant__ LaunchParams launchParams;

extern "C" __global__ void __raygen__renderFrame()
{
	printf("doing ray generation in OptiX!");

	// TODO: write to framebuffer here
	launchParams.FramebufferData[0] = 0;
}

// dummy functions for OptiX pipeline
extern "C" __global__ void __miss__radiance() {}
extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}