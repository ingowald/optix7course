#include "cuda_runtime.h"

#include <optix_device.h>

#include "LaunchParams.h"

extern "C" __constant__ LaunchParams launchParams;

extern "C" __global__ void __raygen__renderFrame()
{
	if (launchParams.FrameID == 0
		&& optixGetLaunchIndex().x == 0
		&& optixGetLaunchIndex().y == 0)
	{
		printf("############################\n");
		printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
			launchParams.FramebufferSize.x,
			launchParams.FramebufferSize.y);
		printf("############################\n");
	}

	const int32_t ix = optixGetLaunchIndex().x;
	const int32_t iy = optixGetLaunchIndex().y;

	const int32_t r = (ix % 256);
	const int32_t g = (iy % 256);
	const int32_t b = ((ix + iy) % 256);

	// convert to 32-bit rgba value, alpha is is explicitly set to 0xff for stb_image_write
	const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

	const uint32_t fbIndex = ix + iy * launchParams.FramebufferSize.x;

	// finally write to frame buffer
	launchParams.FramebufferData[fbIndex] = rgba;
}

// dummy functions for OptiX pipeline
extern "C" __global__ void __miss__radiance() {}
extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}