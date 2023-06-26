#include "cuda_runtime.h"

#include <optix_device.h>

#include "gdt/math/vec.h"
#include "gdt/random/random.h"

#include "LaunchParams.h"
#include "scene/Mesh.h"

typedef gdt::LCG<16> Random;

extern "C" __constant__ LaunchParams launchParams;

struct PerRayData
{
	Random Rand;
	vec3f PixelColor;
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
	const int32_t accumulationId = launchParams.FrameID;

	const CameraOptix& camera = launchParams.Camera;

	// per ray data
	PerRayData perRayData;
	perRayData.Rand.init(
		ix + accumulationId * launchParams.FramebufferSize.x,
		iy + accumulationId * launchParams.FramebufferSize.y);
	perRayData.PixelColor = vec3f(0.f);

	// values we store in the per ray data.
	// due to CUDA/OptiX restraints, we store two 32 bit values in a 64 bit pointer
	uint32_t u0, u1;
	packPointer(&perRayData, u0, u1);

	int32_t numPixelSamples = 16;

	// this is the final output pixel color, accumulating over all
	// the per ray data it receives
	vec3f pixelColor = 0.f;

	for (int32_t sampleId = 0; sampleId < numPixelSamples; sampleId++)
	{
		// get normalized coords (i.e. in [0, 1]^2)
		const vec2f normalizedScreenCoords(vec2f(ix + perRayData.Rand(), iy + perRayData.Rand())
			/ vec2f(launchParams.FramebufferSize));


		// generate ray direction
		vec3f dir = camera.LookingDirection;
		vec3f rayDir = normalize(dir
			+ (normalizedScreenCoords.x - 0.5f) * camera.Horizontal
			+ (normalizedScreenCoords.y - 0.5f) * camera.Vertical);

		optixTrace(launchParams.Traversable,
			camera.Position,
			rayDir,
			0.f,	// tmin -> the earliest a hit will be detected, similar to near clipping plane
			1e20f,	// tmax -> the latest a hit will be detected, similar to far clipping plane
			0.0f,	// ray time (has to be enabled in pipeline compile options, otherwise ignored)
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			RADIANCE_RAY_TYPE,
			RAY_TYPE_COUNT,
			RADIANCE_RAY_TYPE,
			u0, u1 // payload p0, p1 (, up to p7? up to 31?)
		);
		
		pixelColor += perRayData.PixelColor;
	}
	

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

	vec4f rgba(pixelColor / numPixelSamples, 1.f);

	// write / accumulate frame buffer
	const uint32_t fbIndex = ix + iy * launchParams.FramebufferSize.x;

	if (launchParams.FrameID > 0)
	{
		rgba += float(launchParams.FrameID)
			* vec4f(launchParams.FramebufferData[fbIndex]);
		rgba /= (launchParams.FrameID + 1.f);
	}
	// finally write to frame buffer
	launchParams.FramebufferData[fbIndex] = (float4)rgba;
}

extern "C" __global__ void __miss__shadow()
{
	// the shadow ray has not hit anything,
	// therefore the light hits the surface and fully lightens it
	vec3f& perRayData = *(vec3f*)getPerRayData<vec3f>();
	perRayData = vec3f(1.f);
}

extern "C" __global__ void __miss__radiance() 
{
	vec3f& perRayData = *(vec3f*)getPerRayData<vec3f>();

	// set to constant white as background colour
	perRayData = vec3f(1.f);
}

extern "C" __global__ void __closesthit__shadow()
{
	// nothing to do here, shadows will be handled in anyhit
}

extern "C" __global__ void __closesthit__radiance() 
{
	const MeshSbtData& meshData = *(const MeshSbtData*)optixGetSbtDataPointer();
	PerRayData& perRayData = *getPerRayData<PerRayData>();
	const LightOptix& light = launchParams.Light;

	// basic hit information
	const int32_t primitiveId = optixGetPrimitiveIndex();
	const vec3i index = meshData.Indices[primitiveId];
	const vec3f rayDir = optixGetWorldRayDirection();

	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	// read normal if one was provided, otherwise compute normal
	vec3f geometryNormal;
	vec3f surfaceNormal;

	const vec3f& v0 = meshData.Vertices[index.x];
	const vec3f& v1 = meshData.Vertices[index.y];
	const vec3f& v2 = meshData.Vertices[index.z];
	geometryNormal = normalize(cross(v1 - v0, v2 - v0));

	if (meshData.Normals)
	{
		surfaceNormal = (1.f - u - v) * meshData.Normals[index.x]
			+ u * meshData.Normals[index.y]
			+ v * meshData.Normals[index.z];
	}
	else
	{
		surfaceNormal = geometryNormal;
	}

	// make sure the ray dir and calulated geometry normal point in opposite directions
	if (dot(rayDir, geometryNormal) > 0.f)
	{
		geometryNormal = -geometryNormal;
	}

	// make sure the surface normal and geometry point into the same direction
	// TODO: why not just invert surface normal direction?
	if (dot(geometryNormal, surfaceNormal) < 0.f)
	{
		surfaceNormal -= 2.f * dot(geometryNormal, surfaceNormal) * geometryNormal;
	}
	surfaceNormal = normalize(surfaceNormal);

	// get colour from color variable or texture
	vec3f diffuseColor = meshData.DiffuseColor;
	if (meshData.HasTexture)
	{
		const vec2f texCoord = (1.f - u - v) * meshData.TexCoords[index.x]
			+ u * meshData.TexCoords[index.y]
			+ v * meshData.TexCoords[index.z];

		const vec4f diffuseTexColor = tex2D<float4>(meshData.Texture, texCoord.x, texCoord.y);
		diffuseColor *= (vec3f)diffuseTexColor;
	}

	// light the model based on ray / triangle angle (i.e. abs(dot(rayDir, normal)) )
	const float cosAlpha = 0.1f + .8f * fabsf(dot(rayDir, surfaceNormal));
	vec3f pixelColor = (.1f + .2f * cosAlpha) * diffuseColor;

	// compute, the shadow influence
	const vec3f surfacePosition =
		(1.f - u - v) * meshData.Vertices[index.x]
		+ u * meshData.Vertices[index.y]
		+ v * meshData.Vertices[index.z];

	const int32_t numLightSamples = 1;
	for (int32_t lightSampleId = 0; lightSampleId < numLightSamples; lightSampleId++)
	{
		// produce a random light sample on the area light
		// (i.e. a random position on the area it covers)
		const vec3f lightPos = launchParams.Light.Location
			+ perRayData.Rand() * launchParams.Light.Extent.x
			+ perRayData.Rand() * launchParams.Light.Extent.y;

		vec3f lightDir = lightPos - surfacePosition;
		float lightDistance = sqrtf(dot(lightDir, lightDir));
		lightDir = normalize(lightDir);

		const float NdotL = dot(surfaceNormal, lightDir);
		if (NdotL >= 0.f)
		{
			// trace a shadow ray
			vec3f lightVisibility = 0.f;
			uint32_t u0, u1;
			packPointer(&lightVisibility, u0, u1);

			optixTrace(launchParams.Traversable,
				surfacePosition + 1e-3f * geometryNormal,
				lightDir,
				1e-3f,							// tmin
				lightDistance * (1.f - 1e-3f),	// tmax
				0.f,							// ray time
				OptixVisibilityMask(255),
				// skip any/closest hit for shadow rays, and terminate on first intersection
				// the actual work is done in the miss shader, since that dertermines, if 
				// the light is visible from the given surface position
				OPTIX_RAY_FLAG_DISABLE_ANYHIT
				| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
				| OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
				SHADOW_RAY_TYPE,	// Shader binding table offset
				RAY_TYPE_COUNT,		// Sbt stride
				SHADOW_RAY_TYPE,	// Sbt miss program index
				u0, u1);

			pixelColor += lightVisibility * launchParams.Light.Power
				* diffuseColor * (NdotL / (lightDistance * lightDistance * numLightSamples));
		}

		
	}
	
	perRayData.PixelColor = pixelColor;
}

extern "C" __global__ void __anyhit__shadow()
{
	// if the shadow ray hits, the corresponding pixel is in shadow.
	// therefore, it should not (fullly) be shadded. 
	// the idea is, to assume a pixel is shaded and only if a shadow 
	// ray triggers the miss program (i.e. does not hit geometry)
	// will the pixel be fully lighted. this makes any code 
	// outside of the miss program obsolete
	//		-> tl;dr: nothing to do here
}

extern "C" __global__ void __anyhit__radiance() 
{// dummy functions for OptiX pipeline
}