#pragma once

#include "gdt/math/vec.h"

#include "scene/Camera.h"
#include "scene/Light/QuadLight.h"

using namespace gdt;

// simple ray type
enum {
	RADIANCE_RAY_TYPE = 0,
	SHADOW_RAY_TYPE,
	RAY_TYPE_COUNT
};

struct LaunchParams
{
	uint32_t FrameID{ 0 };
	vec2i FramebufferSize;
	uint32_t* FramebufferData = nullptr;

	CameraOptix Camera;

	// TODO: support multiple light sources
	QuadLightOptix Light;

	// the scene(?)
	OptixTraversableHandle Traversable;
};