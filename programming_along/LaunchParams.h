#pragma once

#include "gdt/math/vec.h"

#include "scene/Camera.h"

using namespace gdt;

struct LaunchParams
{
	uint32_t FrameID{ 0 };
	vec2i FramebufferSize;
	uint32_t* FramebufferData = nullptr;

	CameraOptix Camera;

	// the scene(?)
	OptixTraversableHandle Traversable;
};