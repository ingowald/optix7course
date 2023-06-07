#pragma once

#include "gdt/math/vec.h"

using namespace gdt;

struct LaunchParams
{
	vec2i FramebufferSize;
	uint32_t* FramebufferData;
};