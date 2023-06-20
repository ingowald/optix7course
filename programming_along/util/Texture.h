
#pragma once

#include <cstdint>

#include "gdt/math/vec.h"

using namespace gdt;

struct Texture2D
{
	~Texture2D()
	{
		if (Pixels)
		{
			delete[] Pixels;
		}
	}

	uint32_t* Pixels = nullptr;
	vec2i Resolution = vec2i(-1, -1);
};