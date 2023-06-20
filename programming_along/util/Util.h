

#pragma once

#include <string.h>

#include "gdt/math/vec.h"

using namespace gdt;

class Util
{
public:
	static std::string VecToString(const vec2f& vec);
	static std::string VecToString(const vec3f& vec);
	static std::string VecToString(const vec3i& vec);

};