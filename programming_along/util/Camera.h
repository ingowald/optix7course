#pragma once

#include "gdt/math/vec.h"

using namespace osc;

class Camera
{
	Camera(const vec3f& eye = vec3f(0.f, 0.f, 0.f), const vec3f& at, const vec3f& up)
	~Camera();
};