
#pragma once

#include "gdt/math/vec.h"

using namespace gdt;

class Light
{
public:
	Light();

	Light(const vec3f& location);

	virtual void Tick(const float& deltaTime_Seconds);

	vec3f GetLocation() const;

	void SetLocation(const vec3f& location);

protected:
	vec3f Location = vec3f(0.f);
};

/**
* struct to provide easier access for OptiX (/CUDA) kernels
*/
struct LightOptix
{
	vec3f Location;
};