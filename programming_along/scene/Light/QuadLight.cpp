
#include "QuadLight.h"

QuadLight::QuadLight(const vec3f& origin, const vec2f& extent) : Light(origin)
{
	Extent = extent;
}

vec2f QuadLight::GetExtent() const
{
	return Extent;
}

void QuadLight::SetExtent(const vec2f& extent)
{
	Extent = extent;
}