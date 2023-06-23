
#pragma once

#include "Light.h"

struct QuadLightOptix : LightOptix
{
	vec2f Extent;
};

/**
* An area light source in the shape of a quad. The position is set via the origin,
* the size is determined by the extent. Per default in the xz plane.
*/
class QuadLight : public Light 
{
public:
	QuadLight(const vec3f& origin, const vec2f& extent);

	vec2f GetExtent() const;
	void SetExtent(const vec2f& extent);

protected:
	vec2f Extent;
};