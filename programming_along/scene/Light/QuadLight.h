
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
	QuadLight(const vec3f& origin, const vec3f& power = vec3f(1000000.f), 
		const vec2f& extent = vec2f(200.f, 200.f));

	vec2f GetExtent() const;
	void SetExtent(const vec2f& extent);

	virtual std::shared_ptr<LightOptix> GetOptixLight() const override;

protected:
	vec2f Extent;
};