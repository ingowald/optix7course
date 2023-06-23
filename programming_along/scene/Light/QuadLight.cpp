
#include "QuadLight.h"

QuadLight::QuadLight(const vec3f& origin, const vec3f& power /*= vec3f(1000000.f)*/, 
	const vec2f& extent /*= vec2f(200.f, 200.f)*/)
	: Light(origin, power), Extent(extent)
{
}

vec2f QuadLight::GetExtent() const
{
	return Extent;
}

void QuadLight::SetExtent(const vec2f& extent)
{
	Extent = extent;
}

std::shared_ptr<LightOptix> QuadLight::GetOptixLight() const
{
	std::shared_ptr<QuadLightOptix> optixLight = std::make_shared<QuadLightOptix>();
	optixLight->Location = Location;
	optixLight->Extent = Extent;
	optixLight->Power = Power;

	return optixLight;
}