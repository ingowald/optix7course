
#include "QuadLight.h"

QuadLight::QuadLight(const vec3f& origin, const vec3f& power /*= vec3f(1000000.f)*/, 
	const vec2f& extent /*= vec2f(200.f, 200.f)*/)
	: Light(origin, power), Extent(extent)
{
	Name = "QuadLight";
}

void QuadLight::Tick(const float& deltaTime_seconds)
{
	DirtyBit = false;
	if (DynamicEnabled)
	{
		TotalTime_seconds += deltaTime_seconds;
		RotationOffset.x = sin(TotalTime_seconds * RotationSpeed) * RotationRadius;
		RotationOffset.z = cos(TotalTime_seconds * RotationSpeed) * RotationRadius;
		DirtyBit = true;
	}
}

vec2f QuadLight::GetExtent() const
{
	return Extent;
}

void QuadLight::SetExtent(const vec2f& extent)
{
	Extent = extent;
}

float QuadLight::GetRotationSpeed() const
{
	return RotationSpeed;
}

void QuadLight::SetRotationSpeed(const float& speed)
{
	RotationSpeed = speed;
}

float QuadLight::GetRotationRadius() const
{
	return RotationRadius;
}

void QuadLight::SetRotationRadius(const float& radius)
{
	RotationRadius = radius;
}

std::shared_ptr<LightOptix> QuadLight::GetOptixLight() const
{
	std::shared_ptr<QuadLightOptix> optixLight = std::make_shared<QuadLightOptix>();
	optixLight->Location = Location;
	if (DynamicEnabled)
	{
		optixLight->Location += RotationOffset;
	}
	optixLight->Extent = Extent;
	optixLight->Power = Power;

	return optixLight;
}

bool QuadLight::IsDynamic() const
{
	return true;
}
