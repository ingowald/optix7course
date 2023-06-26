
#include "RotatingLight.h"

RotatingLight::RotatingLight(const vec3f& location, float rotationSpeed /* = 1.f*/, float rotationRadius /* = 1.f */ )
{
	Location = location;
	RotationSpeed = rotationSpeed;
	RotationRadius = rotationRadius;

	Name = "RotatingLight";
}

void RotatingLight::Tick(const float& deltaTime_seconds)
{
	if (DynamicEnabled)
	{
		TotalTime_seconds += deltaTime_seconds;
		RotationOffset.x = sin(TotalTime_seconds * RotationSpeed) * RotationRadius;
		RotationOffset.z = cos(TotalTime_seconds * RotationSpeed) * RotationRadius;
		DirtyBit = true;
	}
}

float RotatingLight::GetRotationSpeed() const
{
	return RotationSpeed;
}

void RotatingLight::SetRotationSpeed(const float& rotationSpeed)
{
	RotationSpeed = rotationSpeed;
}

float RotatingLight::GetRotationRadius() const
{
	return RotationRadius;
}

void RotatingLight::SetRotationRadius(const float& rotationRadius)
{
	RotationRadius = rotationRadius;
}

std::shared_ptr<LightOptix> RotatingLight::GetOptixLight() const
{
	std::shared_ptr<LightOptix> l = this->GetOptixLight();
	l->Location += RotationOffset;
	return l;
}

bool RotatingLight::IsDynamic() const
{
	return true;
}
