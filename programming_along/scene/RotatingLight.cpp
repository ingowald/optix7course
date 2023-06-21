
#include "RotatingLight.h"

RotatingLight::RotatingLight(const vec3f& location, float rotationSpeed = 1.f, float rotationRadius = 1.f)
{
	Location = location;
	RotationSpeed = rotationSpeed;
	RotationRadius = rotationRadius;
}

void RotatingLight::Tick(const float& deltaTime_seconds)
{
	TotalTime += deltaTime_seconds;

	vec3f location = Location;
	location.x += sin(deltaTime_seconds * RotationSpeed) * RotationRadius;
	location.z += cos(deltaTime_seconds * RotationSpeed) * RotationRadius;
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