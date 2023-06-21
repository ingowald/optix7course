
#pragma once

#include "Light.h"

class RotatingLight : public Light
{
public:
	RotatingLight(const vec3f& location, float rotationSpeed = 1.f, float rotationRadius = 1.f);

	virtual void Tick(const float& deltaTime_seconds) override;

	float GetRotationSpeed() const;
	void SetRotationSpeed(const float& rotationSpeed);

	float GetRotationRadius() const;
	void SetRotationRadius(const float& rotationRadius);

protected:

	float RotationSpeed = 1.f;
	float RotationRadius = 1.f;
};