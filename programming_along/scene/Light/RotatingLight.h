
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

	virtual std::shared_ptr<LightOptix> GetOptixLight() const override;

	virtual bool IsDynamic() const override;

protected:

	float TotalTime_seconds = 0.f;
	vec3f RotationOffset = vec3f(0.f);
	float RotationSpeed = 1.f;
	float RotationRadius = 1.f;
};