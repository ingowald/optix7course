
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

	virtual void Tick(const float& deltaTime_seconds) override;

	vec2f GetExtent() const;
	void SetExtent(const vec2f& extent);

	float GetRotationSpeed() const;
	void SetRotationSpeed(const float& speed);

	float GetRotationRadius() const;
	void SetRotationRadius(const float& radius);

	virtual std::shared_ptr<LightOptix> GetOptixLight() const override;

	virtual bool IsDynamic() const override;

protected:
	vec2f Extent;

	float RotationSpeed = 1.f;
	float RotationRadius = 1.f;
	vec3f RotationOffset = vec3f(0);
	float TotalTime_seconds = 0.f;
};