#pragma once

#include "gdt/math/vec.h"

using namespace gdt;



class Camera
{
public:
	Camera(const vec3f& eye = vec3f(0.f, 0.f, 0.f), 
		const vec3f& at = vec3f(0.f, 0.f, -1.f), 
		const vec3f& up = vec3f(0.f, 1.f, 0.f),
		const float& fovy = 0.66f);
	~Camera();

	void Tick(const float& deltaTime_seconds);

	void Move(const float& deltaTime_seconds);

	void SetEye(const vec3f& eye);
	vec3f GetEye() const;

	void SetAt(const vec3f& at);
	vec3f GetAt() const;

	void SetUp(const vec3f& up);
	vec3f GetUp() const;

	void SetFovy(const float& fovy);
	float GetFovy() const;

	float GetSpeed() const;

	void KeyDown(const int32_t& key);
	void KeyUp(const int32_t& key);

private:
	vec3f Eye;
	vec3f At;
	vec3f Up;

	float Fovy;

	uint8_t KeyStatus[256];
	float Speed = 1.0f;
};