#pragma once

#include "gdt/math/vec.h"

using namespace gdt;

class Camera
{
	Camera(const vec3f& eye = vec3f(0.f, 0.f, 0.f), 
		const vec3f& at = vec3f(0.f, 0.f, -1.f), 
		const vec3f& up = vec3f(0.f, 1.f, 0.f),
		const float& fovy = 0.66f);
	~Camera();

	void SetEye(const vec3f& eye);
	vec3f GetEye() const;

	void SetAt(const vec3f& at);
	vec3f GetAt() const;

	void SetUp(const vec3f& up);
	vec3f GetUp() const;

	void SetFovy(const float& fovy);
	float GetFovy() const;

private:
	vec3f Eye;
	vec3f At;
	vec3f Up;

	float Fovy;
};