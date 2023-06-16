#pragma once

#include "gdt/math/vec.h"

using namespace gdt;

/**
* Simpler camera struct that can easily be used in OptiX
*/
struct CameraOptix
{
	vec3f Eye;
	vec3f At;
	vec3f Up;
	float Fovy;

	uint32_t Width;
	uint32_t Height;
};

/**
* Full camera class to control the camera on the host
*/
class Camera
{
public:
	Camera(const vec3f& eye = vec3f(0.f, 0.f, 0.f), 
		const vec3f& at = vec3f(0.f, 0.f, -1.f), 
		const vec3f& up = vec3f(0.f, 1.f, 0.f),
		const float& fovy = 0.66f,
		const uint32_t& width = 1024, const uint32_t height = 768);
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

	void SetMousePos(const vec2f& NormalizedMousePos);
	void MouseDown(const int32_t& button);
	void MousUp(const int32_t& button);

	CameraOptix GetOptixCamera() const;

private:
	vec3f Eye;
	vec3f At;
	vec3f Up;

	vec3f InitialEye;
	vec3f InitialAt;
	vec3f InitialUp;

	float Fovy;
	uint32_t Width;
	uint32_t Height;

	uint8_t KeyStatus[256];
	vec2f LastMousePos_Normalized;
	vec2f CurrentMousePos_Normalized;
	uint8_t MouseStatus[8];
	float Speed = 1.0f;
};