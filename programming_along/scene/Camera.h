#pragma once

#include "Entity.h"
#include "IDynamicElement.h"

#include "gdt/math/vec.h"

#include "GLFW/glfw3.h"

using namespace gdt;

/**
* Simpler camera struct that can easily be used in OptiX
*/
struct CameraOptix
{
	vec3f Position;
	vec3f LookingDirection; // this is At - Eye
	float CosFovy;

	vec3f Horizontal;			// not the framebuffer width!
	vec3f Vertical;
};

/**
* Full camera class to control the camera on the host
*/
class Camera : public Entity, public IDynamicElement
{
	//TODO: override SetTransformationMatrix from Entity with Eye/At/Up setting
public:
	Camera(const vec3f& eye = vec3f(0.f, 0.f, 0.f), 
		const vec3f& at = vec3f(0.f, 0.f, -1.f), 
		const vec3f& up = vec3f(0.f, 1.f, 0.f),
		const float& cosFovy = 0.66f,
		const uint32_t& width = 1024, const uint32_t height = 768);
	~Camera();

	void Tick(const float& deltaTime_seconds);

	void Move(const float& deltaTime_seconds);

	void SetFramebufferSize(const vec2i& fbSize);

	void SetEye(const vec3f& eye);
	vec3f GetEye() const;

	void SetAt(const vec3f& at);
	vec3f GetAt() const;

	void SetUp(const vec3f& up);
	vec3f GetUp() const;

	void UpdateInitialEyeAtUp();

	void SetCosFovy(const float& cosFovy);
	float GetCosFovy() const;

	float GetSpeed() const;

	void KeyDown(const int32_t& key);
	void KeyUp(const int32_t& key);

	void SetMousePos(const vec2f& NormalizedMousePos);
	void MouseDown(const int32_t& button);
	void MousUp(const int32_t& button);

	CameraOptix GetOptixCamera() const;

	virtual bool IsDynamic() const override;

private:
	vec3f Eye;
	vec3f At;
	vec3f Up;

	vec3f InitialEye;
	vec3f InitialAt;
	vec3f InitialUp;

	float CosFovy;
	uint32_t Width;
	uint32_t Height;

	uint8_t KeyStatus[GLFW_KEY_LAST];
	vec2f LastMousePos_Normalized;
	vec2f CurrentMousePos_Normalized;
	uint8_t MouseStatus[8];
	float Speed = 100.0f;
};