
#include "Camera.h"

#include "GLFW/glfw3.h"

Camera::Camera(const vec3f& eye /* = vec3f(0.f, 0.f, 0.f)*/, 
	const vec3f& at /* = vec3f(0.f, 0.f, -1.f)*/, 
	const vec3f& up /* = vec3f(0.f, 1.f, 0.f)*/,
	const float& fovy /* = 0.66f*/)
{
	Eye = eye;
	At = at;
	Up = up;
	Fovy = fovy;

	std::fill_n(KeyStatus, 256, 0);
}

Camera::~Camera()
{

}

void Camera::Tick(const float& deltaTime_seconds)
{
	Move(deltaTime_seconds);
}

void Camera::Move(const float& deltaTime_seconds)
{
	if (KeyStatus[GLFW_KEY_A])
	{
		Eye.x -= Speed * deltaTime_seconds;
	}
	if (KeyStatus[GLFW_KEY_D])
	{
		Eye.x += Speed * deltaTime_seconds;
	}
	if (KeyStatus[GLFW_KEY_W])
	{
		Eye.z += Speed * deltaTime_seconds;
	}
	if (KeyStatus[GLFW_KEY_S])
	{
		Eye.z -= Speed * deltaTime_seconds;
	}
	if (KeyStatus[GLFW_KEY_E])
	{
		Eye.y += Speed * deltaTime_seconds;
	}
	if (KeyStatus[GLFW_KEY_Q])
	{
		Eye.y -= Speed * deltaTime_seconds;
	}

	if (KeyStatus[GLFW_KEY_0])
	{
		Eye = vec3f(0.f, 0.f, 0.f);
	}
}

void Camera::SetEye(const vec3f& eye)
{
	Eye = eye;
}

vec3f Camera::GetEye() const
{
	return Eye;
}

void Camera::SetAt(const vec3f& at)
{
	At = at;
}

vec3f Camera::GetAt() const
{
	return At;
}

void Camera::SetUp(const vec3f& up)
{
	Up = up;
}

vec3f Camera::GetUp() const
{
	return Up;
}

void Camera::SetFovy(const float& fovy)
{
	Fovy = fovy;
}

float Camera::GetFovy() const
{
	return Fovy;
}

float Camera::GetSpeed() const
{
	return Speed;
}

void Camera::KeyDown(const int32_t& key)
{
	KeyStatus[key] = 1;
}

void Camera::KeyUp(const int32_t& key)
{
	KeyStatus[key] = 0;
}