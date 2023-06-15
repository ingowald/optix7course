

#include "Camera.h"

Camera::Camera(const vec3f& eye /* = vec3f(0.f, 0.f, 0.f)*/, 
	const vec3f& at /* = vec3f(0.f, 0.f, -1.f)*/, 
	const vec3f& up /* = vec3f(0.f, 1.f, 0.f)*/,
	const float& fovy /* = 0.66f*/)
{
	Eye = eye;
	At = at;
	Up = up;
	Fovy = fovy;
}

Camera::~Camera()
{

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