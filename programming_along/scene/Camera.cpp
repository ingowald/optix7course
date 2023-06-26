
#include "Camera.h"

#include "GLFW/glfw3.h"

#include "gdt/math/Quaternion.h"

#include "../util/Util.h"

Camera::Camera(const vec3f& eye /* = vec3f(0.f, 0.f, 0.f)*/, 
	const vec3f& at /* = vec3f(0.f, 0.f, -1.f)*/, 
	const vec3f& up /* = vec3f(0.f, 1.f, 0.f)*/,
	const float& cosFovy /* = 0.66f*/,
	const uint32_t& width /* = 1024 */, const uint32_t height /* = 768 */)
{
	Eye = eye;
	At = at;
	Up = up;

	InitialEye = eye;
	InitialAt = at;
	InitialUp = up;

	CosFovy = cosFovy;
	Width = width;
	Height = height;

	std::fill_n(KeyStatus, 256, 0);
	std::fill_n(MouseStatus, 8, 0);

	Name = "Camera";
}

Camera::~Camera()
{

}

void Camera::Tick(const float& deltaTime_seconds)
{
	// we track movement per frame for the accumulation of frames
	// -> if the camera has moved, the accumulation has to be reset
	// it is more efficient, to reset the bit here, rather than query and
	// reset the dirty bit of every dynamic element from the renderer
	// seeing as we would only need changes that happend during the current frame
	SetDirtyBitConsumed();
	Move(deltaTime_seconds);
}

void Camera::Move(const float& deltaTime_seconds)
{
	vec3f forward = normalize(At - Eye);
	vec3f right = cross(forward, Up);

	vec3f lastEye = Eye;
	vec3f lastAt = At;
	vec3f lastUp = Up;

	float adjustedSpeed = Speed;

	if (KeyStatus[GLFW_KEY_LEFT_SHIFT])
	{
		adjustedSpeed *= 10;
	}

	if (KeyStatus[GLFW_KEY_A])
	{
		Eye -= adjustedSpeed * deltaTime_seconds * right;
		At = Eye + forward;
	}
	if (KeyStatus[GLFW_KEY_D])
	{
		Eye += adjustedSpeed * deltaTime_seconds * right;
		At = Eye + forward;
	}
	if (KeyStatus[GLFW_KEY_W])
	{
		Eye += adjustedSpeed * deltaTime_seconds * forward;
		At = Eye + forward;
	}
	if (KeyStatus[GLFW_KEY_S])
	{
		Eye -= adjustedSpeed * deltaTime_seconds * forward;
		At = Eye + forward;
	}
	if (KeyStatus[GLFW_KEY_E])
	{
		Eye += adjustedSpeed * deltaTime_seconds * Up;
		At = Eye + forward;
	}
	if (KeyStatus[GLFW_KEY_Q])
	{
		Eye -= adjustedSpeed * deltaTime_seconds * Up;
		At = Eye + forward;
	}

	// safety net
	if (std::isnan(Eye.x) || std::isnan(Eye.y) || std::isnan(Eye.z))
	{
		Eye = lastEye;
	}

	// print camera setup for easier initial setup
	if (KeyStatus[GLFW_KEY_P])
	{
		std::cout << "Eye: "
			<< Util::VecToString(Eye)
			<< std::endl;
		std::cout << "At: "
			<< Util::VecToString(At)
			<< std::endl;
		std::cout << "Up: "
			<< Util::VecToString(Up)
			<< std::endl;
	}

	// mouse action
	if (MouseStatus[GLFW_MOUSE_BUTTON_LEFT])
	{
		vec2f way = CurrentMousePos_Normalized - LastMousePos_Normalized;
		float distance = sqrtf(way.x * way.x + way.y * way.y);
		float distanceX = sqrtf(way.x * way.x);
		float distanceY = sqrtf(way.y * way.y);

		if (distance == 0)
		{
			return;
		}

		const int8_t signX = CurrentMousePos_Normalized.x > LastMousePos_Normalized.x ? 1 : -1;
		const int8_t signY = CurrentMousePos_Normalized.y > LastMousePos_Normalized.y ? 1 : -1;

		float angleX = signX * asin(distanceX);
		float angleY = signY * asin(distanceY);

		Quaternion3f quatX = Quaternion3f::rotate(Up, angleX);
		forward = quatX * forward;

		Quaternion3f quatY = Quaternion3f::rotate(right, angleY);
		forward = quatY * forward;

		At = Eye + forward;
	}
	if (KeyStatus[GLFW_KEY_0])
	{
		Eye = vec3f(0.f, 0.f, 0.f);
	}

	// if setting to a certain position and orientation,
	// ignore any forward calculations!
	if (KeyStatus[GLFW_KEY_9])
	{
		Eye = InitialEye;
		At = InitialAt;
		Up = InitialUp;
	}
	if (KeyStatus[GLFW_KEY_8])
	{
		Eye = vec3f(-40.406f, 48.918f, 57.771f);
		At = vec3f(-39.874f, 48.441f, 57.071f);
		Up = vec3f(0.f, 1.f, 0.f);
	}

	LastMousePos_Normalized = CurrentMousePos_Normalized;

	if (lastEye != Eye || lastAt != At || lastUp != Up)
	{
		DirtyBit = true;
	}
}

void Camera::SetFramebufferSize(const vec2i& fbSize)
{
	Width = fbSize.x;
	Height = fbSize.y;
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

void Camera::UpdateInitialEyeAtUp()
{
	InitialEye = Eye;
	InitialAt = At;
	InitialUp = Up;
}

void Camera::SetCosFovy(const float& cosFovy)
{
	CosFovy = cosFovy;
}

float Camera::GetCosFovy() const
{
	return CosFovy;
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

void Camera::SetMousePos(const vec2f& NormalizedMousePos)
{
	LastMousePos_Normalized = CurrentMousePos_Normalized;
	CurrentMousePos_Normalized = NormalizedMousePos;
}

void Camera::MouseDown(const int32_t& button)
{
	MouseStatus[button] = 1;
}

void Camera::MousUp(const int32_t& button)
{
	MouseStatus[button] = 0;
}

CameraOptix Camera::GetOptixCamera() const
{
	CameraOptix cam;
	cam.Position = Eye;
	cam.LookingDirection = normalize(At - Eye);
	cam.CosFovy = CosFovy;
	const float aspectRatio = Width / (float)Height;
	cam.Horizontal = CosFovy * aspectRatio * normalize(cross(cam.LookingDirection, Up));
	cam.Vertical = CosFovy * normalize(cross(cam.Horizontal, cam.LookingDirection));

	return cam;
}

bool Camera::IsDynamic() const
{
	return true;
}