#pragma once

#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

#include "gdt/math/vec.h"

#include "../Renderer.h"

struct Window// : public GLFWindow
{
public:
	Window(const std::string& windowTitle, const uint32_t& width = 1024, const uint32_t& height = 768);

	~Window();

	static void ErrorCallback(int32_t error, const char* description);

	virtual void Render();

	virtual void Draw();

	virtual void Resize(const vec2i& size);

	/**
	* Starts (and maintains) the render loop
	*/
	virtual void Run();

	/**
	* Callback that is called when mouse button is pressed or released
	*/
	static void OnMouseButtonPressedOrReleased(GLFWwindow* window, int32_t button, int32_t action, int32_t mods);

	static void OnCursorMoved(GLFWwindow* window, double x, double y);

	static void OnKeyPressedOrReleased(GLFWwindow* window, int32_t key, int32_t sanCode, int32_t action, int32_t mods);

	static void OnWindowResize(GLFWwindow* window, int32_t width, int32_t height);

	Renderer* GetRenderer();

	const vec2i& GetFramebufferSize() const;

protected:

	// glfw setup
	GLFWwindow* glfwWindow = nullptr;

	Renderer OptixRenderer;

	vec2i FramebufferSize;
	GLuint FramebufferTexture{ 0 };
	std::vector<vec4f> Pixels;

	// controls
	vec2f MousePos;
};