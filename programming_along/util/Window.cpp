#include "Window.h"

#include <string>


Window::Window(const std::string& windowTitle, const uint32_t& width /* = 1024*/, const uint32_t& height /* = 768 */)// : osc::GLFWindow(windowTitle)
{
	if (!glfwInit())
	{
		throw std::runtime_error("Could not initialize glfw!");
	}

	std::cout << "glfw initialized!" << std::endl;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

	glfwWindow = glfwCreateWindow(width, height, windowTitle.c_str(), nullptr, nullptr);
	if (!glfwWindow)
	{
		throw std::runtime_error("Could not create glfw window!");
	}

	glfwSetWindowUserPointer(glfwWindow, this);
	glfwMakeContextCurrent(glfwWindow);
	glfwSwapInterval(1);
}

Window::~Window()
{
	if (glfwWindow)
	{
		glfwDestroyWindow(glfwWindow);
	}
	glfwTerminate();
}

void Window::Render()
{
	OptixRenderer.Render();
}

void Window::Draw()
{
	std::cout << "I'd draw if I could" << std::endl;
}

void Window::Resize(const vec2i& size)
{
	OptixRenderer.Resize(size);
}

void Window::Run()
{
	int32_t width, height;
	glfwGetFramebufferSize(glfwWindow, &width, &height);
	Resize(osc::vec2i(width, height));

	while (!glfwWindowShouldClose(glfwWindow))
	{
		Render();
		Draw();

		glfwSwapBuffers(glfwWindow);
		glfwPollEvents();
	}
}