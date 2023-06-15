#include "Window.h"

#include <string>


Window::Window(const std::string& windowTitle, const uint32_t& width /* = 1024*/, const uint32_t& height /* = 768 */)// : osc::GLFWindow(windowTitle)
{
	// set error callback first to get all errors
	glfwSetErrorCallback(ErrorCallback);

	if (!glfwInit())
	{
		throw std::runtime_error("Could not initialize glfw!");
	}

	std::cout << "glfw initialized!" << std::endl;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

	OptixRenderer.Resize(osc::vec2i(width, height));
	glfwWindow = glfwCreateWindow(width, height, windowTitle.c_str(), nullptr, nullptr);
	if (!glfwWindow)
	{
		throw std::runtime_error("Could not create glfw window!");
	}

	glfwSetWindowUserPointer(glfwWindow, this);
	glfwMakeContextCurrent(glfwWindow);
	glfwSwapInterval(1);

	// callbacks (other than error)
	glfwSetMouseButtonCallback(glfwWindow, OnMouseButtonPressedOrReleased);
	glfwSetKeyCallback(glfwWindow, OnKeyPressedOrReleased);

	// OpenGL setup that doesn't need to change on draw
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

	glViewport(0, 0, width, height);
}

Window::~Window()
{
	if (glfwWindow)
	{
		glfwDestroyWindow(glfwWindow);
	}
	glfwTerminate();
}

void Window::ErrorCallback(int32_t error, const char* description)
{
	fprintf(stderr, "GLFW Error: %s\n", description);
}

void Window::Render()
{
	OptixRenderer.Render();
}

void Window::Draw()
{
	int32_t width, height;
	glfwGetFramebufferSize(glfwWindow, &width, &height);
	OptixRenderer.DownloadPixels(Pixels.data());

	// to make the output visible, render a simple OpenGL quad and apply the framebuffer content as a texture
	if (FramebufferTexture == 0)
	{
		glGenTextures(1, &FramebufferTexture);
	}

	glBindTexture(GL_TEXTURE_2D, FramebufferTexture);
	GLenum texFormat = GL_RGBA;
	GLenum texType = GL_UNSIGNED_BYTE;
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, Pixels.data());

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, FramebufferTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.f, (float)width, 0.f, (float)height, -1.f, 1.f);

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0.f, 0.f);
		glVertex3f(0.f, 0.f, 0.f);

		glTexCoord2f(0.f, 1.f);
		glVertex3f(0.f, (float)height, 0.f);

		glTexCoord2f(1.f, 1.f);
		glVertex3f((float)width, (float)height, 0.f);

		glTexCoord2f(1.f, 0.f);
		glVertex3f((float)width, 0.f, 0.f);
	}
	glEnd();
}

void Window::Resize(const vec2i& size)
{
	OptixRenderer.Resize(size);
	Pixels.resize(size.x * size.y);
	glViewport(0, 0, size.x, size.y);
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

void Window::OnMouseButtonPressedOrReleased(GLFWwindow* window, int32_t button, int32_t action, int32_t mods)
{
	const bool down = action == GLFW_PRESS;
	if (down)
	{
		std::cout << "mouse button down" << std::endl;
	}
	else
	{
		std::cout << "mouse button up" << std::endl;
	}
}

void Window::OnKeyPressedOrReleased(GLFWwindow* window, int32_t key, int32_t sanCode, int32_t action, int32_t mods)
{
	std::cout << "key " << char(key) << " was ";
	if (action == GLFW_PRESS)
	{
		std::cout << " pressed!" << std::endl;
	}
	else
	{
		std::cout << " released!" << std::endl;
	}

	if (key == GLFW_KEY_ESCAPE)
	{
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}

	
	Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
	Camera* cam = win->OptixRenderer.GetCameraPtr();

	switch (key)
	{
	case GLFW_KEY_A: cam->SetEye(cam->GetEye() + vec3f(-1.f, 0.f, 0.f));
	}

	vec3f pos = cam->GetEye();
	std::cout << "Cam pos: (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
}