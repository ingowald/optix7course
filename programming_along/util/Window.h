#pragma once

#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

#include "gdt/math/vec.h"

#include "../Renderer.h"

using namespace osc;

//struct Window : public GLFWindow
//{
//public:
//	Window(const std::string& windowTitle);
//
//	virtual void render() override;
//
//	virtual void draw() override;
//
//	virtual void resize(const vec2i& size);
//
//protected:
//	vec2i FramebufferSize;
//	GLuint FramebufferTexture{ 0 };
//	Renderer OptixRenderer;
//	std::vector<uint32_t> Pixels;
//};