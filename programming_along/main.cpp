
#include <stdexcept>
#include <iostream>
#include <string>

#include "cuda_runtime.h"

#include "gdt/math/vec.h"

#include "Renderer.h"
#include "util/Window.h"

using namespace gdt;


int main(int ac, char **av)
{
	try
	{
		// initialize
		Renderer renderer;

		// setup OptiX Framebuffer
		const vec2i fbSize(1024, 768);
		renderer.Resize(fbSize);
	}
	catch (std::runtime_error& e)
	{
		std::cerr << "Runtime error!" << std::endl;
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}