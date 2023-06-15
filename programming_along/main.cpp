
#include <stdexcept>
#include <iostream>
#include <string>

#include "cuda_runtime.h"

#include "gdt/math/vec.h"

#include "Renderer.h"
#include "util/Window.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

using namespace gdt;


int main(int ac, char **av)
{
	try
	{
		Window window("OptiX window");
		window.Run();
	}
	catch (std::runtime_error& e)
	{
		std::cerr << "Runtime error!" << std::endl;
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}