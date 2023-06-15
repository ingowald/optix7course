
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
		// initialize
		Renderer renderer;

		// setup OptiX Framebuffer
		const vec2i fbSize(1024, 768);
		renderer.Resize(fbSize);
		renderer.Render();

		std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
		renderer.DownloadPixels(pixels.data());

		const std::string outName = "programming_along_2.png";
		stbi_write_png(outName.c_str(), fbSize.x, fbSize.y, 4, pixels.data(), fbSize.x * sizeof(uint32_t));

		std::cout << "Image rendered and saved to " << outName << "." << std::endl;
	}
	catch (std::runtime_error& e)
	{
		std::cerr << "Runtime error!" << std::endl;
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}