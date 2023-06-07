
#include <stdexcept>
#include <iostream>
#include <string>

#include "cuda_runtime.h"

#include "optix_function_table_definition.h"
#include "optix_stubs.h"

#include "gdt/math/vec.h"

using namespace gdt;

void InitOptix()
{
	// check that CUDA works and a CUDA capable device is found
	cudaFree(0);
	int32_t numDevices = 0;
	cudaGetDeviceCount(&numDevices);
	if (numDevices <= 0)
	{
		throw std::runtime_error("No CUDA Device available!");
	}

	std::cout << "Found " << std::to_string(numDevices) << " CUDA capable devices!" << std::endl;

	OptixResult result = optixInit();
	if (result != OPTIX_SUCCESS)
	{
		throw std::runtime_error("Could not initialize OptiX!");
	}

	std::cout << "OptiX initialized!" << std::endl;
}

int main(int ac, char **av)
{
	try
	{
		// initialize
		InitOptix();

		// setup OptiX Framebuffer
		const vec2i fbSize(1024, 768);
	}
	catch (std::runtime_error& e)
	{
		std::cerr << "Runtime error!" << std::endl;
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}