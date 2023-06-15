
#include <stdexcept>
#include <iostream>
#include <string>

#include "cuda_runtime.h"

#include "util/Window.h"
#include "util/Mesh.h"


using namespace gdt;


int main(int ac, char **av)
{
	try
	{
		Window window("OptiX window");

		Mesh cube;
		cube.AddCube(vec3f(0.f, -1.5f, 0.f), vec3f(10.f, .1f, 10.f));

		window.GetRenderer()->AddMesh(cube);

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