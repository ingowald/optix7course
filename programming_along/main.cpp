
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
		Window window("OptiX window", 1920, 1080);

		Mesh cube;
		//cube.AddCube(vec3f(0.f, -1.5f, 0.f), vec3f(10.f, .1f, 10.f));
		//cube.AddCube(vec3f(0.f, 0.f, 0.f), vec3f(2.f, 2.f, 2.f));
		vec3f verts[3] = {
			vec3f(-1.f, 0.f, 0.f),
			vec3f(0.f, 1.f, 0.f),
			vec3f(1.f, 0.f, 0.f)
		};
		cube.AddTriangle(verts, vec3i(0, 1, 2));

		window.GetRenderer()->AddMesh(cube);

		window.GetRenderer()->SetCameraPositionAndOrientation(
			vec3f(0.f, 0.f, -1.f),//vec3f(-10.f, 2.f, -12.f),	//eye
			vec3f(0.f, 0.f, 0.f),		//at
			vec3f(0.f, 1.f, 0.f)		//up
		);

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