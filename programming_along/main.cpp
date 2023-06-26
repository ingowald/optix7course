
#include <stdexcept>
#include <iostream>
#include <string>

#include "cuda_runtime.h"

#include "util/Window.h"
#include "scene/Mesh.h"
#include "scene/Model.h"
#include "scene/Light/RotatingLight.h"


using namespace gdt;


int main(int ac, char **av)
{
	try
	{
		Window window("OptiX window", 1920, 1080);

		std::shared_ptr<Mesh> cube1 = std::make_shared<Mesh>();
		cube1->AddCube(vec3f(0.f, -1.5f, 0.f), vec3f(100.f, 1.f, 100.f));
		cube1->DiffuseColor = vec3f(.2f, .9f, .05f);
		std::shared_ptr<Model> m1 = std::make_shared<Model>(cube1, "Base cubes");
		std::shared_ptr<Mesh> cube2 = std::make_shared<Mesh>();
		cube2->AddCube(vec3f(0.f, -2.f, 0.f), vec3f(100.f, 1.f, 100.f));
		cube2->DiffuseColor = vec3f(.2f, .1f, .7f);
		m1->AddMesh(cube2);
		std::shared_ptr<Mesh> cube3 = std::make_shared<Mesh>();
		cube3->AddCube(vec3f(0.f, 0.f, 0.f), vec3f(20.f, 20.f, 20.f));
		cube3->DiffuseColor = vec3f(.8f, 0.1f, 0.2f);
		std::shared_ptr<Model> m2 = std::make_shared<Model>(cube3, "Small cube");
		m2->LoadTextureForMesh(cube3, "../../models/crytek_sponza/textures/lion.png");

		// load the sponza level
#ifdef _WIN32
		const std::string filePath = "../../models/crytek_sponza/sponza.obj";
#else
		// untested, but should work on linux
		const std::string filePath = "../models/crytek_sponza/sponza.obj";
#endif
		std::shared_ptr<Model> sponza = std::make_shared<Model>(filePath, "sponza scene");

		Renderer* renderer = window.GetRenderer();
		renderer->AddModel(sponza);
		renderer->AddModel(m1);
		renderer->AddModel(m2);

		renderer->InitializeCamera(
			vec3f(-775.f, 140.f, 4.f),		//eye
			vec3f(-774.f, 139.95f, 4.f),	//at
			vec3f(0.f, 1.f, 0.f)			//up
		);

		//std::shared_ptr<RotatingLight> light = std::make_shared<RotatingLight>(
		//	vec3f(-907.108f, 2205.875f, -400.0267f), 1.f, 50.f
		//);
		std::shared_ptr<QuadLight> light = std::make_shared<QuadLight>(
			vec3f(-1200.f, 800.f, -200.f),
			vec3f(4000000.f),
			vec2f(200.f, 200.f)
		);
		light->SetRotationRadius(100.f);
		renderer->AddLight(light);

		renderer->Init();

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