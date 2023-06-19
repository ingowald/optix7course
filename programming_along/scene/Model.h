
#pragma once

#include "../scene/Mesh.h"

#include <vector>

class Model
{
public:
	Model() {};
	Model(const Mesh& mesh);
	Model(const std::string& meshFilePath);

	std::vector<Mesh> GetMeshList() const;
	std::vector<Mesh>& GetMeshList();

	void AddMesh(const Mesh& mesh);
	void AddMeshFromFile(const std::string& filePath);

protected:
	std::vector<Mesh> MeshList;
};