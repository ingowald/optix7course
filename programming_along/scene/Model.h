
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
	void AddMeshesFromFile(const std::string& filePath);

protected:
	std::vector<Mesh> MeshList;

	// bounding box
	box3f BoundingBox;
};