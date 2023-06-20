
#pragma once

#include "../scene/Mesh.h"
#include "../util/Texture.h"

#include <vector>

class Model
{
public:
	Model() {};
	Model(const Mesh& mesh, const std::string& name = "model");
	Model(const std::string& meshFilePath, const std::string& name = "model");

	std::vector<Mesh> GetMeshList() const;
	std::vector<Mesh>& GetMeshList();

	std::vector<Texture2D> GetTextureList() const;
	std::vector<Texture2D>& GetTextureList();

	void AddMesh(const Mesh& mesh);
	void AddMeshesFromFile(const std::string& filePath);

	Mesh& GetMeshAt(const size_t& index);

	void SetName(const std::string& name);
	std::string GetName() const;

protected:
	std::vector<Mesh> MeshList;
	std::vector<Texture2D> TextureList;

	/**
	* bounding box
	*/
	box3f BoundingBox;

	/**
	* name, not guaranteed to be unique, mostly for debugging purposes
	*/
	std::string Name = "";
};