
#pragma once

#include "../scene/Mesh.h"
#include "../util/Texture.h"

#include <vector>
#include <memory>

class Model
{
public:
	Model() {};
	Model(const Mesh& mesh, const std::string& name = "model");
	Model(const std::string& meshFilePath, const std::string& name = "model");

	~Model();

	std::vector<Mesh> GetMeshList() const;
	std::vector<Mesh>& GetMeshList();

	std::vector<std::shared_ptr<Texture2D>> GetTextureList() const;
	std::vector<std::shared_ptr<Texture2D>>& GetTextureList();

	void LoadTextureForMesh(Mesh& mesh, std::map<std::string, int32_t>& knownTextures,
		const std::string& textureFilePath);

	void AddMesh(const Mesh& mesh);
	void AddMeshesFromFile(const std::string& filePath);

	Mesh& GetMeshAt(const size_t& index);

	void SetName(const std::string& name);
	std::string GetName() const;

protected:
	std::vector<Mesh> MeshList;
	std::vector<std::shared_ptr<Texture2D>> TextureList;

	/**
	* bounding box
	*/
	box3f BoundingBox;

	/**
	* name, not guaranteed to be unique, mostly for debugging purposes
	*/
	std::string Name = "";
};