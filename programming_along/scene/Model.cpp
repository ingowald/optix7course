
#include "Model.h"

Model::Model(const Mesh& mesh)
{
	MeshList.push_back(mesh);
}

Model::Model(const std::string& meshFilePath)
{
	AddMeshFromFile(meshFilePath);
}

std::vector<Mesh> Model::GetMeshList() const 
{
	return MeshList;
}

std::vector<Mesh>& Model::GetMeshList()
{
	return MeshList;
}

void Model::AddMesh(const Mesh& mesh)
{
	MeshList.push_back(mesh);
}

void Model::AddMeshFromFile(const std::string& filePath)
{
	MeshList.push_back(Mesh(filePath));
}