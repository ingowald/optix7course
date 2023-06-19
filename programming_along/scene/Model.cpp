
#include "Model.h"

Model::Model(const Mesh& mesh)
{
	MeshInstance = mesh;
}

Mesh Model::GetMesh() const 
{
	return MeshInstance;
}

Mesh& Model::GetMesh()
{
	return MeshInstance;
}

void Model::SetMesh(const Mesh& mesh)
{
	MeshInstance = mesh;
}