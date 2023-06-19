
#pragma once

#include "../scene/Mesh.h"

class Model
{
public:
	Model() {};
	Model(const Mesh& mesh);

	Mesh GetMesh() const;
	Mesh& GetMesh();

	void SetMesh(const Mesh& mesh);

protected:
	Mesh MeshInstance;
};