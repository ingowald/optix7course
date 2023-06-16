

#include "Mesh.h"

void Mesh::AddUnitCube(const affine3f& transfomrationMatrix)
{
	size_t firstVertexId = Vertices.size();

	Vertices.push_back(xfmPoint(transfomrationMatrix, vec3f(0.f, 0.f, 0.f)));
	Vertices.push_back(xfmPoint(transfomrationMatrix, vec3f(1.f, 0.f, 0.f)));
	Vertices.push_back(xfmPoint(transfomrationMatrix, vec3f(0.f, 1.f, 0.f)));
	Vertices.push_back(xfmPoint(transfomrationMatrix, vec3f(1.f, 1.f, 0.f)));

	Vertices.push_back(xfmPoint(transfomrationMatrix, vec3f(0.f, 0.f, 1.f)));
	Vertices.push_back(xfmPoint(transfomrationMatrix, vec3f(1.f, 0.f, 1.f)));
	Vertices.push_back(xfmPoint(transfomrationMatrix, vec3f(0.f, 1.f, 1.f)));
	Vertices.push_back(xfmPoint(transfomrationMatrix, vec3f(1.f, 1.f, 1.f)));

	int32_t indices[] = {
		0, 1, 3,	2, 3, 0,
		5, 7, 6,	5, 6, 4,
		0, 4, 5,	0, 5, 1,
		2, 3, 7,	2, 7, 6,
		1, 5, 7,	1, 7, 3,
		4, 0, 2,	4, 2, 6
	};
	for (size_t i = 0; i < 12; i++)
	{
		Indices.push_back((int32_t)firstVertexId +
			vec3i(
				indices[3 * i + 0],
				indices[3 * i + 1],
				indices[3 * i + 2]
			)
		);
	}
}

void Mesh::AddCube(const vec3f& center, const vec3f& size)
{
	affine3f transformationMatrix;
	transformationMatrix.p = center - 0.5f * size;
	transformationMatrix.l.vx = vec3f(size.x, 0.f, 0.f);
	transformationMatrix.l.vy = vec3f(0.f, size.y, 0.f);
	transformationMatrix.l.vz = vec3f(0.f, 0.f, size.z);
	AddUnitCube(transformationMatrix);
}

void Mesh::AddTriangle(const vec3f vertices[3], const vec3i indices)
{
	size_t firstVertexId = Vertices.size();
	for (size_t i = 0; i < 3; i++)
	{
		Vertices.push_back(vertices[i]);
	}
	Indices.push_back((int32_t)firstVertexId + indices);
}