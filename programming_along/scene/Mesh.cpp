

#include "Mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"

#include <set>

Mesh::Mesh()
{
	// nothing to do here, mesh will be added dynamically
}

Mesh::Mesh(const std::string& filePath)
{
	LoadFromObj(filePath);
}

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

void Mesh::LoadFromObj(const std::string& filePath)
{
	const std::string materialFileDir = filePath.substr(0, filePath.rfind('/') + 1);
	std::cout << "Searching material .mtl file for " << filePath << " at " << materialFileDir << std::endl;

	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = "";

	bool result = tinyobj::LoadObj(
		&attributes, &shapes, &materials, 
		&err, &err, /* write warnings and errors into the same string */
		filePath.c_str(), materialFileDir.c_str(),
		true /* triangulate */
	);

	if (!result)
	{
		throw std::runtime_error("Could not read obj model from " + filePath
			+ ": " + err);
	}

	if (materials.empty())
	{
		throw std::runtime_error("Could not parse materials from " + materialFileDir);
	}

	std::cout << "Succesfully loaded model at " << filePath << std::endl;

	for (int32_t shapeId = 0; shapeId < (int32_t)shapes.size(); shapeId++)
	{
		tinyobj::shape_t& shape = shapes[shapeId];

		std::set<int32_t> materialIds;
		for (int32_t faceMaterialId : shape.mesh.material_ids)
		{
			materialIds.insert(faceMaterialId);
		}
	}
}