#pragma once

#include <vector>

#include <gdt/math/vec.h>
#include <gdt/math/AffineSpace.h>

#include "3rdParty/tiny_obj_loader.h"

using namespace gdt;

/**
* The mesh as used in the host code
*/
struct Mesh
{
	Mesh();

	void AddUnitCube(const affine3f& transfomrationMatrix);

	void AddCube(const vec3f& center, const vec3f& size);

	void AddTriangle(const vec3f vertices[3], const vec3i indices);

	int32_t FindOrAddVertex(tinyobj::attrib_t& attributes,
		const tinyobj::index_t& idx,
		std::map<tinyobj::index_t, int32_t>& knownVertices);

public:
	vec3f DiffuseColor;
	std::vector<vec3f> Vertices;
	std::vector<vec3f> Normals;
	std::vector<vec2f> TexCoords;
	std::vector<vec3i> Indices;
};

/**
* The mesh as used in the device code
*/
struct MeshSbtData
{
	vec3f DiffuseColor;
	vec3f* Vertices;
	vec3f* Normals;
	vec3i* Indices;
};