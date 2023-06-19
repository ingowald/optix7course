#pragma once

#include <vector>

#include <gdt/math/vec.h>
#include <gdt/math/AffineSpace.h>

using namespace gdt;

/**
* The mesh as used in the host code
*/
struct Mesh
{
	void AddUnitCube(const affine3f& transfomrationMatrix);

	void AddCube(const vec3f& center, const vec3f& size);

	void AddTriangle(const vec3f vertices[3], const vec3i indices);

	vec3f Color;
	std::vector<vec3f> Vertices;
	std::vector<vec3i> Indices;
};

/**
* The mesh as used in the device code
*/
struct MeshSbtData
{
	vec3f Color;
	vec3f* Vertices;
	vec3i* Indices;
};