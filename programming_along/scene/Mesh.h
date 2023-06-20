#pragma once

#include <vector>

#include "cuda_runtime.h"

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
	
	std::vector<vec3f> Vertices;
	std::vector<vec3f> Normals;
	std::vector<vec2f> TexCoords;
	std::vector<vec3i> Indices;

	/**
	* Diffuse color that the mesh is to be rendered with
	* Will be combined with the diffuse texture
	* (set to (1, 1, 1) for only diffuse texture results)
	*/
	vec3f DiffuseColor = vec3f(1, 1, 1);

	// TODO: support multiple (diffuse) textures
	/**
	* Texture id for the diffuse texture. 
	* Will be combined with the diffuse color
	*/
	int32_t DiffuseTextureId = -1;
};

/**
* The mesh as used in the device code
*/
struct MeshSbtData
{
	vec3f DiffuseColor;
	vec3f* Vertices;
	vec3f* Normals;
	vec2f* TexCoords;
	vec3i* Indices;

	cudaTextureObject_t Texture;
	uint8_t HasTexture = false;
};