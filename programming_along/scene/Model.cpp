
#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"

#include <set>

Model::Model(const Mesh& mesh, const std::string& name /* = "model" */) : Name(name)
{
	MeshList.push_back(mesh);
}

Model::Model(const std::string& meshFilePath, const std::string& name /* = "model" */) : Name(name)
{
	AddMeshesFromFile(meshFilePath);
}

std::vector<Mesh> Model::GetMeshList() const 
{
	return MeshList;
}

std::vector<Mesh>& Model::GetMeshList()
{
	return MeshList;
}

std::vector<Texture2D> Model::GetTextureList() const
{
	return TextureList;
}

std::vector<Texture2D>& Model::GetTextureList()
{
	return TextureList;
}

void Model::AddMesh(const Mesh& mesh)
{
	MeshList.push_back(mesh);
}

void Model::AddMeshesFromFile(const std::string& filePath)
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

	std::cout << "Successfully loaded model at " << filePath << std::endl;

	for (int32_t shapeId = 0; shapeId < (int32_t)shapes.size(); shapeId++)
	{
		tinyobj::shape_t& shape = shapes[shapeId];

		std::set<int32_t> materialIds;
		for (int32_t faceMaterialId : shape.mesh.material_ids)
		{
			materialIds.insert(faceMaterialId);
		}

		for (int32_t materialId : materialIds)
		{
			std::map<tinyobj::index_t, int32_t> knownVertices;
			Mesh mesh;

			for (int32_t faceId = 0; faceId < shape.mesh.material_ids.size(); faceId++)
			{
				if (shape.mesh.material_ids[faceId] != materialId)
				{
					continue;
				}

				tinyobj::index_t idx0 = shape.mesh.indices[3 * faceId + 0];
				tinyobj::index_t idx1 = shape.mesh.indices[3 * faceId + 1];
				tinyobj::index_t idx2 = shape.mesh.indices[3 * faceId + 2];

				vec3i idx(
					mesh.FindOrAddVertex(attributes, idx0, knownVertices),
					mesh.FindOrAddVertex(attributes, idx1, knownVertices),
					mesh.FindOrAddVertex(attributes, idx2, knownVertices)
				);
				mesh.Indices.push_back(idx);
				mesh.DiffuseColor = (const vec3f&)materials[materialId].diffuse;

				//TODO: use textures
				std::cout << "Using random colors for model loaded from obj!" << std::endl;
				mesh.DiffuseColor = gdt::randomColor(materialId);
			}

			if (!mesh.Vertices.empty())
			{
				MeshList.push_back(mesh);
			}
		}
	}

	for (const Mesh& mesh : MeshList)
	{
		for (const vec3f& vertex : mesh.Vertices)
		{
			BoundingBox.extend(vertex);
		}
	}

	std::cout << "created a total of " << std::to_string(MeshList.size()) 
		<< " meshes for model at " 
		<< filePath << std::endl;
}

Mesh& Model::GetMeshAt(const size_t& index)
{
	return MeshList[index];
}

void Model::SetName(const std::string& name)
{
	Name = name;
}

std::string Model::GetName() const
{
	return Name;
}
