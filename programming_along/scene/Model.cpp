
#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"

#include <set>
#include <filesystem>

namespace fs = std::filesystem;

Model::Model(std::shared_ptr<Mesh> mesh, const std::string& name /* = "model" */) : Entity(name)
{
	MeshList.push_back(mesh);
}

Model::Model(const std::string& meshFilePath, const std::string& name /* = "model" */) : Entity(name)
{
	AddMeshesFromFile(meshFilePath);
}

Model::~Model()
{

}

void Model::Tick(const float& deltaTime_Seconds)
{

}

std::vector<std::shared_ptr<Mesh>> Model::GetMeshList() const
{
	return MeshList;
}

std::vector<std::shared_ptr<Mesh>>& Model::GetMeshList()
{
	return MeshList;
}

std::vector<std::shared_ptr<Texture2D>> Model::GetTextureList() const
{
	return TextureList;
}

std::vector<std::shared_ptr<Texture2D>>& Model::GetTextureList()
{
	return TextureList;
}

bool Model::LoadTextureForMesh(std::shared_ptr<Mesh> mesh, const std::string& textureFilePath)
{
	if (textureFilePath == "" || fs::is_directory(textureFilePath))
	{
		return false;
	}

	if (textureFilePath.find(".png") == std::string::npos
		&& textureFilePath.find(".jpg") == std::string::npos
		&& textureFilePath.find(".jpeg") == std::string::npos)
	{
		std::cout << "invalid texture path!" << std::endl;
		return false;
	}
	if (!fs::exists(textureFilePath))
	{
		std::cout << "invalid filepath " << textureFilePath << " for model " << Name << std::endl;
		return false;
	}

	// TODO: support reuse of textures

	vec2i resolution;
	int32_t comp;	//components, don't need this here except for stbi
	uint8_t* imageData = stbi_load(textureFilePath.c_str(),
		&resolution.x, &resolution.y, &comp, STBI_rgb_alpha);

	if (imageData)
	{
		mesh->DiffuseTextureId = static_cast<int32_t>(TextureList.size());
		std::shared_ptr<Texture2D> texture = std::make_shared<Texture2D>();
		texture->Resolution = resolution;
		texture->Pixels = reinterpret_cast<uint32_t*>(imageData);

		// mirror along the y axis, due to some error in stbi loading the images that way
		for (int32_t y = 0; y < resolution.y / 2; y++)
		{
			uint32_t* lineY = texture->Pixels + y * resolution.x;
			uint32_t* mirroredY = texture->Pixels + (resolution.y - 1 - y) * resolution.x;
			for (int32_t x = 0; x < resolution.x; x++)
			{
				std::swap(lineY[x], mirroredY[x]);
			}
		}

		TextureList.push_back(texture);
		return true;
	}
	else
	{
		throw std::runtime_error("Could not load texture from " + textureFilePath + " for model " + Name);
	}
}

void Model::LoadTextureForMesh(std::shared_ptr<Mesh> mesh, std::map<std::string, int32_t>& knownTextures,
	const std::string& textureFilePath)
{
	if (knownTextures.find(textureFilePath) != knownTextures.end())
	{
		// texture has been loaded already and can be reused
		mesh->DiffuseTextureId = knownTextures[textureFilePath];
		return;
	}

	if (LoadTextureForMesh(mesh, textureFilePath))
	{
		knownTextures[textureFilePath] = mesh->DiffuseTextureId;
	}	
}

void Model::AddMesh(const std::shared_ptr<Mesh> mesh)
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

	std::map<std::string, int32_t> knownTextures;

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
			std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();

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
					mesh->FindOrAddVertex(attributes, idx0, knownVertices),
					mesh->FindOrAddVertex(attributes, idx1, knownVertices),
					mesh->FindOrAddVertex(attributes, idx2, knownVertices)
				);
				mesh->Indices.push_back(idx);
				mesh->DiffuseColor = (const vec3f&)materials[materialId].diffuse;

				const std::string texturePath =
					materialFileDir	
					//the textures folder is implicit in diffuse_texname
					+ materials[materialId].diffuse_texname; 
				LoadTextureForMesh(mesh, knownTextures, texturePath);

				if (mesh->DiffuseTextureId == -1)
				{
					mesh->DiffuseColor = gdt::randomColor(materialId);
				}				
			}

			if (!mesh->Vertices.empty())
			{
				MeshList.push_back(mesh);
			}
		}
	}

	for (std::shared_ptr<Mesh> mesh : MeshList)
	{
		for (const vec3f& vertex : mesh->Vertices)
		{
			BoundingBox.extend(vertex);
		}
	}

	std::cout << "created a total of " << std::to_string(MeshList.size()) 
		<< " meshes for model at " 
		<< filePath << std::endl;
}

std::shared_ptr<Mesh> Model::GetMeshAt(const size_t& index)
{
	return MeshList[index];
}
