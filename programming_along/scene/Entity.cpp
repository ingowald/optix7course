
#include "Entity.h"

affine3f Entity::GetTransformationMatrix() const
{
	return TransfomrationMatrix;
}

void Entity::SetTransformationMatrix(const affine3f& matrix)
{
	TransfomrationMatrix = matrix;
}

std::string Entity::GetName() const
{
	return Name;
}

void Entity::SetName(const std::string& name)
{
	Name = name;
}

bool Entity::IsDynamic() const
{
	return false;
}