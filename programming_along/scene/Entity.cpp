
#include "Entity.h"

affine3f Entity::GetTransformationMatrix() const
{
	return TransfomrationMatrix;
}

void Entity::SetTransformationMatrix(const affine3f& matrix)
{
	TransfomrationMatrix = matrix;
}