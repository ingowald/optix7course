
#pragma once

#include "gdt/math/AffineSpace.h"

using namespace gdt;

/**
* Common functionality, such as Transforms
*/
class Entity
{
public:
	Entity()
	{
		TransfomrationMatrix = affine3f::translate(vec3f(0.f));
	}

	affine3f GetTransformationMatrix() const;
	void SetTransformationMatrix(const affine3f& matrix);

protected:
	affine3f TransfomrationMatrix;
};