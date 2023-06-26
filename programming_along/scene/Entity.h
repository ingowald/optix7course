
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

	virtual void Tick(const float& deltaTime_Seconds) = 0;

	affine3f GetTransformationMatrix() const;
	virtual void SetTransformationMatrix(const affine3f& matrix);

	virtual bool IsDynamic() const;

protected:
	affine3f TransfomrationMatrix;
};