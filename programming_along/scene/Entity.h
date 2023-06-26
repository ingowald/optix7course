
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

	Entity(const std::string& name) : Name(name)
	{
		TransfomrationMatrix = affine3f::translate(vec3f(0.f));
	}

	virtual void Initialize() {};

	virtual void Tick(const float& deltaTime_Seconds) = 0;

	affine3f GetTransformationMatrix() const;
	virtual void SetTransformationMatrix(const affine3f& matrix);

	std::string GetName() const;
	void SetName(const std::string& name);

	virtual bool IsDynamic() const;

protected:
	affine3f TransfomrationMatrix;

	/**
	* name, not guaranteed to be unique, mostly for debugging purposes
	*/
	std::string Name;
};