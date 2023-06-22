
#pragma once

#include "Entity.h"

#include "gdt/math/vec.h"

using namespace gdt;

struct Mesh;
class Model;

/**
* struct to provide easier access for OptiX (/CUDA) kernels
*/
struct LightOptix
{
	vec3f Location;
};

class Light : public Entity
{
public:
	Light();

	Light(const vec3f& location);

	vec3f GetLocation() const;

	void SetLocation(const vec3f& location);

	bool GetShowProxyMesh() const;
	void SetShowProxyMesh(const bool& showMesh);

	std::shared_ptr<Model> GetProxy() const;

	virtual LightOptix GetOptixLight() const;

protected:
	vec3f Location = vec3f(0.f);

	bool ShowProxyMesh = false;

	std::shared_ptr<Model> Proxy;
};
