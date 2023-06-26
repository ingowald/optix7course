
#pragma once

#include "../Entity.h"
#include "../IDynamicElement.h"

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
	vec3f Power;
};

class Light : public Entity, public IDynamicElement
{
public:
	Light();

	Light(const vec3f& location, const vec3f& power = vec3f(1000000.f));

	virtual void Tick(const float& deltaTime_Seconds) override;

	vec3f GetLocation() const;
	void SetLocation(const vec3f& location);

	vec3f GetPower() const;
	void SetPower(const vec3f& power);

	bool GetShowProxyMesh() const;
	void SetShowProxyMesh(const bool& showMesh);

	std::shared_ptr<Model> GetProxy() const;

	virtual std::shared_ptr<LightOptix> GetOptixLight() const;

protected:
	vec3f Location = vec3f(0.f);
	vec3f Power;

	bool ShowProxyMesh = false;
	std::shared_ptr<Model> Proxy;
};
