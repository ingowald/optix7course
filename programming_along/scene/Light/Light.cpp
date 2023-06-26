
#include "Light.h"

#include "../../scene/Mesh.h"
#include "../../scene/Model.h"

Light::Light()
{
	const std::string spherePath = "../../models/sphere/sphere.obj";
	Proxy = std::make_shared<Model>(spherePath, "Light Proxy");

	Name = "Light";
}

Light::Light(const vec3f& location, const vec3f& power /* = vec3f(1000000.f)*/)
{
	Location = location;
	Power = power;
}

void Light::Tick(const float& deltaTime_Seconds)
{

}

vec3f Light::GetLocation() const
{
	return Location;
}

void Light::SetLocation(const vec3f& location)
{
	Location = location;
}

vec3f Light::GetPower() const
{
	return Power;
}

void Light::SetPower(const vec3f& power)
{
	Power = power;
}

bool Light::GetShowProxyMesh() const
{
	return ShowProxyMesh;
}

void Light::SetShowProxyMesh(const bool& showMesh)
{
	ShowProxyMesh = showMesh;
}

std::shared_ptr<Model> Light::GetProxy() const
{
	return Proxy;
}

std::shared_ptr<LightOptix> Light::GetOptixLight() const
{
	std::shared_ptr<LightOptix> l = std::make_shared<LightOptix>();
	l->Location = Location;
	return l;
}