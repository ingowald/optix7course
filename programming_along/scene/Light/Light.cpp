
#include "Light.h"

#include "../../scene/Mesh.h"
#include "../../scene/Model.h"

Light::Light()
{
	const std::string spherePath = "../../models/sphere/sphere.obj";
	Proxy = std::make_shared<Model>(spherePath, "Light Proxy");
}

Light::Light(const vec3f& location)
{
	Location = location;
}

vec3f Light::GetLocation() const
{
	return Location;
}

void Light::SetLocation(const vec3f& location)
{
	Location = location;
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

LightOptix Light::GetOptixLight() const
{
	LightOptix l;
	l.Location = Location;
	return l;
}