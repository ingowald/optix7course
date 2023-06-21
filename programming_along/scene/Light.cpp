
#include "Light.h"

Light::Light()
{

}

Light::Light(const vec3f& location)
{
	Location = location;
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