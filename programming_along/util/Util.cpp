

#include "Util.h"

std::string Util::VecToString(const vec2f& vec)
{
	return "("
		+ std::to_string(vec.x)
		+ ", "
		+ std::to_string(vec.y)
		+ ")";
}

std::string Util::VecToString(const vec3f& vec)
{
	return "("
		+ std::to_string(vec.x)
		+ ", "
		+ std::to_string(vec.y)
		+ ", "
		+ std::to_string(vec.z)
		+ ")";
}

std::string Util::VecToString(const vec3i& vec)
{
	return "("
		+ std::to_string(vec.x)
		+ ", "
		+ std::to_string(vec.y)
		+ ", "
		+ std::to_string(vec.z)
		+ ")";
}