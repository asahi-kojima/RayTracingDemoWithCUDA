#pragma once
#include "vec3.h"

class ray
{
public:
	__host__ __device__ ray() = default;
	__host__ __device__ ray(const  vec3& a, const vec3& b) { mA = a; mB = b; }

	__host__ __device__ vec3 origin() const { return mA; };
	__host__ __device__ vec3 direction() const { return mB; }
	__host__ __device__ vec3 pointAt(float t) const { return mA + mB * t; }

	vec3 mA;
	vec3 mB;
};