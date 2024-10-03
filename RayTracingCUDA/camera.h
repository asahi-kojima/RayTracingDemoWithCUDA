#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include "root.h"
#include "vec3.h"
#include "util.h"
#include "ray.h"

class Camera
{
public:
	__host__ __device__ Camera() = default;
	__host__ __device__ Camera(f32 vfov, f32 aspect);
	__host__ __device__ Camera(vec3 lookFrom, vec3 lookAt, vec3 vUp, f32 vfov, f32 aspect, f32 aperture = 0, f32 focusDist = 1);

	__host__ __device__ ray getRay(f32 s, f32 t);

	vec3 origin;
	vec3 lowerLeftCorner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	f32 lensRadius;
};