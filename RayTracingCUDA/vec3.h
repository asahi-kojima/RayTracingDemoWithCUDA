#pragma once
#include <math.h>
#include "root.h"
#include "util.h"



class vec3
{
public:
	__host__ __device__ vec3() = default;

	__host__ __device__ vec3(f32 e0, f32 e1, f32 e2)
	{
		e[0] = e0;
		e[1] = e1;
		e[2] = e2;
	}

	__host__ __device__ f32 x() const { return e[0]; }
	__host__ __device__ f32 y() const { return e[1]; }
	__host__ __device__ f32 z() const { return e[2]; }
	__host__ __device__ f32 r() const { return e[0]; }
	__host__ __device__ f32 g() const { return e[1]; }
	__host__ __device__ f32 b() const { return e[2]; }


	__host__ __device__ const vec3& operator+() const { return *this; }
	__host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ f32 operator[](s32 i) const { return e[i]; }
	__host__ __device__ f32& operator[](s32 i) { return e[i]; }

	__host__ __device__ vec3& operator+=(const vec3& v2);
	__host__ __device__ vec3& operator-=(const vec3& v2);
	__host__ __device__ vec3& operator*=(const vec3& v2);
	__host__ __device__ vec3& operator/=(const vec3& v2);
	__host__ __device__ vec3& operator*=(const f32 t);
	__host__ __device__ vec3& operator/=(const f32 t);

	__host__ __device__ f32 length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
	__host__ __device__ f32 squaredLength() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	__host__ __device__ void normalize();
	__host__ __device__ vec3 cross(const vec3&) const;
	f32 e[3];
};


__host__ __device__ vec3 normalize(vec3 v);

__host__ __device__ vec3 operator+(const vec3& a, const vec3& b);

__host__ __device__ vec3 operator-(const vec3& a, const vec3& b);

__host__ __device__ vec3 operator-(const vec3& a, f32 t);

__host__ __device__ vec3 operator*(const vec3& a, const vec3& b);

__host__ __device__ vec3 operator*(const vec3& a, f32 t);

__host__ __device__ vec3 operator*(f32 t, const vec3& a);

__host__ __device__ vec3 operator/(const vec3& a, f32 t);

__host__ __device__ f32 dot(const vec3& a, const vec3& b);

__host__ __device__ vec3 randomInUnitSphere();

__host__ __device__ vec3 reflect(const vec3& v, const vec3& n);

__host__ __device__ vec3 cross(const vec3& a, const vec3& b);