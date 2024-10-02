#include "vec3.h"



vec3& vec3::operator+=(const vec3& v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

vec3& vec3::operator-=(const vec3& v)
{
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

vec3& vec3::operator*=(const vec3& v)
{
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

vec3& vec3::operator/=(const vec3& v)
{
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

vec3& vec3::operator*=(const f32 t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

vec3& vec3::operator/=(const f32 t)
{
	f32 k = 1 / t;
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

void vec3::normalize()
{
	f32 k = 1 / length();
	this->operator*=(k);
}

vec3 normalize(vec3 v)
{
	v.normalize();
	return v;
}


vec3 operator+(const vec3& a, const vec3& b)
{
	vec3 newVec3(a);
	newVec3 += b;
	return newVec3;
}

vec3 operator-(const vec3& a, const vec3& b)
{
	vec3 newVec3(a);
	newVec3 -= b;
	return newVec3;
}

vec3 operator-(const vec3& a, f32 t)
{
	vec3 newVec3(a);
	newVec3.e[0] -= t;
	newVec3.e[1] -= t;
	newVec3.e[2] -= t;
	return newVec3;
}

vec3 operator*(const vec3& a, const vec3& b)
{
	vec3 newVec3(a);
	newVec3 *= b;
	return newVec3;
}

vec3 operator*(const vec3& a, f32 t)
{
	vec3 newVec3(a);
	newVec3 *= t;
	return newVec3;
}


vec3 operator*(f32 t, const vec3& a)
{
	vec3 newVec3(a);
	newVec3 *= t;
	return newVec3;
}

vec3 operator/(const vec3& a, f32 t)
{
	vec3 newVec3(a);
	newVec3 /= t;
	return newVec3;
}

f32 dot(const vec3& a, const vec3& b)
{
	f32 result = 0.0f;
	for (s32 i = 0; i < 3; i++)
	{
		result += a.e[i] * b.e[i];
	}
	return result;
}

vec3 vec3::cross(const vec3& b) const
{
	const f32* ae = this->e;
	const f32* be = b.e;
	return vec3(
		ae[1] * be[2] - ae[2] * be[1],
		ae[2] * be[0] - ae[0] * be[2],
		ae[0] * be[1] - ae[1] * be[0]
	);
}


 vec3 randomInUnitSphere()
{
	vec3 p;
	do
	{
		p = 2.0f * vec3(randomF64(), randomF64(), randomF64()) - vec3(1, 1, 1);
	} while (p.squaredLength() >= 1.0f);
	return p;
}


 vec3 reflect(const vec3& v, const vec3& n)
{
	return v - 2 * dot(v, n) * n;
}

 vec3 cross(const vec3& a, const vec3& b)
{
	return a.cross(b);
}
