#pragma once
#include "hitable.h"
#include "material.h"

class Sphere : public Hitable
{
public:
	__host__  __device__ Sphere() = default;
	__host__  __device__ Sphere(const vec3& cen, float r, Material* pMaterial) : center(cen), radius(r), pMaterial(pMaterial) {}

	__host__ __device__ virtual bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const;

	vec3 center;
	float radius;
	Material* pMaterial;
};

inline bool Sphere::hit(const ray& r, float t_min, float t_max, HitRecord& rec) const
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = 2 * dot(r.direction(), oc);
	float c = dot(oc, oc) - radius * radius;
	float D = b * b - 4 * a * c;

	if (D > 0)
	{
		//‰ğ‚ª³•‰
		if (c < 0)
		{
			float tmp = (-b + sqrt(D)) / (2.0 * a);
			if (tmp < t_max && tmp > t_min)
			{
				rec.t = tmp;
				rec.p = r.pointAt(tmp);
				rec.normal = (rec.p - center) / radius;
				rec.pMaterial = pMaterial;
				return true;
			}
		}
		else
		{
			float tmp = (-b - sqrt(D)) / (2.0 * a);
			if (tmp < t_max && tmp > t_min)
			{
				rec.t = tmp;
				rec.p = r.pointAt(tmp);
				rec.normal = (rec.p - center) / radius;
				rec.pMaterial = pMaterial;
				return true;
			}

			tmp = (-b + sqrt(D)) / (2.0 * a);
			if (tmp < t_max && tmp > t_min)
			{
				rec.t = tmp;
				rec.p = r.pointAt(tmp);
				rec.normal = (rec.p - center) / radius;
				rec.pMaterial = pMaterial;
				return true;
			}
		}
	}

	return false;
}