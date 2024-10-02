#pragma once

#include "ray.h"

class Material;

struct HitRecord
{
	float t;
	vec3 p;
	vec3 normal;
	Material* pMaterial;
};




class Hitable
{
public:
	__host__  __device__ virtual bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};


class HitableList : public Hitable
{
public:
	__host__  __device__  HitableList() = default;
	__host__  __device__  HitableList(Hitable** l, int n)
	{
		list = l;
		listSize = n;
	}

	__host__  __device__ virtual bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const;

	Hitable** list;
	int listSize;
};

bool HitableList::hit(const ray& r, float t_min, float t_max, HitRecord& rec) const
{
	HitRecord tmpRec;
	bool hitAnything = false;

	double closestSoFar = t_max;

	for (int i = 0; i < listSize; i++)
	{
		if (list[i]->hit(r, t_min, closestSoFar, tmpRec))
		{
			hitAnything = true;
			closestSoFar = tmpRec.t;
			rec = tmpRec;
		}
	}

	return hitAnything;
}