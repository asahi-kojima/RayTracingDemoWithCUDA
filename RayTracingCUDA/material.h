#pragma once
#include "hitable.h"
#include "util.h"

class Material
{
public:
	__host__  __device__ virtual bool scatter(const ray& rayIn, const HitRecord& rec, vec3& attenuation, ray& scattered) const = 0;
};

//class Lambertian : public Material
//{
//public:
//	Lambertian(const vec3& a) : albedo(a) {}
//
//	virtual bool scatter(const ray& rayIn, const hitRecord& rec, vec3& attenuation, ray& scattered) const
//	{
//		vec3 target = rec.p + rec.normal + randomInUnitSphere();
//		scattered = ray(rec.p, target - rec.p);
//		attenuation = albedo;
//		return true;
//	}
//
//	vec3 albedo;
//};

class Metal : public Material
{
public:
	__host__  __device__ Metal(const vec3& a, float f = 0) : albedo(a), fuzz(f < 1 ? f : 1) {}

	__host__  __device__ virtual bool scatter(const ray& rayIn, const HitRecord& rec, vec3& attenuation, ray& scattered) const
	{
		vec3 reflected = reflect(normalize(rayIn.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * randomInUnitSphere());
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}

	vec3 albedo;
	float fuzz;
};

class Dielectric : public Material
{
public:
	__host__  __device__ Dielectric(float ref) : refIdx(ref) {}

	__host__  __device__ virtual bool scatter(const ray& rayIn, const HitRecord& rec, vec3& attenuation, ray& scattered) const
	{
		vec3 outwardNormal;
		vec3 reflected = reflect(rayIn.direction(), rec.normal);
		attenuation = vec3(1.0f, 1.0f, 1.0f);
		float niOverNt;
		vec3 refracted;
		float cosine;
		float reflectProb;

		// 内部から出てこようとしている時
		if (dot(rayIn.direction(), rec.normal) > 0)
		{
			outwardNormal = rec.normal;
			niOverNt = refIdx;
			cosine = dot(rayIn.direction(), rec.normal) / rayIn.direction().length();
		}
		// 外部から飛んできている時
		else
		{
			outwardNormal = -rec.normal;
			niOverNt = 1.0 / refIdx;
			cosine = -dot(rayIn.direction(), rec.normal) / rayIn.direction().length();
		}

		if (isRefract(rayIn.direction(), outwardNormal, niOverNt, refracted))
		{
			reflectProb = schlick(cosine, refIdx);
			if (randomF64() < reflectProb)
			{
				scattered = ray(rec.p, reflected);
			}
			else
			{
				scattered = ray(rec.p, refracted);
			}
		}
		else
		{
			scattered = ray(rec.p, reflected);
		}

		return true;
	}

	__host__  __device__ inline static bool isRefract(const vec3& v, const vec3& n, float niOverNt, vec3& refracted)
	{
		vec3 uv = normalize(v);
		float dt = dot(uv, n);

		// スネル則を解いてる。Dはcos^2Thetaに相当し、正なら解がある。
		float D = 1.0 - niOverNt * niOverNt * (1 - dt * dt);

		// 解がある場合。屈折光を算出する。
		if (D > 0)
		{
			refracted = niOverNt * (uv - n * dt) + n * sqrt(D);
			return true;
		}

		// 全反射の場合
		return false;
	}

	__host__  __device__ inline static float schlick(float cosine, float refIdx)
	{
		float r0 = (1 - refIdx) / (1 + refIdx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}

	float refIdx;
};
