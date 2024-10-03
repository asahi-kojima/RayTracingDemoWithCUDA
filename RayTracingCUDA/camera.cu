#include "camera.h"

static __device__ __host__ vec3 randominUnitDisk()
{
	vec3 p;
	do
	{
		p = 2.0 * vec3(randomF64(), randomF64(), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0f);

	return p;
}



Camera::Camera(f32 vfov, f32 aspect)
{
	f32 theta = vfov * M_PI / 180.0f;
	f32 halfHeight = tan(theta / 2);
	f32 halfWidth = aspect * halfHeight;

	origin = vec3(0, 0, 0);
	lowerLeftCorner = vec3(-halfWidth, -halfHeight, -1.0f);
	horizontal = vec3(2 * halfWidth, 0, 0);
	vertical = vec3(0, 2 * halfHeight, 0);
}

Camera::Camera(vec3 lookFrom, vec3 lookAt, vec3 vUp, f32 vfov, f32 aspect, f32 aperture, f32 focusDist)
{
	lensRadius = aperture / 2;

	f32 theta = vfov * M_PI / 180.0f;
	f32 halfHeight = tan(theta / 2);
	f32 halfWidth = aspect * halfHeight;

	origin = lookFrom;
	w = normalize(lookFrom - lookAt);//z
	u = normalize(cross(vUp, w));//x
	v = cross(w, u);//y
	lowerLeftCorner = origin - focusDist * w - focusDist * halfWidth * u - focusDist * halfHeight * v;
	horizontal = focusDist * 2 * halfWidth * u;
	vertical = focusDist * 2 * halfHeight * v;
}

__host__ __device__ ray Camera::getRay(f32 s, f32 t)
{
	vec3 rd = lensRadius * randominUnitDisk();
	vec3 offset = u * rd.x() + v * rd.y();
	vec3 rayOrigin = origin;// +offset;

	return ray(rayOrigin, lowerLeftCorner + s * horizontal + t * vertical - rayOrigin);
}