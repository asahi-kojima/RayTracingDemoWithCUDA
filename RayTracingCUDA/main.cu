#include <float.h>
#include "Sphere.h"
#include "camera.h"
#include <fstream>
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

namespace
{
	constexpr f32 BaseSize = 20;
	constexpr u32 ScreenWidth = static_cast<u32>(16 * BaseSize);
	constexpr u32 ScreenHeight = static_cast<u32>(9 * BaseSize);
	constexpr u32 ScreenPixelNum = ScreenWidth * ScreenHeight;
	__managed__ vec3 renderTarget[ScreenPixelNum];

	constexpr u32 SampleNum = 50;
	constexpr u32 Depth = 3;

	constexpr f32 MAXFLOAT = FLT_MAX;
}


__device__ void prepareObject(Hitable** list, const u32 MaxObjectNum, u32& actualObjectNum)
{
	list[actualObjectNum++] = new Sphere(vec3(0, -1000, 0), 1000, new Metal(vec3(0.3, 0.3, 0.3) * 1.0f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(-12, 1, 2), 1.0f, new Metal(vec3(0.5, 0.8, 0.3) * 0.8f, 0.3f));
	list[actualObjectNum++] = new Sphere(vec3(-8, 1, 0), 1.0f, new Metal(vec3(1, 1, 0.2) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(-4, 1, 0), 1.0f, new Metal(vec3(0.5, 0.8, 0.3) * 0.8f, 0.3f));
	list[actualObjectNum++] = new Sphere(vec3(0, 1, 0), 1.0f, new Metal(vec3(1, 1, 1), 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(4, 1, 0), 1.0f, new Metal(vec3(0.5, 0.8, 0.3) * 0.8f, 0.3f));
	list[actualObjectNum++] = new Sphere(vec3(8, 1, 0), 1.0f, new Metal(vec3(0, 1, 0.9) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(12, 1, 2), 1.0f, new Metal(vec3(0, 1, 0.9) * 1.0f, 0.0f));
}

__device__ vec3 getColor(ray r, Hitable* world, const s32 depth, u32 xid, u32 yid)
{
	HitRecord rec;
	bool isHit = world->hit(r, 0.001, MAXFLOAT, rec);

	if (isHit)
	{
		ray scattered;
		vec3 attenuation;

		if (depth >= 0 && rec.pMaterial->scatter(r, rec, attenuation, scattered))
		{
			vec3 resultColor = getColor(scattered, world, depth - 1, xid, yid);
			return attenuation * resultColor;
		}
		else
		{
			return vec3(0, 0, 0);
		}
	}
	else
	{
		vec3 unitDirection = normalize(r.direction());
		float t = 0.5f * (unitDirection.y() + 1.0f);
		return vec3(1.0f, 1.0f, 1.0f)* (1.0f - t) + vec3(0.5f, 0.7f, 1.0f) * t;
	}
}

__device__ vec3 getColorFromRay(ray r, Hitable* world, const s32 depth, u32 xid, u32 yid)
{
	HitRecord rec;
	bool isHit = world->hit(r, 0.001, MAXFLOAT, rec);

	if (isHit)
	{
		ray scattered;
		vec3 attenuation;

		if (depth >= 0 && rec.pMaterial->scatter(r, rec, attenuation, scattered))
		{
			vec3 resultColor = getColor(scattered, world, depth - 1, xid, yid);
			return attenuation * resultColor;
		}
		else
		{
			return vec3(0, 0, 0);
		}
	}
	else
	{
		vec3 unitDirection = normalize(r.direction());
		float t = 0.5f * (unitDirection.y() + 1.0f);
		return vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + vec3(0.5f, 0.7f, 1.0f) * t;
	}
}

__device__ vec3 collectColor(ray r, Hitable* world, const s32 depth, u32 xid, u32 yid)
{
	vec3 resultColor(1,1,1);

	for (u32 loop = 0; loop < Depth; loop++)
	{
		vec3 colorFromThisRay;



		resultColor *= colorFromThisRay;
	}

	return resultColor;
}

__global__ void castRayToWorld(Camera camera, Hitable* world)
{
	const u32 xid = threadIdx.x + blockDim.x * blockIdx.x;
	const u32 yid = threadIdx.y + blockDim.y * blockIdx.y;
	const u32 index = yid * ScreenWidth + xid;

	if (!(xid < ScreenWidth && yid < ScreenHeight))
	{
		return;
	}

	vec3 color;
	for (u32 sampleNo = 0; sampleNo < SampleNum; sampleNo++)
	{
		f32 x = (static_cast<f32>(xid) + (srandomF64() * 0.5))/ (ScreenWidth - 1);
		f32 y = (static_cast<f32>(yid) + (srandomF64() * 0.5))/ (ScreenHeight - 1);

		ray r = camera.getRay(x, y);
		vec3 resultColor = getColor(r, world, Depth, xid, yid);
		color += resultColor;
	}


	color /= SampleNum;

	renderTarget[index] = color;
}

__global__ void startRayTracing()
{
	constexpr u32 MaxObjectNum = 100;
	u32 actualObjectNum = 0;
	vec3 objectPos[MaxObjectNum];

	Hitable** list = new Hitable * [MaxObjectNum + 1];
	////事前準備
	prepareObject(list, MaxObjectNum, actualObjectNum);

	//カメラの準備
	vec3 lookFrom(13, 2, 5);
	vec3 lookAt(0, 0, 0);
	Camera camera(lookFrom, lookAt, vec3(0, 1, 0), 20, f32(ScreenWidth) / f32(ScreenHeight), 0.01, (lookFrom - lookAt).length());

	Hitable* world = new HitableList(list, actualObjectNum);
	//レイを飛ばす
	dim3 block(16, 16);//スレッドブロック
	dim3 grid((ScreenWidth + block.x - 1) / block.x, (ScreenHeight + block.y - 1) / block.y);
	castRayToWorld << <grid, block >> > (camera, world);
}


int main()
{
	startRayTracing << <1, 1 >> > ();
	CHECK(cudaDeviceSynchronize());
	printf("Finish RayTracing\n");
	std::ofstream outputFile("renderResult.ppm");
	outputFile << "P3\n" << ScreenWidth << " " << ScreenHeight << "\n255\n";
	for (s32 yid = ScreenHeight - 1; yid >= 0; yid--)
	{
		for (u32 xid = 0; xid < ScreenWidth; xid++)
		{
			const u32 index = yid * ScreenWidth + xid;
			vec3& col = renderTarget[index];
			col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
			outputFile << int(255.99 * col[0]) << " " << int(255.99 * col[1]) << " " << int(255.99 * col[2]) << "\n";
		}
	}
	outputFile.close();

	return 0;
}
