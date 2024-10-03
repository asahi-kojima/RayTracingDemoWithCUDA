﻿#include <curand_kernel.h>
#include <fstream>
#include <float.h>
#include "Sphere.h"
#include "camera.h"
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
	constexpr f32 BaseResolution = 1500;
	constexpr u32 ScreenWidth = static_cast<u32>(1.960 * BaseResolution);
	constexpr u32 ScreenHeight = static_cast<u32>(1.080 * BaseResolution);
	constexpr u32 ScreenPixelNum = ScreenWidth * ScreenHeight;
	__managed__ vec3 renderTarget[ScreenPixelNum];

	constexpr u32 SampleNum = 30;
	constexpr u32 Depth = 30;

	constexpr f32 MAXFLOAT = FLT_MAX;
}


__device__ Camera camera;
__device__ Hitable* world;

__device__ void prepareObject(Hitable** list, const u32 MaxObjectNum, u32& actualObjectNum)
{
	list[actualObjectNum++] = new Sphere(vec3(0, -1000, 0), 1000, new Metal(vec3(0.3, 0.3, 0.3) * 1.0f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(-12, 1, 2), 1.0f, new Metal(vec3(0.5, 0.8, 0.3) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(-8, 1, 0), 1.0f, new Metal(vec3(1, 1, 0.2) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(-4, 1, 0), 1.0f, new Metal(vec3(0.5, 0.8, 0.3) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(0, 1, 0), 1.0f, new Metal(vec3(1, 1, 1), 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(4, 1, 0), 1.0f, new Metal(vec3(0.5, 0.8, 0.3) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(8, 1, 0), 1.0f, new Metal(vec3(0, 1, 0.9) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(12, 1, 2), 1.0f, new Metal(vec3(0, 1, 0.9) * 1.0f, 0.0f));

	const s32 objectMaxNum = 10;
	curandState s;
	curand_init(0, objectMaxNum * objectMaxNum * 20, 0, &s);

	for (s32 a = -objectMaxNum; a < objectMaxNum; a++)
	{
		for (s32 b = -objectMaxNum; b < objectMaxNum; b++)
		{
			vec3 center(a + curand_uniform(&s), 0.2, b + curand_uniform(&s));

			//if ((center - vec3(4, 0.2f, 0)).length() > 0.9f)
			{
				list[actualObjectNum++] = new Sphere(center, 0.2, new Metal(vec3(curand_uniform(&s), curand_uniform(&s), curand_uniform(&s)), curand_uniform(&s)));
			}
		}
	}

	printf("objectNum = %d\n", actualObjectNum);
}

__global__ void prepare()
{
	vec3 lookFrom(13, 2, 5);
	vec3 lookAt(0, 0, 0);
	camera = Camera(lookFrom, lookAt, vec3(0, 1, 0), 20, static_cast<f32>(ScreenWidth) / static_cast<f32>(ScreenHeight), 0.01, (lookFrom - lookAt).length());


	constexpr u32 MaxObjectNum = 500;
	u32 actualObjectNum = 0;
	Hitable** list = new Hitable * [MaxObjectNum + 1];
	prepareObject(list, MaxObjectNum, actualObjectNum);
	world = new HitableList(list, actualObjectNum);
}


__device__ bool getColorFromRay(ray* pRay, Hitable* world, const s32 depth, vec3* pV)
{
	ray r = *pRay;
	HitRecord rec;
	bool isHit = world->hit(r, 0.001, MAXFLOAT, rec);

	if (isHit)
	{
		ray scattered;
		vec3 attenuation;

		if (depth < Depth && rec.pMaterial->scatter(r, rec, attenuation, scattered))
		{
			*pV = attenuation;
			*pRay = scattered;
			return false;
		}
		else
		{
			*pV = vec3(0, 0, 0);
			return true;
		}
	}
	else
	{
		vec3 unitDirection = normalize(r.direction());
		f32 t = 0.5f * (unitDirection.y() + 1.0f);
		*pV = vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + vec3(0.5f, 0.7f, 1.0f) * t;
		return true;
	}
}

__device__ vec3 collectColor(const ray& r, Hitable* world)
{
	vec3 resultColor(1, 1, 1);

	ray currentRay = r;
	f32 d = 0;
	for (u32 depth = 0; depth < Depth; depth++)
	{
		d = depth;
		vec3 colorFromThisRay;
		bool isRayTerminated = getColorFromRay(&currentRay, world, depth, &colorFromThisRay);
		resultColor *= colorFromThisRay;

		if (isRayTerminated)
		{
			break;
		}
	}
	f32 df = d / Depth;
	//return vec3(df, df, df);

	return resultColor;
}

__global__ void castRayToWorld()
{
	const u32 xid = threadIdx.x + blockDim.x * blockIdx.x;
	const u32 yid = threadIdx.y + blockDim.y * blockIdx.y;
	const u32 index = yid * ScreenWidth + xid;

	if (!(xid < ScreenWidth && yid < ScreenHeight))
	{
		return;
	}

	curandState s;
	curand_init(index, 0, 0, &s);


	vec3 color(0, 0, 0);

	for (u32 sampleNo = 0; sampleNo < SampleNum; sampleNo++)
	{
		f32 x = (static_cast<f32>(xid) + (2 * curand_uniform(&s) - 1) * 0.5) / static_cast<f32>(ScreenWidth - 1);
		f32 y = (static_cast<f32>(yid) + (2 * curand_uniform(&s) - 1) * 0.5) / static_cast<f32>(ScreenHeight - 1);

		ray r = camera.getRay(x, y);
		vec3 resultColor = collectColor(r, world);
		color += resultColor;
	}


	color /= SampleNum;

	renderTarget[index] = color;
}



int main()
{
	for (u32 i = 0; i < ScreenPixelNum; i++)
	{
		renderTarget[i] = vec3(0.0f, 0.0f, 0.0f);
	}
	//=======================================================
	// RayTracing
	//=======================================================
	printf("RayTracing Start\n");

	printf("Call prepare<<<1, 1>>>\n");
	prepare << <1, 1 >> > ();
	CHECK(cudaDeviceSynchronize());

	//レイを飛ばす
	dim3 block(16, 16);//スレッドブロック
	dim3 grid((ScreenWidth + block.x - 1) / block.x, (ScreenHeight + block.y - 1) / block.y);


	printf("Call castRayToWorld<<<(%d, %d), (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);
	castRayToWorld << <grid, block >> > ();
	CHECK(cudaDeviceSynchronize());


	printf("Finish RayTracing\n");



	//=======================================================
	// 結果をファイルに出力
	//=======================================================
	std::ofstream outputFile("renderResult.ppm");
	outputFile << "P3\n" << ScreenWidth << " " << ScreenHeight << "\n255\n";
	for (s32 yid = ScreenHeight - 1; yid >= 0; yid--)
	{
		for (u32 xid = 0; xid < ScreenWidth; xid++)
		{
			const u32 index = yid * ScreenWidth + xid;
			vec3& col = renderTarget[index];
			col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
			outputFile << static_cast<s32>(255.99 * col[0]) << " " << static_cast<s32>(255.99 * col[1]) << " " << static_cast<s32>(255.99 * col[2]) << "\n";
		}
	}
	outputFile.close();

	return 0;
}
