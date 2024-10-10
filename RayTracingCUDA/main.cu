#include <curand_kernel.h>
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
	constexpr f32 BaseResolution = 2000;
	constexpr u32 ScreenWidth = static_cast<u32>(1.960 * BaseResolution);
	constexpr u32 ScreenHeight = static_cast<u32>(1.080 * BaseResolution);
	constexpr u32 ScreenPixelNum = ScreenWidth * ScreenHeight;
	constexpr u32 SampleNum = 32;
	constexpr u32 Depth = 30;
	constexpr f32 MAXFLOAT = FLT_MAX;


	__managed__ vec3 renderTarget[ScreenPixelNum];
	__managed__ f32 depthTarget[ScreenPixelNum];

	__managed__ vec3 renderTargetOfSampleRay[ScreenPixelNum * SampleNum];
	__managed__ f32 depthTargetOfSampleRay[ScreenPixelNum * SampleNum];

}

struct vec4
{
	vec3 v;
	f32 w;
};

__device__ Camera camera;
__device__ Hitable* world;
__device__ curandState s[32];

__device__ void prepareObject(Hitable** list, const u32 MaxObjectNum, u32& actualObjectNum)
{
	list[actualObjectNum++] = new Sphere(vec3(0, -1000, 0), 1000, new Metal(vec3(0.3, 0.3, 0.3) * 1.0f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(-12, 1, 2), 1.0f, new Metal(vec3(0.5, 0.8, 0.3) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(-8, 1, 0), 1.0f, new Metal(vec3(1, 1, 0.2) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(-4, 1, 0), 1.0f, new Dielectric(2));
	list[actualObjectNum++] = new Sphere(vec3(0, 1, 0), 1.0f, new Metal(vec3(1, 1, 1), 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(4, 1, 0), 1.0f, new Dielectric(1.5f));
	list[actualObjectNum++] = new Sphere(vec3(4, 1, 0), -0.9f, new Dielectric(1.5f));
	list[actualObjectNum++] = new Sphere(vec3(8, 1, 0), 1.0f, new Metal(vec3(0, 1, 0.9) * 0.8f, 0.0f));
	list[actualObjectNum++] = new Sphere(vec3(12, 1, 2), 1.0f, new Metal(vec3(0, 1, 0.9) * 1.0f, 0.0f));

	const u32 BaseObjectNum = actualObjectNum;

	const s32 objectNumPerEdge = 15;
	const s32 objectRange = 10;
	const f32 distBetweenOjbects = objectRange * 2.0f / objectNumPerEdge;
	curandState sLocal;
	curand_init(0, 0, 0, &sLocal);


	for (f32 a = -objectRange; a < objectRange; a += distBetweenOjbects)
	{
		for (f32 b = -objectRange; b < objectRange; b += distBetweenOjbects)
		{
			vec3 center(a + curand_uniform(&sLocal), 0.2, b + curand_uniform(&sLocal));

			bool isDuplicated = false;
			for (u32 baseObjIndex = 0; baseObjIndex < BaseObjectNum; baseObjIndex++)
			{
				const Sphere* s = reinterpret_cast<Sphere*>(list[baseObjIndex]);
				if ((center - s->center).length() < (0.2 + s->radius))
				{
					isDuplicated = true;
					break;
				}
			}
			if (!isDuplicated)
			{
				Material* material = nullptr;
				f32 whichMaterial = curand_uniform(&sLocal);
				if (whichMaterial < 0.7f)
				{
					material = new Metal(vec3(curand_uniform(&sLocal), curand_uniform(&sLocal), curand_uniform(&sLocal)), 0);
				}
				else
				{
					material = new Dielectric(1 + 2 * curand_uniform(&sLocal));
				}
				list[actualObjectNum++] = new Sphere(center, 0.2, material);
			}
		}
	}

	printf("objectNum = %d\n", actualObjectNum);
}

__global__ void prepare()
{
	for (u32 i = 0; i < 32; i++)
	{
		curand_init(i, 0, 0, &s[i]);
	}

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

	if (world->hit(r, 0.001, MAXFLOAT, rec))
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

__device__ vec4 collectColor(const ray& r, Hitable* world)
{
	vec3 resultColor(1, 1, 1);

	ray currentRay = r;
	f32 d;
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

	vec4 v;
	v.v = resultColor;
	v.w = d;
	return v;
}

__global__ void castRayForDenoise(const u32 parentPixelIdx, const u32 parentPixelIdy, const u32 index)
{
	const u32 sampleID = threadIdx.x;
	const u32 offset = SampleNum * index + sampleID;
	if (!(sampleID < SampleNum))
	{
		return;
	}

	const f32 SampleRangeInPixel = 0.2;
	f32 x = (static_cast<f32>(parentPixelIdx) + (2 * randomF64() - 1) * SampleRangeInPixel) / static_cast<f32>(ScreenWidth - 1);
	f32 y = (static_cast<f32>(parentPixelIdy) + (2 * randomF64() - 1) * SampleRangeInPixel) / static_cast<f32>(ScreenHeight - 1);

	ray r = camera.getRay(x, y);
	vec4 resultColor = collectColor(r, world);
	renderTargetOfSampleRay[offset] = resultColor.v;
	depthTargetOfSampleRay[offset] = resultColor.w;

	if (threadIdx.x == 0)
	{
		renderTarget[index] = resultColor.v;
		depthTarget[index] = resultColor.w;
	}

}

__global__ void castRayToWorld()
{
	const u32 xid = threadIdx.x + blockDim.x * blockIdx.x;
	const u32 yid = threadIdx.y + blockDim.y * blockIdx.y;
	const u32 sampleID = threadIdx.z + blockDim.z * blockIdx.z;

	const u32 index = yid * ScreenWidth + xid;
	const u32 offset = SampleNum * index + sampleID;

	if (!(xid < ScreenWidth && yid < ScreenHeight && sampleID < SampleNum))
	{
		return;
	}

	const f32 SampleRangeInPixel = 0.01;
	f32 x = (static_cast<f32>(xid) + (2 * randomF64(sampleID) - 1) * SampleRangeInPixel) / static_cast<f32>(ScreenWidth - 1);
	f32 y = (static_cast<f32>(yid) + (2 * randomF64(sampleID) - 1) * SampleRangeInPixel) / static_cast<f32>(ScreenHeight - 1);

	ray r = camera.getRay(x, y);
	vec4 resultColor = collectColor(r, world);

	renderTargetOfSampleRay[offset] = resultColor.v;
	depthTargetOfSampleRay[offset] = resultColor.w;
}

__global__ void sumUp()
{
	const u32 xid = threadIdx.x + blockDim.x * blockIdx.x;
	const u32 yid = threadIdx.y + blockDim.y * blockIdx.y;
	const u32 index = yid * ScreenWidth + xid;

	if (!(xid < ScreenWidth && yid < ScreenHeight))
	{
		return;
	}


	vec3 color(0, 0, 0);
	f32 depth = 0;
	for (u32 sampleId = 0; sampleId < SampleNum; sampleId++)
	{
		const u32 offset = SampleNum * index + sampleId;
		color += renderTargetOfSampleRay[offset];
		depth += depthTargetOfSampleRay[offset];
	}

	color /= SampleNum;
	depth /= SampleNum;

	renderTarget[index] = color;
	depthTarget[index] = depth;
}

int main()
{
	//=======================================================
	// RayTracing
	//=======================================================

	prepare << <1, 1 >> > ();
	CHECK(cudaDeviceSynchronize());

	//レイを飛ばす
	printf("RayTracing Start\n");

	dim3 block(16, 16, 1);//スレッドブロック
	dim3 grid((ScreenWidth + block.x - 1) / block.x, (ScreenHeight + block.y - 1) / block.y, (SampleNum + block.z - 1) / block.z);
	cudaDeviceSetLimit(cudaLimitStackSize, 4096);
	castRayToWorld << <grid, block >> > ();
	CHECK(cudaDeviceSynchronize());

	printf("Finish RayTracing\n");


	sumUp << <grid, block >> > ();
	CHECK(cudaDeviceSynchronize());
	printf("Finish sumup\n");



	//=======================================================
	// 結果をファイルに出力
	//=======================================================
	std::ofstream outputFile("renderResult.ppm");
	outputFile << "P3\n" << ScreenWidth * 2 << " " << ScreenHeight << "\n255\n";
	for (s32 yid = ScreenHeight - 1; yid >= 0; yid--)
	{
		for (u32 xid = 0; xid < ScreenWidth; xid++)
		{
			const u32 index = yid * ScreenWidth + xid;
			vec3& col = renderTarget[index];
			col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
			outputFile << static_cast<s32>(255.99 * col[0]) << " " << static_cast<s32>(255.99 * col[1]) << " " << static_cast<s32>(255.99 * col[2]) << "\n";
		}

		for (u32 xid = 0; xid < ScreenWidth; xid++)
		{
			const u32 index = yid * ScreenWidth + xid;

			f32 depth = sqrt(depthTarget[index] / Depth);
			outputFile << static_cast<s32>(255.99 * depth) << " " << static_cast<s32>(255.99 * depth) << " " << static_cast<s32>(255.99 * depth) << "\n";
		}
	}
	outputFile.close();

	return 0;
}
