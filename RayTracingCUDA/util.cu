#pragma once
#include <curand_kernel.h>
#include "util.h"


extern __device__ curandState s[32];


__host__  __device__ f64 randomF64()
{
#if __CUDA_ARCH__ >= 500
	const f32 rnd = curand_uniform(&s[(threadIdx.x + blockIdx.x) % 32]);
	return rnd;
#elif !defined(__CUDA_ARCH__)
	return rand() / (RAND_MAX + 1.0f);
#endif
}

__host__  __device__ f64 randomF64(const u32 offset)
{
#if __CUDA_ARCH__ >= 500
	const f32 rnd = curand_uniform(&s[offset % 32]);
	return rnd;
#elif !defined(__CUDA_ARCH__)
	return rand() / (RAND_MAX + 1.0f);
#endif
}

__host__  __device__ f64 srandomF64()
{
	return 2 * randomF64() - 1;
}