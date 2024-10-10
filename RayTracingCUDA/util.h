#pragma once
#include <iostream>
#include <random>
#include "root.h"

__host__  __device__ f64 randomF64();
__host__  __device__ f64 randomF64(const u32 offset);

__host__  __device__ f64 srandomF64();

template <typename T>
bool __host__  __device__ isInRange(T t, T min, T max)
{
	return (t >= min) && (t <= max);
}
