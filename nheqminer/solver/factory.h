#pragma once

#include <vector>
#include "solver.h"

enum ForceMode
{
	NONE = 0,
	CPU_TROMP = 1,
	CPU_XENON_AVX = 2,
	CPU_XENON_AVX2 = 3	
};

class Factory {
public:
	static std::vector<Solver*> AllocateSolvers(
		int cpu_threads, ForceMode forceMode,
		int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
		int opencl_count, int* opencl_en, int* opencl_t);
};
