#pragma once

#include <vector>
#include "solver.h"

class Factory {
public:
	static std::vector<Solver*> AllocateSolvers(
		int cpu_threads, bool use_avx2, int cuda_count,
		int* cuda_en, int* cuda_b, int* cuda_t,
		int opencl_count, int opencl_platf, int* opencl_en);
};
