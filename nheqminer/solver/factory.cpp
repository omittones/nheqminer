#include "factory.h"
#include <thread>

#ifdef USE_CPU_TROMP
#define __AVX__
#include "../cpu_tromp/cpu_tromp.hpp"
#endif
#ifdef USE_CPU_XENONCAT
#include "../cpu_xenoncat/cpu_xenoncat.hpp"
#endif
#ifdef USE_CUDA_TROMP
#include "../cuda_tromp/cuda_tromp.hpp"
#endif
#ifdef USE_OCL_XMP
#include "../ocl_xpm/ocl_xmp.hpp"
#include "../ocl_xpm/ocl_silentarmy.hpp"
#endif

std::vector<Solver*> Factory::AllocateSolvers(
	int cpu_threads, bool use_avx2, int cuda_count,
	int* cuda_en, int* cuda_b, int* cuda_t,
	int opencl_count, int opencl_platf, int* opencl_en) {

		std::vector<Solver*> ret;
		
#if USE_CUDA_TROMP
		for (int i = 0; i < cuda_count; ++i)
		{
		    auto context = new cuda_tromp(0, cuda_en[i]);
			if (cuda_b[i] > 0)
				context->blocks = cuda_b[i];
			if (cuda_t[i] > 0)
				context->threadsperblock = cuda_t[i];
			ret.push_back(context);
		}
#endif	
#ifdef USE_OCL_XMP
		for (int i = 0; i < opencl_count; ++i)
		{
			auto context = new ocl_silentarmy(opencl_platf, opencl_en[i]);
			// todo: save local&global work size
			ret.push_back(context);
		}
#endif
#if USE_CPU_TROMP
		if (cpu_threads < 0) {
			cpu_threads = std::thread::hardware_concurrency();
			if (cpu_threads < 1)
				cpu_threads = 1;
			else if (ret.size() > 0)
				--cpu_threads; // decrease number of threads if there are GPU workers
		}

		for (int i = 0; i < cpu_threads; ++i)
		{
			auto context = new cpu_tromp();
			context->use_opt = use_avx2;
			ret.push_back(context);
		}
#endif
		
		return ret;
}