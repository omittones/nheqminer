#include "SolverStub.h"

#ifdef USE_CPU_TROMP
#define __AVX__
#include "../cpu_tromp/cpu_tromp.hpp"
#else
using cpu_tromp = SolverStub;
#endif
#ifdef USE_CPU_XENONCAT
#include "../cpu_xenoncat/cpu_xenoncat.hpp"
#else
using cpu_xenoncat = SolverStub1;
#endif
#ifdef USE_CUDA_TROMP
#include "../cuda_tromp/cuda_tromp.hpp"
#else
using cuda_tromp = SolverStub;
#endif
#ifdef USE_OCL_XMP
#include "../ocl_xpm/ocl_xmp.hpp"
#include "../ocl_xpm/ocl_silentarmy.hpp"
using open_cl_solver = ocl_silentarmy;
#else
using open_cl_solver = SolverStub;
#endif

//
//ZcashMiner::ZcashMiner(
//	int cpu_threads, std::vector<*Solver> cudaSolvers,
//	int opencl_count, int opencl_platf, int* opencl_en)
//	: minerThreads{ nullptr }
//{
//	m_isActive = false;
//	nThreads = 0;
//
//	for (int i = 0; i < cuda_count; ++i)
//	{
//		CUDASolver* context = new CUDASolver(0, cuda_en[i]);
//		if (cuda_b[i] > 0)
//			context->blocks = cuda_b[i];
//		if (cuda_t[i] > 0)
//			context->threadsperblock = cuda_t[i];
//
//		cuda_contexts.push_back(context);
//	}
//	nThreads += cuda_contexts.size();
//
//
//	for (int i = 0; i < opencl_count; ++i)
//	{
//		OPENCLSolver* context = new OPENCLSolver(opencl_platf, opencl_en[i]);
//		// todo: save local&global work size
//		opencl_contexts.push_back(context);
//	}
//	nThreads += opencl_contexts.size();
//
//
//
//	if (cpu_threads < 0) {
//		cpu_threads = std::thread::hardware_concurrency();
//		if (cpu_threads < 1) cpu_threads = 1;
//		else if (cuda_contexts.size() + opencl_contexts.size() > 0) --cpu_threads; // decrease number of threads if there are GPU workers
//	}
//
//
//	for (int i = 0; i < cpu_threads; ++i)
//	{
//		CPUSolver* context = new CPUSolver();
//		context->use_opt = use_avx2;
//		cpu_contexts.push_back(context);
//	}
//	nThreads += cpu_contexts.size();
//
//
//	//	nThreads = cpu_contexts.size() + cuda_contexts.size() + opencl_contexts.size();
//}