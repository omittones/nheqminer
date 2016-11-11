#include "factory.h"
#include <thread>
#include <bitset>
#include <array>
#include "boost\log\trivial.hpp"

#include "../cpu_tromp/cpu_tromp.hpp"
#include "../cpu_xenoncat/cpu_xenoncat.hpp"
#include "../cuda_tromp/cuda_tromp.hpp"
#include "../ocl_silentarmy/ocl_silentarmy.hpp"
#include "../cuda_silentarmy/cuda_silentarmy.hpp"

void detect_AVX_and_AVX2(bool &canUseAvx1, bool &canUseAvx2)
{
	canUseAvx1 = false;
	canUseAvx2 = false;

	// Fix on Linux
	//int cpuInfo[4] = {-1};
	std::array<int, 4> cpui;
	std::vector<std::array<int, 4>> data_;
	std::bitset<32> f_1_ECX_;
	std::bitset<32> f_7_EBX_;

	// Calling __cpuid with 0x0 as the function_id argument
	// gets the number of the highest valid function ID.
	__cpuid(cpui.data(), 0);
	int nIds_ = cpui[0];

	for (int i = 0; i <= nIds_; ++i)
	{
		__cpuidex(cpui.data(), i, 0);
		data_.push_back(cpui);
	}

	if (nIds_ >= 1)
	{
		f_1_ECX_ = data_[1][2];
		canUseAvx1 = f_1_ECX_[28];
	}

	// load bitset with flags for function 0x00000007
	if (nIds_ >= 7)
	{
		f_7_EBX_ = data_[7][1];
		canUseAvx2 = f_7_EBX_[5];
	}
}

std::vector<Solver*> Factory::AllocateSolvers(
	int cpu_threads, ForceMode forceMode,
	int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t, bool use_cuda_silentarmy,
	int opencl_count, int* opencl_en, int* opencl_t) {

	std::vector<Solver*> ret;

	bool canAvx1, canAvx2;
	detect_AVX_and_AVX2(canAvx1, canAvx2);

	BOOST_LOG_TRIVIAL(info) << "Using SSE2: YES";
	BOOST_LOG_TRIVIAL(info) << "Using AVX: " << (canAvx1 ? "YES" : "NO");
	BOOST_LOG_TRIVIAL(info) << "Using AVX2: " << (canAvx2 ? "YES" : "NO");

	for (int i = 0; i < cuda_count; ++i)
	{
		if (use_cuda_silentarmy) {
			auto context = new cuda_sa_solver(cuda_en[i]);
			if (cuda_b[i] > 0)
				context->blocks = cuda_b[i];
			if (cuda_t[i] > 0)
				context->threadsperblock = cuda_t[i];
			ret.push_back(context);
		}
		else {
			auto context = new cuda_tromp(0, cuda_en[i]);
			if (cuda_b[i] > 0)
				context->blocks = cuda_b[i];
			if (cuda_t[i] > 0)
				context->threadsperblock = cuda_t[i];
			ret.push_back(context);
		}
	}

	for (int i = 0; i < opencl_count; ++i)
	{
		auto context = new ocl_silentarmy(opencl_en[i], opencl_t[i]);
		ret.push_back(context);
	}

	if (cpu_threads < 0) {
		cpu_threads = std::thread::hardware_concurrency();
		if (cpu_threads < 1)
			cpu_threads = 1;
		else if (ret.size() > 0)
			--cpu_threads;
	}

	if (forceMode == ForceMode::NONE) {
		forceMode = ForceMode::CPU_TROMP;
		if (canAvx1)
			forceMode = ForceMode::CPU_XENON_AVX;
		if (canAvx2)
			forceMode = ForceMode::CPU_XENON_AVX2;
	}
	
	for (int i = 0; i < cpu_threads; ++i)
	{
		if (forceMode == ForceMode::CPU_TROMP) {
			auto context = new cpu_tromp();
			ret.push_back(context);
		}
		else if (forceMode == ForceMode::CPU_XENON_AVX) {
			auto context = new cpu_xenoncat(false);
			ret.push_back(context);
		}
		else if (forceMode == ForceMode::CPU_XENON_AVX2) {
			auto context = new cpu_xenoncat(true);
			ret.push_back(context);
		}
	}

	return ret;
}