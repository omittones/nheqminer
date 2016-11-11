#pragma once

#ifdef WIN32
#ifdef _LIB
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX
#endif
#endif

#include "solver/solver.h"

struct sa_cuda_context;

struct DLL_PREFIX cuda_sa_solver : Solver
{
private:
	std::string m_gpu_name;
	std::string m_version;
	int device_id;
	sa_cuda_context* context;

public:

	static int getCount();
	static void getDevice(int deviceId, std::string& gpuName, int& smCount, std::string& version);

	int threadsperblock;
	int blocks;

	cuda_sa_solver(int dev_id);
	std::string getname() { return "CUDA-SILENTARMY"; }
	std::string getdevinfo();
	void start();
	void stop();
	void solve(
		const char *header,
		unsigned int header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);
};