#include <iostream>
#include <functional>
#include <vector>
#include <stdint.h>
#include <string>

#include "cuda_tromp.hpp"

struct proof;
#include "eqcuda.hpp"


cuda_tromp::cuda_tromp(int platf_id, int dev_id) :
	Solver(platf_id, dev_id)
{
	getinfo(0, dev_id, m_gpu_name, m_sm_count, m_version);

	// todo: determine default values for various GPUs here
	threadsperblock = 64;
	blocks = m_sm_count * 7;
}

std::string cuda_tromp::getdevinfo()
{
	return m_gpu_name + " (#" + std::to_string(this->dev_id) + ") BLOCKS=" + 
		std::to_string(blocks) + ", THREADS=" + std::to_string(threadsperblock);
}


int cuda_tromp::getcount()
{
	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	return device_count;
}

void cuda_tromp::getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version)
{
	//int runtime_version;
	//checkCudaErrors(cudaRuntimeGetVersion(&runtime_version));

	cudaDeviceProp device_props;

	checkCudaErrors(cudaGetDeviceProperties(&device_props, d_id));

	gpu_name = device_props.name;
	sm_count = device_props.multiProcessorCount;
	version = std::to_string(device_props.major) + "." + std::to_string(device_props.minor);
}

void cuda_tromp::start() 
{ 
	this->context = new eq_cuda_context(this->threadsperblock,
		this->blocks,
		this->dev_id);
}

void cuda_tromp::stop()
{
	if (this->context)
		delete this->context;
}

void cuda_tromp::solve(
	const char *header,
	unsigned int header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef)
{
	this->context->solve(
		header,
		header_len,
		nonce,
		nonce_len,
		cancelf,
		solutionf,
		hashdonef);
}