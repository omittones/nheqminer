#include <iostream>
#include <functional>
#include <vector>
#include <stdint.h>
#include <string>

#include "cuda_tromp.hpp"

struct proof;
#include "eqcuda.hpp"


cuda_tromp::cuda_tromp(int platf_id, int dev_id)
{
	this->dev_id = dev_id;
	this->platf_id = platf_id;

	cudaDeviceProp device_props;
	checkCudaErrors(cudaGetDeviceProperties(&device_props, dev_id));

	this->m_gpu_name = device_props.name;
	this->m_sm_count = device_props.multiProcessorCount;
	this->m_version = std::to_string(device_props.major) + "." + std::to_string(device_props.minor);

	if (CUDAARCH > device_props.major * 10 + device_props.minor) {
		throw std::runtime_error("GPU does not support this CUDA version!");
	}


	// todo: determine default values for various GPUs here
	threadsperblock = 64;
	blocks = m_sm_count * 7;
}

void cuda_tromp::getDevice(int deviceId, std::string& gpuName, int& smCount, std::string& version) {

	cudaDeviceProp device_props;
	checkCudaErrors(cudaGetDeviceProperties(&device_props, deviceId));

	gpuName = device_props.name;
	smCount = device_props.multiProcessorCount;
	version = std::to_string(device_props.major) + "." + std::to_string(device_props.minor);
}

std::string cuda_tromp::getdevinfo()
{
	return m_gpu_name + " (#" + std::to_string(this->dev_id) + ") BLOCKS=" +
		std::to_string(blocks) + ", THREADS=" + std::to_string(threadsperblock);
}

int cuda_tromp::getCount()
{
	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	return device_count;
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