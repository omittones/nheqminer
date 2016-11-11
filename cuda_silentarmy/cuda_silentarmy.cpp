#include <iostream>
#include <functional>
#include <vector>
#include <stdint.h>
#include <string>

#include "cuda_silentarmy.hpp"
#include "sa_cuda_context.hpp"

cuda_sa_solver::cuda_sa_solver(int dev_id)
{
	device_id = dev_id;

	int smCount;
	getDevice(dev_id, m_gpu_name, smCount, m_version);

	// todo: determine default values for various GPUs here
	threadsperblock = 64;
	blocks = smCount * 32;
}

std::string cuda_sa_solver::getdevinfo()
{
	return m_gpu_name + " (#" + std::to_string(device_id) + ") BLOCKS=" +
		std::to_string(blocks) + ", THREADS=" + std::to_string(threadsperblock);
}

int cuda_sa_solver::getCount()
{
	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	return device_count;
}

void cuda_sa_solver::getDevice(int d_id, std::string & gpu_name, int & sm_count, std::string & version)
{
	cudaDeviceProp device_props;

	checkCudaErrors(cudaGetDeviceProperties(&device_props, d_id));

	gpu_name = device_props.name;
	sm_count = device_props.multiProcessorCount;
	version = std::to_string(device_props.major) + "." + std::to_string(device_props.minor);

}

void cuda_sa_solver::start()
{
	this->context = new sa_cuda_context(
		threadsperblock,
		blocks,
		device_id);
}

void cuda_sa_solver::stop()
{
	if (this->context)
		delete this->context;
}

void cuda_sa_solver::solve(
	const char * header,
	unsigned int header_len,
	const char * nonce,
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