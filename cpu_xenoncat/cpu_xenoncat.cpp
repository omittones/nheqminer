#include <iostream>
#include <functional>
#include <vector>
#include <stdint.h>
#include <cassert>

#ifdef WIN32
#include <Windows.h>
#else
#include <string.h>
#include <stdlib.h>
#endif

#include "cpu_xenoncat.hpp"

#define CONTEXT_SIZE 178033152

extern "C" 
{
	//Linkage with assembly
	//EhPrepare takes in 136 bytes of input. The remaining 4 bytes of input is fed as nonce to EhSolver.
	//EhPrepare saves the 136 bytes in context, and EhSolver can be called repeatedly with different nonce.
	void EhPrepareAVX1(void *context, void *input);
	int32_t EhSolverAVX1(void *context, uint32_t nonce);

	void EhPrepareAVX2(void *context, void *input);
	int32_t EhSolverAVX2(void *context, uint32_t nonce);
}

void cpu_xenoncat::start() {
	this->memory_alloc = malloc(CONTEXT_SIZE + 4096);
	this->memory = (void*)(((long long)this->memory_alloc + 4095) & -4096);
	// todo: improve memory; LOCKED_PAGES ?
}

void cpu_xenoncat::stop()
{
	free(this->memory_alloc);
}

int cpu_xenoncat::getcount()
{
	return 0;
}

void cpu_xenoncat::getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version)
{
}

void cpu_xenoncat::solve(
	const char *header,
	unsigned int header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef)
{
	assert(header_len == 108);
	assert(nonce_len == 32);

	unsigned char context[140];
	int32_t i, numsolutions;

	memcpy(context, header, 108);
	memcpy(context + 108, nonce, 32);

	if (this->use_avx2)
	{
		EhPrepareAVX2(this->memory, (void *)context);
		numsolutions = EhSolverAVX2(this->memory, *(uint32_t *)(context + 136));
	}
	else
	{
		EhPrepareAVX1(this->memory, (void *)context);
		numsolutions = EhSolverAVX1(this->memory, *(uint32_t *)(context + 136));
	}

	for (i = 0; i < numsolutions; i++)
	{
		solutionf(std::vector<uint32_t>(0), 1344, (unsigned char*)this->memory + (1344 * i));

		if (cancelf()) return;

		//validBlock(validBlockData, (unsigned char*)context + (1344 * i));
	}

	hashdonef();
}