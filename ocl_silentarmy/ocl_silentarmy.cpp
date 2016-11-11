#include "ocl_silentarmy.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <fstream>
#include "CL/opencl.h"

#include "remote/param.h"
#include "remote/blake.h"
#include "remote/solver.hpp"
#include "remote/logging.hpp"

#define COLLISION_BIT_LENGTH (PARAM_N / (PARAM_K+1))
#define COLLISION_BYTE_LENGTH ((COLLISION_BIT_LENGTH+7)/8)
#define PROOFSIZE (1u<<PARAM_K)
#define COMPRESSED_PROOFSIZE ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)))

ocl_silentarmy::ocl_silentarmy(int gpuId, size_t noThreadsPerBlock) :
	noThreadsPerBlock(noThreadsPerBlock) {

	this->ctx = nullptr;

	auto plats = scan_platforms();
	if (plats.size() <= gpuId)
		fatal("Unknown gpu id!");

	devId = plats[gpuId].devId;
	devName = plats[gpuId].name;
}

ocl_silentarmy::~ocl_silentarmy() {
	assert(this->ctx == nullptr);
}

std::string ocl_silentarmy::getdevinfo() {
	return "GPU_ID(" + devName + ")";
}

void ocl_silentarmy::printInfo() {
	printf("Listing OpenCL supported devices: \n");
	auto plats = scan_platforms();
	for (auto i = 0; i < plats.size(); i++) {
		printf("    %d. %s\n", i, plats[i].name.c_str());
	}
}

void ocl_silentarmy::start() {

	if (this->ctx != nullptr)
		throw std::exception("Solver already started!");

	this->ctx = new solver_context_t();
	
	setup_context(*this->ctx, (cl_device_id) devId);
}

void ocl_silentarmy::stop() {

	if (this->ctx) {
		destroy_context(*this->ctx);
		delete this->ctx;
		this->ctx = nullptr;
	}
	else {
		throw std::exception("Solver already stopped!");
	}
}

static void compress(uint8_t *out, uint32_t *inputs, uint32_t n)
{
	uint32_t byte_pos = 0;
	int32_t bits_left = PREFIX + 1;
	uint8_t x = 0;
	uint8_t x_bits_used = 0;
	uint8_t *pOut = out;
	while (byte_pos < n)
	{
		if (bits_left >= 8 - x_bits_used)
		{
			x |= inputs[byte_pos] >> (bits_left - 8 + x_bits_used);
			bits_left -= 8 - x_bits_used;
			x_bits_used = 8;
		}
		else if (bits_left > 0)
		{
			uint32_t mask = ~(-1 << (8 - x_bits_used));
			mask = ((~mask) >> bits_left) & mask;
			x |= (inputs[byte_pos] << (8 - x_bits_used - bits_left)) & mask;
			x_bits_used += bits_left;
			bits_left = 0;
		}
		else if (bits_left <= 0)
		{
			assert(!bits_left);
			byte_pos++;
			bits_left = PREFIX + 1;
		}
		if (x_bits_used == 8)
		{
			*pOut++ = x;
			x = x_bits_used = 0;
		}
	}
}

void ocl_silentarmy::solve(
	const char *header,
	unsigned int header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef) {

	assert(this->ctx != nullptr);
	assert(header_len == ZCASH_BLOCK_HEADER_LEN - ZCASH_NONCE_LEN);
	assert(nonce_len == ZCASH_NONCE_LEN);
	unsigned char context[ZCASH_BLOCK_HEADER_LEN];
	memset(context, 0, ZCASH_BLOCK_HEADER_LEN);
	memcpy(context, header, header_len);
	memcpy(context + header_len, nonce, nonce_len);

	debug("\nSolving nonce %s\n", s_hexdump(nonce, nonce_len));

	auto miner = this->ctx;

	clFlush(miner->queue);

	if (cancelf())
		return;

	auto sols = solve_equihash(*miner, this->noThreadsPerBlock, context, ZCASH_BLOCK_HEADER_LEN);

	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	uint32_t noValidSols = 0;
	for (unsigned sol_i = 0; sol_i < sols->nr; sol_i++) {

		if (verify_sol(sols, sol_i)) {
			noValidSols++;
			compress(proof, (uint32_t *)(sols->values[sol_i]), 1 << PARAM_K);
			solutionf(std::vector<uint32_t>(0), 1344, proof);
		}

		if (cancelf())
			break;
	}

	debug("Nonce %s: %d sol%s\n", s_hexdump(nonce, nonce_len), noValidSols, noValidSols == 1 ? "" : "s");
	debug("Stats: %d likely invalids\n", sols->likely_invalids);

	free(sols);

	hashdonef();
}
