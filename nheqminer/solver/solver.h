#pragma once

#ifdef _LIB
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX
#endif

#include <string>
#include <functional>
#include <vector>

struct DLL_PREFIX Solver
{
public:

	Solver()
	{
	}

	virtual std::string getdevinfo() = 0;
	virtual void start() = 0;
	virtual void stop() = 0;
	
	virtual void solve(
		const char *header,
		unsigned int header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef) = 0;

	virtual std::string getname() = 0;
};
