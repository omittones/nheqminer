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
protected:
	int platf_id;
	int dev_id;

public:

	Solver(int platfId, int devId)
		:platf_id(platfId), dev_id(devId)	
	{
	}

	virtual std::string getdevinfo() = 0;
	virtual int getcount() = 0;
	virtual void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version) = 0;
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
