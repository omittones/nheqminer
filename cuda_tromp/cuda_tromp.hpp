#ifdef WIN32
#ifdef _LIB
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX
#endif
#endif

#include "solver/solver.h"

struct eq_cuda_context;

struct DLL_PREFIX cuda_tromp : Solver
{
private:
	eq_cuda_context* context;
	std::string m_gpu_name;
	std::string m_version;
	int m_sm_count;
	int platf_id;
	int dev_id;

public:
	
	static int getCount();
	static void getDevice(int deviceId, std::string& gpuName, int& smCount, std::string& version);

	int threadsperblock;
	int blocks;

	cuda_tromp(int platf_id, int dev_id);
	std::string getname() { return "CUDA-TROMP"; }
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