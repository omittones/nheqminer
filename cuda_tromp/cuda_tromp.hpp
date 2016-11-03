#ifdef _LIB
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX
#endif

#include "solver/solver.h"

struct eq_cuda_context;

struct DLL_PREFIX cuda_tromp : Solver
{
	int threadsperblock;
	int blocks;
	eq_cuda_context* context;

	std::string getname() { return "CUDA-TROMP"; }

private:
	std::string m_gpu_name;
	std::string m_version;
	int m_sm_count;

public:
	cuda_tromp(int platf_id, int dev_id);
	std::string getdevinfo();
	int getcount();
	void getinfo(int platf_id, int d_id, std::string & gpu_name, int & sm_count, std::string & version);
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