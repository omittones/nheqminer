#ifdef _LIB
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX
#endif

#include "solver/solver.h"

#if defined(__AVX__)

#define CPU_TROMP_NAME "CPU-TROMP-AVX"

#else

#define CPU_TROMP_NAME "CPU-TROMP-SSE2"

#endif

struct DLL_PREFIX cpu_tromp : Solver
{
public:

	cpu_tromp() : Solver(0, 0) {
	}

	std::string getdevinfo() { return ""; }

	void start();

	void stop();

	void solve(const char *header,
		unsigned int header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);

	std::string getname() { return CPU_TROMP_NAME; }

	int getcount();
	
	void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);
};
