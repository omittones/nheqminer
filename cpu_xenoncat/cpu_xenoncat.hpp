#ifdef _LIB
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX
#endif

#include "solver/solver.h"

struct DLL_PREFIX cpu_xenoncat : Solver
{
private:
	bool use_avx2;
	void *memory_alloc, *memory;

public:

	std::string getdevinfo() { return ""; }

	cpu_xenoncat(bool use_avx2) : Solver(0,0) {
		this->use_avx2 = use_avx2;
	}

	void start();
	void stop();
	int getcount();
	void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);
	void solve(const char *header,
		unsigned int header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);

	std::string getname()
	{
		if (use_avx2)
			return "CPU-XENONCAT-AVX2";
		else
			return "CPU-XENONCAT-AVX";
	}
};