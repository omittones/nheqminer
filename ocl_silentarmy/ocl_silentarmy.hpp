#ifdef WIN32
#ifdef _LIB
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX
#endif
#endif

#include "solver/solver.h"

// remove after
#include <string>
#include <functional>
#include <vector>
#include <cstdint>

struct solver_context_t;

struct DLL_PREFIX ocl_silentarmy : Solver
{

private:
	int gpu_id;
	solver_context_t* ctx;

public:
	static void printInfo();

	ocl_silentarmy(int gpu_id);
	virtual ~ocl_silentarmy();

	std::string getname() { return "OCL_SILENTARMY"; }
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