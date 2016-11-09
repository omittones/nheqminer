#ifdef WIN32
#ifdef _LIB
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX
#endif
#endif

#include "../nheqminer/solver/solver.h"

// remove after
#include <string>
#include <functional>
#include <vector>
#include <cstdint>

struct OclContext;

struct DLL_PREFIX ocl_silentarmy : Solver
{

private:
	int blocks;
	int device_id;
	int platform_id;
	OclContext* oclc;
	unsigned threadsNum;
	unsigned wokrsize;
	bool is_init_success = false;

public:
	static int getcount();
	static void printInfo();
	static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);

	ocl_silentarmy(int platf_id, int dev_id);
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