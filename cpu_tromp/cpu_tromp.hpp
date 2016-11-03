#ifdef _USRDLL
#define DLL_CPU_TROMP __declspec(dllexport)
#else
#define DLL_CPU_TROMP
#endif

#if defined(__AVX__)

#define CPU_TROMP_NAME "CPU-TROMP-AVX"

#else

#define CPU_TROMP_NAME "CPU-TROMP-SSE2"

#endif

struct cpu_tromp
{
	std::string getdevinfo() { return ""; }

	static void start(cpu_tromp& device_context);

	static void stop(cpu_tromp& device_context);

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		cpu_tromp& device_context);

	std::string getname() { return CPU_TROMP_NAME; }

	int use_opt;
};
