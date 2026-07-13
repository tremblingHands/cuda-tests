#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#define PAGE_SHIFT 12
#define PAGE_SIZE (1UL << PAGE_SHIFT)
#define PAGEMAP_ENTRY_SIZE 8

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

enum StageId {
    STAGE_INIT = 0,
    STAGE_ALLOC_VRAM,
    STAGE_ALLOC_HOST,
    STAGE_MEMSET,
    STAGE_WARMUP,
    STAGE_HOST_LAYOUT,
    STAGE_BENCHMARK,
    STAGE_FREE_HOST,
    STAGE_FREE_VRAM,
    STAGE_DONE,
    STAGE_COUNT
};

struct StageInfo {
    StageId id;
    const char* name;
    const char* title;
    const char* cudaApi;
    const char* dmesgTags;
    const char* driverBehavior;
};

static const StageInfo kStages[] = {
    {STAGE_INIT,
     "init",
     "Program start / CUDA context lazy init",
     "(first CUDA API triggers driver init)",
     "[D2H-TRACE] (context attach, no memory alloc yet)",
     "Driver opens /dev/nvidia*, creates CUDA context and binds GPU on first "
     "CUDA call. No user memory allocated yet."},

    {STAGE_ALLOC_VRAM,
     "alloc_vram",
     "Allocate GPU device memory (cudaMalloc)",
     "cudaMalloc(&d_ptr, size)",
     "[ALLOC-VRAM] [MAP-GMMU]",
     "RM allocates FB/VIDMEM physical pages, builds MEMORY_DESCRIPTOR, and "
     "installs GMMU page tables so d_ptr becomes a GPU virtual address."},

    {STAGE_ALLOC_HOST,
     "alloc_host",
     "Allocate pinned host memory (cudaMallocHost)",
     "cudaMallocHost(&h_ptr, size)",
     "[ALLOC-HOST]",
     "Driver allocates and pins CPU DRAM (SYSMEM), records IOVA/DMA addresses "
     "in pte_array, and mmap's h_ptr into the process. GPU CE can DMA directly."},

    {STAGE_MEMSET,
     "memset",
     "Initialize device memory (cudaMemset)",
     "cudaMemset(d_ptr, pattern, size)",
     "[OP-MEMSET]",
     "GPU Copy Engine (or user channel) fills VRAM. No new user-visible "
     "allocation; operates on existing d_ptr backing store."},

    {STAGE_WARMUP,
     "warmup",
     "Warm-up D2H transfer (cudaMemcpy)",
     "cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost)",
     "[OP-D2H] (RM path if visible)",
     "CE reads VRAM over PCIe/NVLink and writes pinned SYSMEM. First copy may "
     "also pay one-time TLB/DMA setup cost. Often submitted from libcuda user "
     "channel; RM [OP-D2H] tags appear only on internal CE path."},

    {STAGE_HOST_LAYOUT,
     "host_layout",
     "Inspect host virtual/physical layout",
     "va2pa(h_ptr) via /proc/self/pagemap",
     "(userspace inspection, no new driver alloc)",
     "Shows CPU VA->PA mapping of pinned host buffer. Confirms pages are "
     "resident and not swapped; useful with NUMA local/remote setups."},

    {STAGE_BENCHMARK,
     "benchmark",
     "Timed D2H benchmark loop",
     "cudaEvent + N x cudaMemcpy(D2H)",
     "[OP-D2H] (if RM path active)",
     "Repeated VRAM->pinned SYSMEM DMA on the same buffers. Steady-state "
     "throughput reflects CE + PCIe bandwidth (and NUMA placement if remote)."},

    {STAGE_FREE_HOST,
     "free_host",
     "Release pinned host memory (cudaFreeHost)",
     "cudaFreeHost(h_ptr)",
     "[FREE] [ALLOC-HOST scope]",
     "Unmaps user VA, tears down IOMMU DMA mappings, unpins and frees SYSMEM "
     "pages (nv_free_pages / osFreePagesInternal)."},

    {STAGE_FREE_VRAM,
     "free_vram",
     "Release device memory (cudaFree)",
     "cudaFree(d_ptr)",
     "[FREE] [ALLOC-VRAM scope]",
     "Frees FB/VIDMEM pages, removes GMMU mappings, and destroys RM memory "
     "handles for d_ptr."},

    {STAGE_DONE,
     "done",
     "Program exit",
     "(none)",
     "(context teardown may follow process exit)",
     "CUDA context and driver state are torn down when the process exits."},
};

struct Config {
    float dataSizeGB = 1.0f;
    int numIterations = 10;
    bool interactive = false;    // -i: pause after every stage
    bool pauseBeforeBenchmark = false; // -p: pause before timed benchmark cudaMemcpy
    bool showHostLayout = false; // -H: print host VA/PA layout (va2pa)
    bool captureDmesg = false;   // -d: embed incremental dmesg at stage boundaries
    const char* outputPath = nullptr; // -o: tee merged log to file
};

class TeeStreambuf : public std::streambuf {
public:
    TeeStreambuf(std::streambuf* primary, std::streambuf* secondary)
        : outPrimary(primary), outSecondary(secondary)
    {
    }

protected:
    int overflow(int c) override
    {
        if (c == traits_type::eof()) {
            return traits_type::not_eof(c);
        }

        if (outPrimary->sputc(static_cast<char>(c)) == traits_type::eof()) {
            return traits_type::eof();
        }
        if (outSecondary->sputc(static_cast<char>(c)) == traits_type::eof()) {
            return traits_type::eof();
        }
        return c;
    }

    int sync() override
    {
        outPrimary->pubsync();
        outSecondary->pubsync();
        return 0;
    }

private:
    std::streambuf* outPrimary;
    std::streambuf* outSecondary;
};

static size_t g_dmesg_line_offset = 0;
static const char* g_last_stage_name = "start";

static uint64_t va2pa(uint64_t va)
{
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        perror("open pagemap");
        return UINT64_MAX;
    }

    uint64_t va_page = va & ~(PAGE_SIZE - 1);
    uint64_t offset = (va_page / PAGE_SIZE) * PAGEMAP_ENTRY_SIZE;

    if (lseek(fd, offset, SEEK_SET) == (off_t)-1) {
        perror("lseek");
        close(fd);
        return UINT64_MAX;
    }

    uint64_t entry;
    if (read(fd, &entry, PAGEMAP_ENTRY_SIZE) != PAGEMAP_ENTRY_SIZE) {
        perror("read pagemap");
        close(fd);
        return UINT64_MAX;
    }
    close(fd);

    if (!(entry & (1ULL << 63))) {
        fprintf(stderr, "page not present (swap/unmapped)\n");
        return UINT64_MAX;
    }

    uint64_t pfn = entry & ((1ULL << 55) - 1);
    return (pfn << PAGE_SHIFT) | (va & (PAGE_SIZE - 1));
}

static void printUserTimestamp()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        std::cout << "[USER-TIME] (unavailable)\n";
        return;
    }

    struct tm localTm;
    localtime_r(&ts.tv_sec, &localTm);

    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &localTm);
    std::cout << "[USER-TIME] " << buf << "."
              << (ts.tv_nsec / 1000000) << "\n";
}

static size_t countDmesgLines(FILE* fp)
{
    size_t lines = 0;
    char buf[8192];

    while (fgets(buf, sizeof(buf), fp) != nullptr) {
        ++lines;
    }
    return lines;
}

static FILE* openDmesgPipe()
{
    FILE* fp = popen("dmesg -T 2>/dev/null", "r");
    if (fp != nullptr) {
        return fp;
    }
    return popen("dmesg 2>/dev/null", "r");
}

static void initDmesgCapture()
{
    FILE* fp = openDmesgPipe();
    if (fp == nullptr) {
        std::cerr << "[DMESG] warning: failed to run dmesg at startup "
                     "(try: sudo ./test_d2h_debug ...)\n";
        g_dmesg_line_offset = 0;
        return;
    }

    g_dmesg_line_offset = countDmesgLines(fp);
    pclose(fp);
    std::cout << "[DMESG] baseline set to " << g_dmesg_line_offset
              << " existing kernel log line(s); new lines captured per stage\n";
}

static void dumpDmesgDelta()
{
    FILE* fp = openDmesgPipe();
    if (fp == nullptr) {
        std::cerr << "[DMESG] failed to run dmesg (need root/CAP_SYSLOG?)\n";
        return;
    }

    std::cout << "[DMESG:since_" << g_last_stage_name << "]\n";

    size_t line = 0;
    size_t printed = 0;
    char buf[8192];
    while (fgets(buf, sizeof(buf), fp) != nullptr) {
        ++line;
        if (line > g_dmesg_line_offset) {
            std::cout << buf;
            ++printed;
        }
    }

    pclose(fp);
    g_dmesg_line_offset = line;

    std::cout << "[DMESG:end] " << printed << " new line(s)\n";
}

static void printUsage(const char* programName)
{
    std::cout <<
        "Usage: " << programName << " [options]\n\n"
        "Options:\n"
        "  -s <GB>     Data size in GB (default: 1.0)\n"
        "  -n <count>  Timed D2H iterations (default: 10)\n"
        "  -i          Interactive: pause after every stage (default: no pause)\n"
        "  -p          Pause before timed benchmark cudaMemcpy (after warmup)\n"
        "  -d          Capture full dmesg delta at each stage boundary\n"
        "  -o <file>   Tee merged user+dmesg output to file (works with -d)\n"
        "  -H          Print host VA/PA layout via /proc/self/pagemap (default: off)\n"
        "  -h          Show this help\n\n"
        "Stages (boundaries printed when -i or -d is set):\n";
    for (const StageInfo& stage : kStages) {
        std::cout << "  " << stage.name << " : " << stage.title << "\n";
    }
    std::cout <<
        "\nExamples:\n"
        "  " << programName << " -i -d\n"
        "      # step through stages; dmesg merged inline (run with sudo)\n"
        "  " << programName << " -d -o merged.log\n"
        "      # non-interactive merged log\n"
        "  " << programName << " -s 4 -n 10 -i -d -o d2h_4g.log\n"
        "  " << programName << " -s 4 -d -p -o d2h_4g.log\n"
        "      # pause before benchmark; align GPU trace with dmesg\n\n"
        "Note: reading dmesg usually requires root:\n"
        "  sudo ./test_d2h_debug -i -d -o merged.log\n";
}

static bool parseArguments(int argc, char** argv, Config& cfg)
{
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "-i") == 0) {
            cfg.interactive = true;
        } else if (strcmp(argv[i], "-p") == 0) {
            cfg.pauseBeforeBenchmark = true;
        } else if (strcmp(argv[i], "-d") == 0) {
            cfg.captureDmesg = true;
        } else if (strcmp(argv[i], "-H") == 0) {
            cfg.showHostLayout = true;
        } else if (strcmp(argv[i], "-s") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: missing value for -s\n";
                return false;
            }
            cfg.dataSizeGB = static_cast<float>(std::atof(argv[++i]));
            if (cfg.dataSizeGB <= 0) {
                std::cerr << "Error: data size must be positive\n";
                return false;
            }
        } else if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: missing value for -n\n";
                return false;
            }
            cfg.numIterations = std::atoi(argv[++i]);
            if (cfg.numIterations <= 0) {
                std::cerr << "Error: iterations must be positive\n";
                return false;
            }
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: missing value for -o\n";
                return false;
            }
            cfg.outputPath = argv[++i];
        } else {
            std::cerr << "Error: unknown option '" << argv[i] << "'\n";
            return false;
        }
    }
    return true;
}

static void waitForUserContinue(const Config& cfg, const char* label,
                                void* d_ptr, void* h_ptr, size_t dataSize)
{
    if (cfg.captureDmesg) {
        printUserTimestamp();
        dumpDmesgDelta();
        g_last_stage_name = label;
    }

    std::cout << "\n========================================\n"
              << "[PAUSE:" << label << "]\n"
              << "  d_ptr = " << d_ptr << "\n"
              << "  h_ptr = " << h_ptr << "\n"
              << "  size  = " << dataSize << " bytes\n"
              << "========================================\n"
              << "Press Enter to start timed cudaMemcpy...";
    std::cout.flush();

    if (getchar() == EOF) {
        std::cerr << "stdin closed, exiting.\n";
        exit(EXIT_FAILURE);
    }
}

static void stageBoundary(const Config& cfg, StageId id)
{
    const StageInfo& stage = kStages[id];
    const bool showBanner = cfg.interactive || cfg.captureDmesg;

    if (!showBanner && !cfg.captureDmesg) {
        return;
    }

    if (cfg.captureDmesg) {
        printUserTimestamp();
        dumpDmesgDelta();
    }

    if (showBanner) {
        std::cout << "\n========================================\n"
                  << "[STAGE:" << stage.name << "]\n"
                  << "  Title   : " << stage.title << "\n"
                  << "  CUDA API: " << stage.cudaApi << "\n"
                  << "  dmesg   : " << stage.dmesgTags << "\n"
                  << "  Driver  : " << stage.driverBehavior << "\n"
                  << "========================================\n";
    }

    if (cfg.captureDmesg) {
        g_last_stage_name = stage.name;
    }

    if (cfg.interactive) {
        std::cout << "Press Enter to continue...";
        std::cout.flush();

        if (getchar() == EOF) {
            std::cerr << "stdin closed, exiting.\n";
            exit(EXIT_FAILURE);
        }
    }
}

static void printHostLayout(void* h_ptr, size_t dataSize)
{
    uint64_t base_va = reinterpret_cast<uint64_t>(h_ptr);
    uint64_t pa = va2pa(base_va);
    uint64_t end_va = base_va + static_cast<uint64_t>(dataSize) - 1;
    uint64_t trigger_va = base_va + static_cast<uint64_t>(4ULL << 30);
    if (trigger_va >= end_va) {
        trigger_va = base_va;
    }

    uint64_t end_pa = va2pa(end_va);
    uint64_t trigger_pa = va2pa(trigger_va);

    std::cout << "Host buffer layout:\n";
    if (pa != UINT64_MAX) {
        std::cout << "  base   va=" << h_ptr << " pa=0x" << std::hex << pa << std::dec << "\n";
    } else {
        std::cout << "  base   va=" << h_ptr << " pa=(unavailable)\n";
    }
    if (end_pa != UINT64_MAX) {
        std::cout << "  end    va=0x" << std::hex << end_va << " pa=0x" << end_pa << std::dec << "\n";
    }
    if (trigger_pa != UINT64_MAX) {
        std::cout << "  sample va=0x" << std::hex << trigger_va << " pa=0x" << trigger_pa << std::dec << "\n";
    }
}

int main(int argc, char** argv)
{
    Config cfg;
    if (!parseArguments(argc, argv, cfg)) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    std::ofstream logFile;
    std::streambuf* coutBackup = nullptr;
    TeeStreambuf* teeCout = nullptr;

    if (cfg.outputPath != nullptr) {
        logFile.open(cfg.outputPath, std::ios::out | std::ios::trunc);
        if (!logFile.is_open()) {
            std::cerr << "Error: cannot open output file '" << cfg.outputPath << "'\n";
            return EXIT_FAILURE;
        }
        coutBackup = std::cout.rdbuf();
        teeCout = new TeeStreambuf(coutBackup, logFile.rdbuf());
        std::cout.rdbuf(teeCout);
    }

    if (cfg.captureDmesg) {
        initDmesgCapture();
    }

    size_t dataSize = static_cast<size_t>(cfg.dataSizeGB * (1LL << 30));

    std::cout << "D2H debug benchmark configuration:\n"
              << "  Data size    : " << cfg.dataSizeGB << " GB (" << dataSize << " bytes)\n"
              << "  Iterations   : " << cfg.numIterations << "\n"
              << "  Interactive  : " << (cfg.interactive ? "yes (-i)" : "no") << "\n"
              << "  Pause bench  : " << (cfg.pauseBeforeBenchmark ? "yes (-p)" : "no") << "\n"
              << "  Dmesg capture: " << (cfg.captureDmesg ? "yes (-d)" : "no") << "\n"
              << "  Output file  : " << (cfg.outputPath ? cfg.outputPath : "(stdout only)") << "\n"
              << "  Host layout  : " << (cfg.showHostLayout ? "yes (-H)" : "no") << "\n"
              << "  Memory type  : pinned host + device VRAM\n"
              << "  Direction    : Device -> Host\n";

    stageBoundary(cfg, STAGE_INIT);

    void* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, dataSize));
    std::cout << "d_ptr = " << d_ptr << "\n";
    stageBoundary(cfg, STAGE_ALLOC_VRAM);

    void* h_ptr = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_ptr, dataSize));
    std::cout << "h_ptr = " << h_ptr << "\n";
    stageBoundary(cfg, STAGE_ALLOC_HOST);

    CHECK_CUDA(cudaMemset(d_ptr, 0xAA, dataSize));
    stageBoundary(cfg, STAGE_MEMSET);

    CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, dataSize, cudaMemcpyDeviceToHost));
    stageBoundary(cfg, STAGE_WARMUP);

    if (cfg.showHostLayout) {
        printHostLayout(h_ptr, dataSize);
        stageBoundary(cfg, STAGE_HOST_LAYOUT);
    }

    if (cfg.pauseBeforeBenchmark) {
        waitForUserContinue(cfg, "before_benchmark", d_ptr, h_ptr, dataSize);
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < cfg.numIterations; ++i) {
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, dataSize, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float totalGB = (dataSize * cfg.numIterations) / static_cast<float>(1 << 30);
    float totalSeconds = elapsed_ms / 1000.0f;
    float throughput = totalSeconds > 0.0f ? (totalGB / totalSeconds) : 0.0f;

    std::cout << "\nResults:\n"
              << "  Total data transferred: " << totalGB << " GB\n"
              << "  Total time: " << totalSeconds << " s\n"
              << "  Throughput: " << throughput << " GB/s\n";
    stageBoundary(cfg, STAGE_BENCHMARK);

    CHECK_CUDA(cudaFreeHost(h_ptr));
    stageBoundary(cfg, STAGE_FREE_HOST);

    CHECK_CUDA(cudaFree(d_ptr));
    stageBoundary(cfg, STAGE_FREE_VRAM);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    stageBoundary(cfg, STAGE_DONE);

    if (cfg.captureDmesg) {
        printUserTimestamp();
        dumpDmesgDelta();
    }

    if (teeCout != nullptr) {
        std::cout.rdbuf(coutBackup);
        delete teeCout;
    }

    return EXIT_SUCCESS;
}
