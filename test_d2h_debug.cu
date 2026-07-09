#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <fcntl.h>
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
    bool interactive = false;   // -i: pause after every stage
    bool showHostLayout = false; // -H: print host VA/PA layout (va2pa)
};

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

static void printUsage(const char* programName)
{
    std::cout <<
        "Usage: " << programName << " [options]\n\n"
        "Options:\n"
        "  -s <GB>     Data size in GB (default: 1.0)\n"
        "  -n <count>  Timed D2H iterations (default: 10)\n"
        "  -i          Interactive: pause after every stage (default: no pause)\n"
        "  -H          Print host VA/PA layout via /proc/self/pagemap (default: off)\n"
        "  -h          Show this help\n\n"
        "Stages (paused when -i is set):\n";
    for (const StageInfo& stage : kStages) {
        std::cout << "  " << stage.name << " : " << stage.title << "\n";
    }
    std::cout <<
        "\nExamples:\n"
        "  " << programName << " -i              # step through all stages\n"
        "  " << programName << " -H -i           # include host_layout stage\n"
        "  " << programName << " -s 0.5 -n 1     # run without pauses\n\n"
        "Tip: in another terminal run:\n"
        "  sudo dmesg -w | grep D2H-TRACE\n";
}

static bool parseArguments(int argc, char** argv, Config& cfg)
{
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "-i") == 0) {
            cfg.interactive = true;
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
        } else {
            std::cerr << "Error: unknown option '" << argv[i] << "'\n";
            return false;
        }
    }
    return true;
}

static void stagePause(bool interactive, StageId id)
{
    if (!interactive) {
        return;
    }

    const StageInfo& stage = kStages[id];
    std::cout << "\n========================================\n"
              << "[STAGE:" << stage.name << "] (inspect dmesg, then continue)\n"
              << "  Title   : " << stage.title << "\n"
              << "  CUDA API: " << stage.cudaApi << "\n"
              << "  dmesg   : " << stage.dmesgTags << "\n"
              << "  Driver  : " << stage.driverBehavior << "\n"
              << "========================================\n"
              << "Press Enter to continue...";
    std::cout.flush();

    if (getchar() == EOF) {
        std::cerr << "stdin closed, exiting.\n";
        exit(EXIT_FAILURE);
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

    size_t dataSize = static_cast<size_t>(cfg.dataSizeGB * (1LL << 30));

    std::cout << "D2H debug benchmark configuration:\n"
              << "  Data size    : " << cfg.dataSizeGB << " GB (" << dataSize << " bytes)\n"
              << "  Iterations   : " << cfg.numIterations << "\n"
              << "  Interactive  : " << (cfg.interactive ? "yes (-i)" : "no") << "\n"
              << "  Host layout  : " << (cfg.showHostLayout ? "yes (-H)" : "no") << "\n"
              << "  Memory type  : pinned host + device VRAM\n"
              << "  Direction    : Device -> Host\n";

    stagePause(cfg.interactive, STAGE_INIT);

    void* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, dataSize));
    std::cout << "d_ptr = " << d_ptr << "\n";
    stagePause(cfg.interactive, STAGE_ALLOC_VRAM);

    void* h_ptr = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_ptr, dataSize));
    std::cout << "h_ptr = " << h_ptr << "\n";
    stagePause(cfg.interactive, STAGE_ALLOC_HOST);

    CHECK_CUDA(cudaMemset(d_ptr, 0xAA, dataSize));
    stagePause(cfg.interactive, STAGE_MEMSET);

    CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, dataSize, cudaMemcpyDeviceToHost));
    stagePause(cfg.interactive, STAGE_WARMUP);

    if (cfg.showHostLayout) {
        printHostLayout(h_ptr, dataSize);
        stagePause(cfg.interactive, STAGE_HOST_LAYOUT);
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
    stagePause(cfg.interactive, STAGE_BENCHMARK);

    CHECK_CUDA(cudaFreeHost(h_ptr));
    stagePause(cfg.interactive, STAGE_FREE_HOST);

    CHECK_CUDA(cudaFree(d_ptr));
    stagePause(cfg.interactive, STAGE_FREE_VRAM);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    stagePause(cfg.interactive, STAGE_DONE);
    return EXIT_SUCCESS;
}
