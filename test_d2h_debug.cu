#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>

#define PAGE_SHIFT 12
#define PAGE_SIZE (1UL << PAGE_SHIFT)
#define PAGEMAP_ENTRY_SIZE 8

uint64_t va2pa(uint64_t va)
{
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        perror("open pagemap");
        return -1;
    }

    // 计算该虚拟页在pagemap中的偏移
    uint64_t va_page = va & ~(PAGE_SIZE - 1);
    uint64_t offset = (va_page / PAGE_SIZE) * PAGEMAP_ENTRY_SIZE;

    if (lseek(fd, offset, SEEK_SET) == (off_t)-1) {
        perror("lseek");
        close(fd);
        return -1;
    }

    uint64_t entry;
    if (read(fd, &entry, PAGEMAP_ENTRY_SIZE) != PAGEMAP_ENTRY_SIZE) {
        perror("read pagemap");
        close(fd);
        return -1;
    }
    close(fd);

    // 判断页面是否驻留内存
    if (!(entry & (1ULL << 63))) {
        fprintf(stderr, "page not present (swap/unmapped)\n");
        return -1;
    }

    uint64_t pfn = entry & ((1ULL << 55) - 1);
    uint64_t pa = (pfn << PAGE_SHIFT) | (va & (PAGE_SIZE - 1));
    return pa;
}


#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// 显示用法信息
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -s <size>      Data size in GB (default: 1.0)\n";
    std::cout << "  -n <iterations> Number of iterations (default: 10)\n";
    std::cout << "  -h             Show this help message\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " -s 2.0 -n 20    # 2GB data, 20 iterations\n";
    std::cout << "  " << programName << " -s 0.5          # 0.5GB data, 10 iterations\n";
}

// 解析命令行参数
void parseArguments(int argc, char** argv, float& dataSizeGB, int& numIterations) {
    // 默认值
    dataSizeGB = 1.0f;
    numIterations = 10;
    
    // 解析参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if (strcmp(argv[i], "-s") == 0) {
            if (i + 1 < argc) {
                dataSizeGB = std::atof(argv[++i]);
                if (dataSizeGB <= 0) {
                    std::cerr << "Error: Data size must be positive\n";
                    exit(EXIT_FAILURE);
                }
            } else {
                std::cerr << "Error: Missing value for -s option\n";
                printUsage(argv[0]);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) {
                numIterations = std::atoi(argv[++i]);
                if (numIterations <= 0) {
                    std::cerr << "Error: Number of iterations must be positive\n";
                    exit(EXIT_FAILURE);
                }
            } else {
                std::cerr << "Error: Missing value for -n option\n";
                printUsage(argv[0]);
                exit(EXIT_FAILURE);
            }
        }
        else {
            std::cerr << "Error: Unknown option '" << argv[i] << "'\n";
            printUsage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    // 解析命令行参数
    float dataSizeGB;
    int numIterations;
    parseArguments(argc, argv, dataSizeGB, numIterations);
    
    // 计算数据大小（字节）
    size_t dataSize = static_cast<size_t>(dataSizeGB * (1L << 30));
    
    std::cout << "Benchmark Configuration:\n";
    std::cout << "  Data size: " << dataSizeGB << " GB (" << dataSize << " bytes)\n";
    std::cout << "  Iterations: " << numIterations << "\n";
    std::cout << "  Memory type: Pinned Memory\n";
    std::cout << "  Direction: Device to Host\n";
    std::cout << std::endl;

    // Allocate device memory
    void* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, dataSize));

    // Allocate pinned host memory
    void* h_ptr = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_ptr, dataSize));

    // Fill device memory with dummy values
    CHECK_CUDA(cudaMemset(d_ptr, 0xAA, dataSize));

    printf("按下回车，开始warmup...\n");
    getchar();

    // Warm-up
    CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, dataSize, cudaMemcpyDeviceToHost));

    uint64_t pa = va2pa((uint64_t)h_ptr);
    uint64_t end_va = (uint64_t)h_ptr + (uint64_t)dataSize - 1;
    uint64_t trigger_va = (uint64_t)h_ptr + (uint64_t)(4 * (1L << 30));

    uint64_t end_pa = va2pa(end_va);
    uint64_t trigger_pa = va2pa(trigger_va);
    if (pa != -1) {
	std::cout << "va = " << h_ptr << ", pa = " << (void* )pa << "\n";
	std::cout << "end_va = " << (void *)end_va << ", end_pa = " << (void* )end_pa << "\n";
	std::cout << "trigger_va = " << (void *)trigger_va << ", trigger_pa = " << (void* )trigger_pa << "\n";
    }
    printf("按下回车，开始测试...\n");
    getchar();

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Start timing
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < numIterations; ++i) {
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, dataSize, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Calculate throughput
    float totalGB = (dataSize * numIterations) / (float)(1 << 30); // in GB
    float totalSeconds = elapsed_ms / 1000.0f;
    float throughput = totalGB / totalSeconds;

    std::cout << "Results:\n";
    std::cout << "  Total data transferred: " << totalGB << " GB\n";
    std::cout << "  Total time: " << totalSeconds << " seconds\n";
    std::cout << "  Throughput: " << throughput << " GB/s\n";

    printf("按下回车，结束测试...\n");
    getchar();
    // Clean up
    CHECK_CUDA(cudaFreeHost(h_ptr));
    CHECK_CUDA(cudaFree(d_ptr));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
