#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << static_cast<int>(err)              \
                      << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

void printUsage(const char* program) {
    std::cout << "Usage: " << program
              << " --size <MB> --direction <0|1> [--repeat <count>]\n"
              << "  --size       Data size per transfer in MB (default 1024 = 1GB)\n"
              << "  --direction  Transfer direction: 0 = GPU0 -> GPU1, 1 = GPU1 -> GPU0\n"
              << "  --repeat     Number of repeats (default 10)\n";
}

// Parse command line args
void parseArgs(int argc, char** argv, size_t& dataSize, int& direction, int& numRepeats) {
    dataSize = 1UL << 30; // 1 GB
    direction = 0;
    numRepeats = 10;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--size") && i + 1 < argc) {
            dataSize = static_cast<size_t>(atoll(argv[++i])) << 20; // MB -> Bytes
        } else if (!strcmp(argv[i], "--direction") && i + 1 < argc) {
            direction = atoi(argv[++i]);
            if (direction != 0 && direction != 1) {
                std::cerr << "Invalid direction! Must be 0 or 1.\n";
                exit(EXIT_FAILURE);
            }
        } else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) {
            numRepeats = atoi(argv[++i]);
        } else {
            printUsage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    size_t dataSize;
    int direction;
    int numRepeats;

    parseArgs(argc, argv, dataSize, direction, numRepeats);

    // GPU IDs
    int srcDev = (direction == 0) ? 0 : 1;
    int dstDev = (direction == 0) ? 1 : 0;

    std::cout << "P2P Bandwidth Test\n";
    std::cout << "Data Size: " << (dataSize >> 20) << " MB\n";
    std::cout << "Direction: GPU" << srcDev << " -> GPU" << dstDev << "\n";
    std::cout << "Repeats: " << numRepeats << "\n";

    // Check P2P access
    int canAccessPeer = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, dstDev, srcDev));
    if (!canAccessPeer) {
        std::cerr << "ERROR: P2P not supported between GPU" << srcDev
                  << " and GPU" << dstDev << std::endl;
        return EXIT_FAILURE;
    }

    // Enable P2P
    CHECK_CUDA(cudaSetDevice(srcDev));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(dstDev, 0));
    CHECK_CUDA(cudaSetDevice(dstDev));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(srcDev, 0));

    // Allocate memory on both GPUs
    CHECK_CUDA(cudaSetDevice(srcDev));
    void* d_src = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, dataSize));

    CHECK_CUDA(cudaSetDevice(dstDev));
    void* d_dst = nullptr;
    CHECK_CUDA(cudaMalloc(&d_dst, dataSize));

    // Create multiple streams for pipeline
    const int numStreams = 8;
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Warm-up: do one P2P copy to stabilize performance
    size_t chunkSize = dataSize / numStreams;
    for (int s = 0; s < numStreams; ++s) {
        void* srcPtr = static_cast<char*>(d_src) + s * chunkSize;
        void* dstPtr = static_cast<char*>(d_dst) + s * chunkSize;
        CHECK_CUDA(cudaMemcpyPeerAsync(dstPtr, dstDev,
                                       srcPtr, srcDev,
                                       chunkSize, streams[s]));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < numRepeats; ++r) {
        for (int s = 0; s < numStreams; ++s) {
            void* srcPtr = static_cast<char*>(d_src) + s * chunkSize;
            void* dstPtr = static_cast<char*>(d_dst) + s * chunkSize;
            CHECK_CUDA(cudaMemcpyPeerAsync(dstPtr, dstDev,
                                           srcPtr, srcDev,
                                           chunkSize, streams[s]));
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));

    // Calculate bandwidth
    //float totalGB = (dataSize * numRepeats) / static_cast<float>(1UL << 30);
    //float totalSeconds = elapsedMs / 1000.0f;
    float totalGB = (dataSize * numRepeats) / (double)1e9;
    float totalSeconds = elapsedMs / 1e3;
    float bandwidth = totalGB / totalSeconds;

    std::cout << "P2P Bandwidth: " << bandwidth << " GB/s\n";

    // Cleanup
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

