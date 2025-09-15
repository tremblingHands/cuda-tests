#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

int main() {
    const size_t dataSize = 1L << 30; // 1GB
    const int numRepeats = 10;        // 重复次数，增加统计稳定性

    // ------------------------------------------------------------
    // 1. 检查至少有 2 个 GPU
    // ------------------------------------------------------------
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Error: Need at least 2 GPUs for P2P test." << std::endl;
        return EXIT_FAILURE;
    }

    int srcDev = 0;
    int dstDev = 1;
    std::cout << "Testing P2P bandwidth between GPU " << srcDev 
              << " -> GPU " << dstDev << std::endl;

    // ------------------------------------------------------------
    // 2. 检查两个 GPU 是否支持 P2P
    // ------------------------------------------------------------
    int canAccessPeer01 = 0, canAccessPeer10 = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer01, srcDev, dstDev));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer10, dstDev, srcDev));

    if (!canAccessPeer01 || !canAccessPeer10) {
        std::cerr << "Error: GPUs do not support Peer-to-Peer access!" << std::endl;
        return EXIT_FAILURE;
    }

    // 启用 Peer Access
    CHECK_CUDA(cudaSetDevice(srcDev));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(dstDev, 0));
    CHECK_CUDA(cudaSetDevice(dstDev));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(srcDev, 0));

    // ------------------------------------------------------------
    // 3. 在两个 GPU 上分配内存
    // ------------------------------------------------------------
    void* d_src = nullptr;
    void* d_dst = nullptr;

    CHECK_CUDA(cudaSetDevice(srcDev));
    CHECK_CUDA(cudaMalloc(&d_src, dataSize));
    CHECK_CUDA(cudaMemset(d_src, 0, dataSize)); // 初始化

    CHECK_CUDA(cudaSetDevice(dstDev));
    CHECK_CUDA(cudaMalloc(&d_dst, dataSize));
    CHECK_CUDA(cudaMemset(d_dst, 0, dataSize));

    // ------------------------------------------------------------
    // 4. 创建 CUDA 事件进行计时
    // ------------------------------------------------------------
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ------------------------------------------------------------
    // 5. 预热拷贝（避免首次调用干扰）
    // ------------------------------------------------------------
    CHECK_CUDA(cudaMemcpyPeer(d_dst, dstDev, d_src, srcDev, dataSize));

    // ------------------------------------------------------------
    // 6. 正式计时测试
    // ------------------------------------------------------------
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < numRepeats; ++i) {
        CHECK_CUDA(cudaMemcpyPeer(d_dst, dstDev, d_src, srcDev, dataSize));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // ------------------------------------------------------------
    // 7. 计算实际带宽
    // ------------------------------------------------------------
    float totalGB = (dataSize * numRepeats) / (float)(1 << 30); // 总传输量(GB)
    float totalSeconds = elapsed_ms / 1000.0f;
    float bandwidth = totalGB / totalSeconds;

    std::cout << "P2P Bandwidth GPU " << srcDev << " -> GPU " << dstDev
              << " : " << bandwidth << " GB/s" << std::endl;

    // ------------------------------------------------------------
    // 8. 清理资源
    // ------------------------------------------------------------
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}

