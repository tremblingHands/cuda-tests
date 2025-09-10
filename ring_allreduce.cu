// ring_allreduce.cu
// nvcc -O2 -arch=native -o ring_allreduce ring_allreduce.cu
// ./ring_allreduce [num_elements_per_gpu]
// Example: ./ring_allreduce 1048576

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1); \
    } \
  } while (0)

__global__ void add_kernel(float* dst, const float* src, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < N; i += stride) {
    dst[i] += src[i];
  }
}

__global__ void copy_kernel(float* dst, const float* src, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < N; i += stride) {
    dst[i] = src[i];
  }
}

int main(int argc, char** argv) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    fprintf(stderr, "No CUDA devices found.\n");
    return 1;
  }
  int p = device_count;

  size_t elems_per_gpu = 1<<20; // default 1M floats
  if (argc >= 2) elems_per_gpu = strtoull(argv[1], nullptr, 10);

  printf("Ring AllReduce on %d GPUs, %zu elements per GPU (float)\n", p, elems_per_gpu);

  // compute total per GPU buffer size (we will split buffer into p chunks)
  size_t N = elems_per_gpu;

  // compute chunk sizes (distribute remainder)
  std::vector<size_t> chunk_size(p);
  std::vector<size_t> chunk_offset(p);
  size_t base = N / p;
  size_t rem = N % p;
  size_t offset = 0;
  size_t max_chunk = 0;
  for (int i = 0; i < p; ++i) {
    chunk_size[i] = base + (i < (int)rem ? 1 : 0);
    chunk_offset[i] = offset;
    offset += chunk_size[i];
    if (chunk_size[i] > max_chunk) max_chunk = chunk_size[i];
  }
  assert(offset == N);

  // allocate per-device buffers, temp recv buffer, and stream
  std::vector<float*> dev_data(p, nullptr);
  std::vector<float*> dev_tmp(p, nullptr);
  std::vector<cudaStream_t> streams(p);
  std::vector<int> canAccessPeer(p * p, 0);

  // enable peer access where possible
  for (int dev = 0; dev < p; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaStreamCreate(&streams[dev]));

    // main data buffer per device
    CUDA_CHECK(cudaMalloc(&dev_data[dev], N * sizeof(float)));
    // temporary buffer big enough for largest chunk
    CUDA_CHECK(cudaMalloc(&dev_tmp[dev], max_chunk * sizeof(float)));

    // init dev_data with some values (unique per device so we can verify)
    // fill with value = dev (as float) for simplicity
    std::vector<float> host_init(N);
    for (size_t i = 0; i < N; ++i) host_init[i] = float(dev + 1); // dev+1 to avoid zero
    CUDA_CHECK(cudaMemcpy(dev_data[dev], host_init.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  }

  for (int a = 0; a < p; ++a) {
    for (int b = 0; b < p; ++b) {
      if (a == b) { canAccessPeer[a*p + b] = 1; continue; }
      int can = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can, a, b));
      canAccessPeer[a*p + b] = can;
      if (can) {
        CUDA_CHECK(cudaSetDevice(a));
        // try to enable peer access (ignore if already enabled)
        cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
          fprintf(stderr, "Warning: enable peer access %d -> %d failed: %s\n", a, b, cudaGetErrorString(e));
        }
      }
    }
  }

  // Helper lambda to do peer copy (dst on dstDev, src on srcDev)
  auto peer_copy_async = [&](int dstDev, float* dstPtr, int srcDev, const float* srcPtr, size_t bytes, cudaStream_t dstStream) {
    if (srcDev == dstDev) {
      // same device: device-to-device copy
      CUDA_CHECK(cudaSetDevice(dstDev));
      CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, bytes, cudaMemcpyDeviceToDevice, dstStream));
    } else if (canAccessPeer[srcDev*p + dstDev]) {
      // src can access dst peer? we will call cudaMemcpyPeerAsync
      // Note: cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, ...)
      CUDA_CHECK(cudaMemcpyPeerAsync(dstPtr, dstDev, srcPtr, srcDev, bytes, dstStream));
    } else if (canAccessPeer[dstDev*p + srcDev]) {
      // dst can access src -- still cudaMemcpyPeerAsync works
      CUDA_CHECK(cudaMemcpyPeerAsync(dstPtr, dstDev, srcPtr, srcDev, bytes, dstStream));
    } else {
      // fallback: host staging (slower)
      std::vector<char> hostbuf(bytes);
      CUDA_CHECK(cudaSetDevice(srcDev));
      CUDA_CHECK(cudaMemcpy(hostbuf.data(), srcPtr, bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaSetDevice(dstDev));
      CUDA_CHECK(cudaMemcpyAsync(dstPtr, hostbuf.data(), bytes, cudaMemcpyHostToDevice, dstStream));
    }
  };

  // ---------------------------------------------------------
  // Phase 1: Reduce-Scatter (p steps)
  // After this phase, each device 'r' holds the reduced chunk with index == r
  // Algorithm (classic ring):
  //  for s=0..p-1:
  //    send_idx = (r - s + p) % p
  //    recv_idx = (r - s - 1 + p) % p
  //    send block send_idx to next, receive recv_idx from prev, and reduce into local chunk recv_idx
  // ---------------------------------------------------------
  printf("Starting reduce-scatter...\n");
  for (int step = 0; step < p; ++step) {
    for (int r = 0; r < p; ++r) {
      int next = (r + 1) % p;
      int prev = (r - 1 + p) % p;
      int send_idx = (r - step + p) % p;
      int recv_idx = (r - step - 1 + p) % p;

      float* send_ptr = dev_data[r] + chunk_offset[send_idx];
      float* dst_tmp_on_next = dev_tmp[next]; // will receive to next->tmp
      size_t bytes = chunk_size[send_idx] * sizeof(float);

      // issue peer copy from (r, send_ptr) -> (next, dev_tmp[next])
      peer_copy_async(next, dst_tmp_on_next, r, send_ptr, bytes, streams[next]);

      // schedule reduction kernel on next: dev_data[next][offset(recv_idx)] += dev_tmp[next]
      // But ensure ordering: copy is on streams[next], kernel also enqueue on streams[next]
      CUDA_CHECK(cudaSetDevice(next));

      size_t elems = chunk_size[send_idx];
      int threads = 256;
      int blocks = (elems + threads - 1) / threads;
      // reduce: dst = dev_data[next] + chunk_offset[recv_idx]
      float* dst_chunk_ptr = dev_data[next] + chunk_offset[recv_idx];
      // call kernel on streams[next] - it will wait until copy completes on that stream
      add_kernel<<<std::max(1, blocks), threads, 0, streams[next]>>>(dst_chunk_ptr, dst_tmp_on_next, elems);
      // Note: if recv_idx points to a region not yet initialized meaningfully, adding is fine (initial values included)
    }

    // After scheduling all device ops for this step, we can (optionally) synchronize per-step.
    // Not strictly necessary, but keeping for simplicity.
    for (int d = 0; d < p; ++d) {
      CUDA_CHECK(cudaSetDevice(d));
      CUDA_CHECK(cudaStreamSynchronize(streams[d]));
    }
  }

  // ---------------------------------------------------------
  // Phase 2: Allgather (p-1 steps)
  // Each device initially holds block index == r (reduced).
  // For s=0..p-2:
  //   send_idx = (r - s + p) % p   // the block currently held by this process
  //   recv_idx = (r - s - 1 + p) % p
  //   send block send_idx to next; next stores it at recv_idx
  // After p-1 steps each process has all blocks.
  // ---------------------------------------------------------
  printf("Starting allgather...\n");
  // current block that device r holds and will send first is block == r
  std::vector<int> curr_block(p);
  for (int r = 0; r < p; ++r) curr_block[r] = r;

  for (int step = 0; step < p - 1; ++step) {
    for (int r = 0; r < p; ++r) {
      int next = (r + 1) % p;
      int recv_idx = (r - step - 1 + p) % p;
      int send_idx = curr_block[r]; // block index in dev_data[r] to send

      float* send_ptr = dev_data[r] + chunk_offset[send_idx];
      float* dst_ptr_on_next = dev_data[next] + chunk_offset[recv_idx];
      size_t bytes = chunk_size[send_idx] * sizeof(float);

      // copy block to next at proper offset
      peer_copy_async(next, dst_ptr_on_next, r, send_ptr, bytes, streams[next]);
    }

    // After scheduling copies, synchronize per-device and update curr_block
    for (int d = 0; d < p; ++d) {
      CUDA_CHECK(cudaSetDevice(d));
      CUDA_CHECK(cudaStreamSynchronize(streams[d]));
    }

    // update curr_block: after sending, each process's current block becomes (curr_block - 1)
    for (int r = 0; r < p; ++r) {
      curr_block[r] = (curr_block[r] - 1 + p) % p;
    }
  }

  // final sync
  for (int d = 0; d < p; ++d) {
    CUDA_CHECK(cudaSetDevice(d));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Validate result on host for correctness:
  // Each final element should be sum of initial values across GPUs.
  // We initialized dev 'd' to (d+1) for all positions, so sum should be sum_{i=1..p} i = p*(p+1)/2
  float expected = 0.0f;
  for (int d = 0; d < p; ++d) expected += float(d + 1);

  printf("Validating results on host...\n");
  bool ok = true;
  std::vector<float> host_check(N);
  for (int d = 0; d < p; ++d) {
    CUDA_CHECK(cudaSetDevice(d));
    CUDA_CHECK(cudaMemcpy(host_check.data(), dev_data[d], N * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < N; ++i) {
      // allow small fp error
      if (fabs(host_check[i] - expected) > 1e-3f) {
        fprintf(stderr, "Validation failed on device %d at element %zu: got %f expected %f\n",
                d, i, host_check[i], expected);
        ok = false;
        break;
      }
    }
    if (!ok) break;
  }
  if (ok) printf("AllReduce result correct! Expected = %f\n", expected);

  // cleanup
  for (int d = 0; d < p; ++d) {
    CUDA_CHECK(cudaSetDevice(d));
    CUDA_CHECK(cudaFree(dev_data[d]));
    CUDA_CHECK(cudaFree(dev_tmp[d]));
    CUDA_CHECK(cudaStreamDestroy(streams[d]));
  }

  return ok ? 0 : 2;
}

