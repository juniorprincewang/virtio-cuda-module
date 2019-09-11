#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#define NUM_KERNELS 8
#define NUM_STREAMS 9 // 8 clock blocks + 1 reduction sum

// Run for at least the specified amount of ticks
__global__ void clockBlockKernel(clock_t *output_d, clock_t clockCount) {
  clock_t startClock = clock();
  clock_t clockOffset = 0;

  while (clockOffset < clockCount) {
    clockOffset = clock() - startClock;
  }

  output_d[0] = clockOffset;
}

// Single-warp (32 threads) reduction sum of previous kernels
__global__ void sumKernel(clock_t *clocks_d, int N) {
  __shared__ clock_t clocks_s[32];
  clock_t sum = 0;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    sum += clocks_d[i];
  }

  clocks_s[threadIdx.x] = sum;
  __syncthreads();

  for (int i = 16; i > 0; i /= 2) {
    if (threadIdx.x < i) {
      clocks_s[threadIdx.x] += clocks_s[threadIdx.x + i];
    }
    __syncthreads();
  }
  clocks_d[0] = clocks_s[0];
}

int main() {
  float kernelTime = 10;
  float elapsedTime;
  int device = 0;

  // Find device clock rate to calculate number of cycles (for 10ms)
  cudaDeviceProp deviceProp;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&deviceProp, device);
  int clockRate = deviceProp.clockRate;
  printf("Device clock rate: %.3f GHz\n", (float)clockRate/1000000);

  // Check card supports concurrency
  if (deviceProp.concurrentKernels == 0) {
    printf("GPU does not support concurrent kernel execution\n");
    printf("CUDA kernel runs will be serialised\n");
  }

  // Allocate host and device memory
  clock_t *a_h = 0;
  cudaMallocHost((void**)&a_h, NUM_KERNELS * sizeof(clock_t));
  clock_t *a_d = 0;
  cudaMalloc((void**)&a_d, NUM_KERNELS * sizeof(clock_t));

  // Create array of streams and array of events
  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&(streams[i]));
  }

  cudaEvent_t kernelEvent[NUM_KERNELS];
  for (int i = 0; i < NUM_KERNELS; i++) {
    cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming);
  }

  // Create CUDA events
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  // Calculate number of cycles for each kernel to run for
  clock_t totalClocks = 0;
  clock_t timeClocks = (clock_t)(kernelTime * clockRate);

  // Start timer and launch kernels concurrently
  cudaEventRecord(startEvent, 0);
  for (int i = 0; i < NUM_KERNELS; ++i) {
    clockBlockKernel<<<1, 1, 0, streams[i]>>>(&a_d[i], timeClocks);
    totalClocks += timeClocks;
    cudaEventRecord(kernelEvent[i], streams[i]);
    cudaStreamWaitEvent(streams[NUM_STREAMS-1], kernelEvent[i], 0);
  }

  // Sum individual kernel times using single warp reduction sum
  sumKernel<<<1, 32, 0, streams[NUM_STREAMS-1]>>>(a_d, NUM_KERNELS);
  cudaMemcpyAsync(a_h, a_d, sizeof(clock_t), cudaMemcpyDeviceToHost, streams[NUM_STREAMS-1]);
  bool correctResult = (a_h[0] < totalClocks);

  // Stop timer
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

  printf("Expected time for serial execution of %d kernels = %.3fms\n", NUM_KERNELS, NUM_KERNELS * kernelTime);
  printf("Expected time for concurrent execution of %d kernels = %.3fms\n", NUM_KERNELS, kernelTime);
  printf("Measured time for sample = %.3fms\n", elapsedTime);

  // Clean up
  for (int i = 0; i < NUM_KERNELS; i++) {
    cudaStreamDestroy(streams[i]);
    cudaEventDestroy(kernelEvent[i]);
  }
  cudaFree(a_d);
  cudaFreeHost(a_h);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  cudaDeviceReset();

  // Check result
  if (!correctResult) {
    printf("FAILED!\n");
    exit(EXIT_FAILURE);
  }
  printf("PASSED\n");
  return 0;
}
