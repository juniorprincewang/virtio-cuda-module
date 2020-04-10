// test constant variable and cudaMemcpyToSymbol
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

__constant__ float dfactor;
__constant__ float dSecrets[8]={0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1};

__global__ void test(float *a, int size, int value)
{
    int idx = threadIdx.x;
    if(idx<size)
        a[idx] = dfactor;
}

__global__ void test2(float *a, int size, float *b)
{
    int idx = threadIdx.x;
    if(idx<size)
        a[idx] = dfactor;
}

__global__ void test3(float *a)
{
    int idx = threadIdx.x;
    a[idx] = dfactor;
}

int main(void)
{

    float factor=9.0f;
    float h_factor = 0;
    cudaMemcpyToSymbol(dfactor, &factor, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyFromSymbol(&h_factor, dfactor, sizeof(float), 0, cudaMemcpyDeviceToHost);
    printf("host factor = %f\n", h_factor);
    return 0;
    float *da;
    float ha=0;
    
    std::cout << "the original value is " << ha << std::endl;
    cudaMalloc((void **)&da, sizeof(float));
    test<<<1,1>>>(da,1,2);
    cudaMemcpy(&ha, da, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaDeviceSynchronize();
    std::cout << "the value is now " << ha << std::endl;
    
    return 0;
}
