// test constant variable and cudaMemcpyToSymbol
#include <iostream>
#include <cuda_runtime.h>

__constant__ float dfactor;
__constant__ float dSecrets[32];

__global__ void test(float *a, int size)
{
    int idx = threadIdx.x;
    if(idx<size)
        a[idx] = dfactor;
}

int main(void)
{

    float factor=9.0f;

    cudaMemcpyToSymbol(dfactor, &factor, sizeof(float), 0, cudaMemcpyHostToDevice);

    float *da;
    float ha=0;

    std::cout << "the original value is " << ha << std::endl;
    cudaMalloc((void **)&da, sizeof(float));
    test<<<1,1>>>(da,1);
    cudaMemcpy(&ha, da, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaDeviceSynchronize();
    std::cout << "the value is now " << ha << std::endl;

    return 0;
}
