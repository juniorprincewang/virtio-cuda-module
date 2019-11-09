/*
* nvcc -arch=sm_35 -o t1034 t1034.cu
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define USECPSEC 1000000ULL

#define MAX_DELAY 30

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

#define APPRX_CLKS_PER_SEC 1000000000ULL
__global__ void delay_kernel(unsigned seconds, int *a, size_t n){
    printf("&a = %p\n", a);
    printf("a[n=%d] = %d\n", n, a[n]);
    unsigned long long dt = clock64();
    while (clock64() < (dt + (seconds*APPRX_CLKS_PER_SEC)));
}

#define CHECK(call) { \
    cudaError_t err; \
    if ( (err = (call)) != cudaSuccess) { \
        fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}



int checkResult(float *data, const int n, const float x)
{
    for (int i = 0; i < n; i++)
    {
        if (data[i] != x)
        {
            printf("Error! data[%d] = %f, ref = %f\n", i, data[i], x);
            return 0;
        }
    }

    return 1;
}

__global__ void kernel(float *g_data, int num)
{
    // float g = g_data[-num];
    printf("spy kernel\n");
    // printf("spy g_data[%d]=%f\n", -num/2, g_data[-num/2]);
}

int main()
{
    int devID=1;
    int count = 0;
    struct cudaDeviceProp props;
    float *d_a=0;
    float *h_a=0;
    dim3 block, grid;
    int num = 1 << 24;
    int nbytes = num * sizeof(float);
    //test();
    /* test case 2
    * add   cudaGetDeviceCount
            cudaGetDevice
            cudaGetDeviceProperties
    */
    devID = 0;
    CHECK(cudaSetDevice(devID));
    CHECK(cudaGetDeviceCount(&count));
    printf("cuda count=%d\n", count);
    // CHECK(cudaGetDevice(&devID));
    CHECK(cudaGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",devID, props.name, props.major, props.minor);

    // printf("num 0x%x\n", num);

    printf("sending 0x%x\n", nbytes);
    printf("allocating 0x%x\n", nbytes);
    h_a=(float*)malloc(nbytes);

    printf("initing mem\n");
    memset(h_a, 0, nbytes);
    printf("spy h_a \t=%p\n", h_a);
    // h_a[0] = 1;
    // start

    CHECK(cudaMalloc((void**)&d_a, nbytes));
    printf("spy d_a address \t= %p\n", d_a);
    CHECK(cudaMemset(d_a, 0, nbytes));
    
    // set kernel launch configuration
    block = dim3(4);
    grid  = dim3((num + block.x - 1) / block.x);

    // CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyDefault));
    kernel<<<1, 1>>>(d_a, num);
    // CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDefault));
    // end
    CHECK(cudaFree(d_a));
    free(h_a);

    /* test case 3
    * add   cudaMalloc
            cudaMemset
            cudaMemcpy
            cudaLaunch
            cudaFree
    */

    return EXIT_SUCCESS;
}
