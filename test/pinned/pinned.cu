#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define checkCudaErrors(call) { \
  cudaError_t err; \
  if ( (err = (call)) != cudaSuccess) { \
    fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
        __FILE__, __LINE__); \
  } \
}

/* A very simple kernel function */
__global__ void kernel(int *g_data, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
}

int checkResult(int *data, const int n, const int x)
{
    for (int i = 0; i < n; i++)
    {
        if (data[i] != x)
        {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return 0;
        }
    }

    return 1;
}

void test_hostregister()
{
    int * h_p1;
    int * h_p2;
    int * h_p3;
    int n=1<<10;

    printf("Allocate data size 0x%x\n", n);
    h_p1 = (int *)malloc(n);
    printf("host p1 is at %p\n", h_p1);
    h_p2 = (int *)malloc(n);
    printf("host p2 is at %p\n", h_p2);
    h_p3 = (int *)malloc(n);
    printf("host p3 is at %p\n", h_p3);
    checkCudaErrors(
        cudaHostRegister(h_p1, n, cudaHostRegisterDefault) );
    checkCudaErrors(
        cudaHostRegister(h_p2, n, cudaHostRegisterDefault) );
    checkCudaErrors(
        cudaHostRegister(h_p3, n, cudaHostRegisterDefault) );
    checkCudaErrors( cudaHostUnregister(h_p1) );
    checkCudaErrors( cudaHostUnregister(h_p2) );
    checkCudaErrors( cudaHostUnregister(h_p3) );
    free(h_p1);
    free(h_p2);
    free(h_p3);
}

void test_hostalloc()
{
    int ns = 1<<20;
    int data_size = ns * sizeof(int);
    int * host_p;
    checkCudaErrors(
        cudaHostAlloc((void**)&host_p, data_size, cudaHostAllocDefault) );
    checkCudaErrors( cudaFreeHost(host_p) );
}

int main(void) {  
    int ns = 1<<18;
    // int ns = 1<<3;
    int data_size = ns * sizeof(int);
    int * host_p;
    int * host_p2;
    int * host_result;
    int * dev_p;
    int value = 10;
    dim3 block, grid;
    size_t page_size = 0;
    
    page_size = sysconf(_SC_PAGE_SIZE);
    printf("Page size is 0x%lx\n", page_size);
    printf("Allocate data size 0x%x\n", data_size);
    /* Allocate host_p as pinned memory */
    checkCudaErrors(
        cudaHostAlloc((void**)&host_p, data_size, cudaHostAllocDefault) );
    printf("host p is at %p\n", host_p);
    checkCudaErrors(
        cudaHostAlloc((void**)&host_p2, data_size, cudaHostAllocDefault) );
    printf("host p2 is at %p\n", host_p2);
    /* Allocate host_result as pinned memory */
    checkCudaErrors(
        cudaHostAlloc((void**)&host_result, data_size, cudaHostAllocDefault) );
    printf("host result is at %p\n", host_result);
    /* Allocate dev_p on the device global memory */
    checkCudaErrors(cudaMalloc((void**)&dev_p, data_size) );
    printf("dev p is at %p\n", dev_p);

    /* Initialise host_p*/
    memset(host_p, 0, data_size);
    /* Transfer data to the device host_p .. dev_p */
    checkCudaErrors(
        cudaMemcpy(dev_p, host_p, data_size, cudaMemcpyHostToDevice) );
      
    /* Now launch the kernel... */
    // set kernel launch configuration
    block = dim3(16);
    grid  = dim3((ns + block.x - 1) / block.x);
    kernel<<<grid, block>>>(dev_p, value);
    checkCudaErrors(cudaGetLastError());
    /* Copy the result from the device back to the host */
    checkCudaErrors(
        cudaMemcpy(host_result, dev_p, data_size, cudaMemcpyDeviceToHost) );

    printf("Check if no failures, then pass.\n");
    int bFinalResults = checkResult(host_result, ns, value);
    printf("result:%s\n", bFinalResults? "PASS" : "FAILED");

    /*
    * Now free the memory!
    */
    checkCudaErrors( cudaFree(dev_p) );
    checkCudaErrors( cudaFreeHost(host_p) );
    checkCudaErrors( cudaFreeHost(host_p2) );
    checkCudaErrors( cudaFreeHost(host_result) );
    
    return 0;
}