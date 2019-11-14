#include <stdio.h>  
#include <cuda_runtime.h>  

#define checkCudaErrors(call) { \
  cudaError_t err; \
  if ( (err = (call)) != cudaSuccess) { \
    fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
        __FILE__, __LINE__); \
  } \
}

/* A very simple kernel function */
 __global__ void kernel(int *d_var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_var[idx] += 10; 
} 
 
 int * host_p;  
 int * host_result;  
 int * dev_p;  
 
int main(void) {  
      // int ns = 4;
      // int ns = 1<<8;
        int ns = 1<<15;
      int data_size = ns * sizeof(int);
      
      /* Allocate host_p as pinned memory */
      checkCudaErrors( 
        cudaHostAlloc((void**)&host_p, data_size, 
        cudaHostAllocDefault) );  
      
      /* Allocate host_result as pinned memory */
      checkCudaErrors( 
        cudaHostAlloc((void**)&host_result, data_size, 
        cudaHostAllocDefault) );  
      /* Allocate dev_p on the device global memory */
      checkCudaErrors( 
        cudaMalloc((void**)&dev_p, data_size) );  
      
      /* Initialise host_p*/
      for (int i=0; i<ns; i++){  
           host_p[i] = i + 1;  
      }  
      
      /* Transfer data to the device host_p .. dev_p */
      checkCudaErrors( 
        cudaMemcpy(dev_p, host_p, data_size, cudaMemcpyHostToDevice) );
      
    /* Now launch the kernel... */
    dim3 block, grid;
    // set kernel launch configuration
    block = dim3(32);
    grid  = dim3((ns + block.x - 1) / block.x);
    kernel<<<grid, block>>>(dev_p);  
    checkCudaErrors(cudaGetLastError());
      
      /* Copy the result from the device back to the host */
      checkCudaErrors( 
        cudaMemcpy(host_result, dev_p, data_size, cudaMemcpyDeviceToHost) );
      
      printf("Check if no failures, then pass.\n");      
      /* and print the result */
      for (int i=0; i<ns; i++){  
            if (host_result[i] != i+11)
                printf("Failed result[%d] = %d\n", i, host_result[i]);  
      }
      /*
       * Now free the memory!
       */
      checkCudaErrors( cudaFree(dev_p) );  
      checkCudaErrors( cudaFreeHost(host_p) );  
      checkCudaErrors( cudaFreeHost(host_result) );  
      
      return 0;  
 } 