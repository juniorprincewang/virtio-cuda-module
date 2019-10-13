#include <cuda.h>
#include <stdio.h>


__global__ void kernel(int *a, int *b, int c)
{
	int tx = threadIdx.x;
	a[tx] = a[tx] + 2;
}

int main()
{
	int devID=1;
	int count = 0;
	struct cudaDeviceProp props;
	int *d_a, *d_b;
	int *a, b;
	int z=16;
	int big_data_size=(1<<22)+(1<<12);
	dim3 threads(10,1);
	dim3 blocks(1,1);
	cudaMallocHost(4);
	
	return 0;
}
