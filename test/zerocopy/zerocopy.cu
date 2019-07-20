#include <cuda.h>
#include <stdio.h>
__global__ void kernel(int *a, int *b, int c)
{
	int tx = threadIdx.x;
	a[tx] = a[tx] + 2;
}

__global__ void kernel2(int *a, int *b, int *c, int *d)
{
	int tx = threadIdx.x;
	a[tx] = a[tx] + 10;
}

__global__ void kernel3(int *a)
{
	int tx = threadIdx.x;
	a[tx] = a[tx] + 20;
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
	//test();

	cudaGetDeviceCount(&count);
	printf("cuda count=%d\n", count);	
	// return 0;
	
	printf("[=] Before devID is %d\n",  devID);
	cudaGetDevice(&devID);
	printf("[=] After devID is %d\n",  devID);
	printf("prop=%lu\n", sizeof(struct cudaDeviceProp));	
	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n",devID, props.name, props.major, props.minor);
	// return 0;
	printf("allocate %d\n", big_data_size);
	a=(int*)malloc(sizeof(int)*big_data_size);
	for(int i=0; i<big_data_size; i++)
		a[i] = i;
	b = 16;
	// start
	cudaMalloc((void**)&d_a, big_data_size);
	cudaMalloc((void**)&d_b, sizeof(int));
	cudaMemcpy(d_a, a, big_data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);
	printf("a[%d] = %d\n", big_data_size-1, a[big_data_size-1]);
	kernel<<<blocks, threads>>>(d_a, d_b, big_data_size);
	cudaMemcpy(a, d_a, big_data_size, cudaMemcpyDeviceToHost);
	
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);
	printf("a[%d] = %d\n", big_data_size-1, a[big_data_size-1]);
	cudaFree(d_a);
	cudaFree(d_b);
	free(a);
	return 0;
}
