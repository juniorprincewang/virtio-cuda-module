#include <cuda.h>
#include <stdio.h>
__global__ void kernel(int *a, int *b, int *c)
{
	int tx = threadIdx.x;
	switch (tx)
	{
		case 0:
			a[tx] = a[tx] + 2;
			break;
		case 1:
			a[tx] = a[tx] + 3;
			break;
		default:
			break;
	}
}

void test()
{
	printf("sizeof(cudaEvent_t)=%lu\n", sizeof(cudaEvent_t));
	printf("sizeof(cudaStream_t)=%lu\n", sizeof(cudaStream_t));
	printf("sizeof(cudaError_t)=%lu\n", sizeof(cudaError_t));
	printf("sizeof(uint64_t)=%lu\n", sizeof(uint64_t));
	printf("sizeof(uint32_t)=%lu\n", sizeof(uint32_t));
}

int main()
{
	int devID=1;
	int count = 0;
	cudaDeviceProp props;
	int *d_a;
	int *a;
	dim3 threads(2,1);
	dim3 blocks(1,1);
	//test();

	//cudaGetDeviceCount(&count);
	//printf("cuda count=%d\n", count);	
	/*
	printf("[=] Before devID is %d\n",  devID);
	cudaGetDevice(&devID);
	printf("[=] After devID is %d\n",  devID);
	
	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n",devID, props.name, props.major, props.minor);
	return 0;
*/
	a=(int*)malloc(sizeof(int)*2);
	a[0]=1;
	a[1]=2;
	// start
	cudaMalloc((void**)&d_a, sizeof(int)*2);
	cudaMemcpy(d_a, a, sizeof(int)*2, cudaMemcpyHostToDevice);
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);

	// kernel<<<blocks, threads>>>(d_a, d_a, d_a);
	cudaMemcpy(a, d_a, sizeof(int)*2, cudaMemcpyDeviceToHost);
	// end
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);
	cudaFree(d_a);
	return 0;
}
