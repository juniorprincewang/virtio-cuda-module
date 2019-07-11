#include <cuda.h>
#include <stdio.h>
__global__ void kernel(int *a, int *b, int c)
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
	struct cudaDeviceProp props;
	int *d_a, *d_b;
	int *a, b;
	int z=0;
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

	a=(int*)malloc(sizeof(int)*2);
	a[0]=1;
	a[1]=2;
	b = 16;
	// start
	cudaMalloc((void**)&d_a, sizeof(int)*2);
	cudaMalloc((void**)&d_b, sizeof(int));
	cudaMemcpy(d_a, a, sizeof(int)*2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);
	kernel<<<blocks, threads>>>(d_a, d_b, z);
	cudaMemcpy(a, d_a, sizeof(int)*2, cudaMemcpyDeviceToHost);
	// end
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);
	cudaFree(d_a);
	cudaFree(d_b);

	return 0;

	cudaMemcpy(d_a, a, sizeof(int)*2, cudaMemcpyHostToDevice);
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);
	kernel2<<<blocks, threads>>>(d_a, d_a, d_a, d_a);
	cudaMemcpy(a, d_a, sizeof(int)*2, cudaMemcpyDeviceToHost);
	// end
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);

	cudaMemcpy(d_a, a, sizeof(int)*2, cudaMemcpyHostToDevice);
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);
	kernel3<<<blocks, threads>>>(d_a);
	cudaMemcpy(a, d_a, sizeof(int)*2, cudaMemcpyDeviceToHost);
	// end
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);


	cudaFree(d_a);
	return 0;
}
