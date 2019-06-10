#include <cuda.h>
#include <stdio.h>
#include "mycuda.h"

__global__ void kernel(int *a)
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

int main()
{
	int devID=1;
	cudaDeviceProp props;
	int *d_a;
	int *a;
	dim3 threads(2,1);
	dim3 blocks(1,1);
//	printf("before devID is %d\n",  *(int*)(&devID));
//	myGetDevice(&devID);
//	printf("after devID is %d\n",  devID);
//	myGetDeviceProperties(&props, devID);
//	printf("Device %d: \"%s\" with Compute %d.%d capability\n",devID, props.name, props.major, props.minor);
	a=(int*)malloc(sizeof(int)*2);
	a[0]=1;
	a[1]=2;
	myMalloc((void**)&d_a, sizeof(int)*2);
	myMemcpy(d_a, a, sizeof(int)*2, cudaMemcpyHostToDevice);
	kernel<<<blocks, threads>>>(d_a);
	myMemcpy(a, d_a, sizeof(int)*2, cudaMemcpyDeviceToHost);
	myFree(d_a);
	return 0;
}
