//nvcc scopy.c -lcudart -o scopy
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6 // length of x,y

int main(void){
	cudaError_t cudaStat; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	cublasHandle_t handle; // CUBLAS context
	int j; // index of elements
	float* x; // n-vector on the host
	float* y; // n-vector on the host
	x=(float *)malloc (n*sizeof(*x));// host memory alloc for x
	for(j=0;j<n;j++)
		x[j]=(float)j; // x={0,1,2,3,4,5}
	printf("x: ");
	for(j=0;j<n;j++)
		printf("%2.0f,",x[j]); // print x
	printf("\n");
	y=(float *)malloc (n*sizeof(*y));// host memory alloc for y
	// on the device
	float* d_x; // d_x - x on the device
	float* d_y; // d_y - y on the device
	cudaStat=cudaMalloc((void**)&d_x,n*sizeof(*x)); // device
	// memory alloc for x
	cudaStat=cudaMalloc((void**)&d_y,n*sizeof(*y)); // device
	// memory alloc for y
	stat = cublasCreate(&handle); // initialize CUBLAS context
	stat = cublasSetVector(n,sizeof(*x),x,1,d_x ,1); //cp x->d_x
	// copy the vector d_x into d_y: d_x -> d_y
	stat=cublasScopy(handle,n,d_x,1,d_y,1);
	stat=cublasGetVector(n,sizeof(float),d_y,1,y,1);//cp d_y->y
	printf("y after copy:\n");
	for(j=0;j<n;j++)
		printf("%2.0f,",y[j]); // print y
	printf("\n");
	cudaFree(d_x); // free device memory
	cudaFree(d_y); // free device memory
	cublasDestroy(handle); // destroy CUBLAS context
	free(x); // free host memory
	free(y); // free host memory
	return EXIT_SUCCESS;
}
// x: 0, 1, 2, 3, 4, 5,
// y after Scopy: // {0,1,2,3,4,5} -> {0,1,2,3,4,5}
// 0, 1, 2, 3, 4, 5,