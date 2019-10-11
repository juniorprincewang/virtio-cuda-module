//  nvcc saxpy.c -lcudart -o saxpy
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
	float* res; // n-vector on the host
	float al=2.0; // al=2
	x=(float *)malloc (n*sizeof(*x));// host memory alloc for x
	for(j=0;j<n;j++)
		x[j] = 0.1 * j; // x={0, 0.1, 0.2, 0.3, 0.4, 0.5}
	y=(float *)malloc (n*sizeof(*y));// host memory alloc for y
	for(j=0;j<n;j++)
		y[j] = 0.1 * j; // x={0, 0.1, 0.2, 0.3, 0.4, 0.5}
	res=(float *)malloc (n*sizeof(*res));// host memory alloc for y
	memset(res, 0, sizeof(res));
	printf("x,y:\n");
	for(j=0;j<n;j++) {
		printf("%g, ",x[j]); // print x,y
		res[j] = al * x[j] + y[j];
	}
	printf("\n");
	// on the device
	float* d_x; // d_x - x on the device
	float* d_y; // d_y - y on the device
	cudaStat=cudaMalloc((void**)&d_x,n*sizeof(*x)); //device
	// memory alloc for x
	cudaStat=cudaMalloc((void**)&d_y,n*sizeof(*y)); //device
	// memory alloc for y
	stat = cublasCreate(&handle); // initialize CUBLAS context
	stat = cublasSetVector(n,sizeof(*x),x,1,d_x ,1); //cp x->d_x
	stat = cublasSetVector(n,sizeof(*y),y,1,d_y ,1); //cp y->d_y
	// multiply the vector d_x by the scalar al and add to d_y
	// d_y = al*d_x + d_y, d_x,d_y - n-vectors; al - scalar
	stat=cublasSaxpy(handle,n,&al,d_x,1,d_y,1);
	stat=cublasGetVector(n,sizeof(float),d_y,1,y,1);//cp d_y->y
	printf("y after Saxpy:\n"); // print y after Saxpy
	for(j=0;j<n;j++)
		printf("%g, ",y[j]);
	printf("\n");
	for(j=0;j<n;j++) {
		if (res[j] != y[j]){
			printf("Result: %s, res[%d]=%g , y[%d]=%g\n", "FAILED", j, res[j], j, y[j]);
			goto FAILED;
		}
	}
	printf("Result: %s\n", "PASS");
FAILED:
	cudaFree(d_x); // free device memory
	cudaFree(d_y); // free device memory
	cublasDestroy(handle); // destroy CUBLAS context
	free(x); // free host memory
	free(y); // free host memory
	return EXIT_SUCCESS;
}
// x,y:
// 0, 1, 2, 3, 4, 5,
// y after Saxpy:
// 0, 3, 6, 9,12,15,// 2*x+y = 2*{0,1,2,3,4,5} + {0,1,2,3,4,5}