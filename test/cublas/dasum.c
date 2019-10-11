// nvcc dasum.c -lcudart -o dasum
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6 // length of x

int main(void){
	cudaError_t cudaStat; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	cublasHandle_t handle; // CUBLAS context
	int j; // index of elements
	double* x; // n-vector on the host
	x=(double *)malloc (n*sizeof(*x)); // host memory alloc
	for(j=0;j<n;j++)
		x[j]=(double)(-j * 0.1); // x={0 , -0.1 , -0.2 , -0.3 , -0.4 , -0.5}
	printf("x: ");
	double res = 0;
	for(j=0;j<n;j++) {
		printf("%g,",x[j]); // print x
		res += x[j] < 0? -x[j] : x[j];
	}
	printf("\n actual asum = %g\n", res);
	// on the device
	double* d_x; // d_x - x on the device
	cudaStat=cudaMalloc((void**)&d_x,n*sizeof(*x)); //device
	// memory alloc
	stat = cublasCreate(&handle); // initialize CUBLAS context
	stat = cublasSetVector(n,sizeof(*x),x,1,d_x ,1);// cp x->d_x
	double result;
	// add absolute values of elements of the array d_x:
	// |d_x[0]|+...+|d_x[n-1]|
	stat=cublasDasum(handle,n,d_x,1,&result);
	//print the result
	printf("sum of the absolute values of elements of x:%g\n", result);
	cudaFree(d_x); // free device memory
	cublasDestroy(handle); // destroy CUBLAS context
	free(x); // free host memory
	printf("Result: %s\n", res!=result ? "FAILED" : "PASS");
	return EXIT_SUCCESS;
}
// x: 0, 1, 2, 3, 4, 5,
// sum of the absolute values of elements of x: 15
//|0|+|1|+|2|+|3|+|4|+|5|=15