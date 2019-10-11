// nvcc dscal.c -lcudart -o dscal
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
	double* res; // n-vector on the host
	double al=2.0; // al=2
	x=(double *)malloc (n*sizeof(*x));// host memory alloc for x
	for(j=0;j<n;j++)
		x[j] = 0.1 * j; // x={0, 0.1, 0.2, 0.3, 0.4, 0.5}
	printf("x:\n");
	for(j=0;j<n;j++)
		printf("%g, ",x[j]); // print x
	printf("\n");
	res=(double *)malloc (n*sizeof(*res));
	for(j=0;j<n;j++)
		res[j] = al * x[j];
	// on the device
	double* d_x; // d_x - x on the device
	cudaStat=cudaMalloc((void**)&d_x,n*sizeof(*x)); //device
	// memory alloc for x
	stat = cublasCreate(&handle); // initialize CUBLAS context
	stat = cublasSetVector(n,sizeof(*x),x,1,d_x ,1);// cp x->d_x
	// scale the vector d_x by the scalar al: d_x = al*d_x
	stat=cublasDscal(handle,n,&al,d_x,1);
	stat=cublasGetVector(n,sizeof(double),d_x,1,x,1);//cp d_x->x
	printf("x after Dscal:\n"); // print x after Dscal:
	for(j=0;j<n;j++)
		printf("%g, ",x[j]); // x={0,2,4,6,8,10}
	printf("\n");
	for(j=0;j<n;j++) {
		if (res[j] != x[j]){
			printf("Result: %s, res[%d]=%g , x[%d]=%g\n", "FAILED", j, res[j], j, x[j]);
			goto FAILED;
		}
	}
	printf("Result: %s\n", "PASS");
FAILED:
	cudaFree(d_x); // free device memory
	cublasDestroy(handle); // destroy CUBLAS context
	free(x); // free host memory
	return EXIT_SUCCESS;
}
// x:
// 0, 1, 2, 3, 4, 5,
// x after Dscal:
// 0, 2, 4, 6, 8,10, // 2*{0,1,2,3,4,5}