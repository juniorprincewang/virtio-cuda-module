// nvcc ddot.c -lcudart -o ddot
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6 // length of x,y


int main ( void ){
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ;
	int j; // index of elements
	double * x; // n- vector on the host
	double * y; // n- vector on the host
	x=( double *) malloc (n* sizeof (*x)); // host memory alloc for x
	for(j=0;j<n;j++)
		x[j]= j*0.1; // x={0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5}
	y=( double *) malloc (n* sizeof (*y)); // host memory alloc for y
	for(j=0;j<n;j++)
		y[j]= 0.1 * j; // y={0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5}
	printf ("x,y:\n");
	double res = 0.0;
	for(j=0;j<n;j++){
		printf (" %f,",x[j]); // print x,y
		res += x[j] * y[j];
	}
	printf ("\nresult ought to be %g\n", res);

	// on the device
	double * d_x; // d_x - x on the device
	double * d_y; // d_y - y on the device
	cudaStat = cudaMalloc (( void **)& d_x ,n* sizeof (*x)); // device
	// memory alloc for x
	cudaStat = cudaMalloc (( void **)& d_y ,n* sizeof (*y)); // device
	// memory alloc for y
	stat = cublasCreate (&handle ); // initialize CUBLAS context
	stat = cublasSetVector (n, sizeof (*x) ,x ,1 ,d_x ,1); // cp x- >d_x
	stat = cublasSetVector (n, sizeof (*y) ,y ,1 ,d_y ,1); // cp y- >d_y
	double result ;
	// dot product of two vectors d_x ,d_y :
	// d_x [0]* d_y [0]+...+ d_x [n -1]* d_y [n -1]
	stat=cublasDdot(handle,n,d_x,1,d_y,1,&result);
	printf ("dot product x.y: %g \n", result ); // print the result
	cudaFree (d_x ); // free device memory
	cudaFree (d_y ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	free (x); // free host memory
	free (y); // free host memory
	printf("Result: %s\n", res!=result ? "FAILED" : "PASS");
	return EXIT_SUCCESS ;
}
// x,y:
// 0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 ,
// dot product x.y: // x.y=
// 55 // 1*1+2*2+3*3+4*4+5*5