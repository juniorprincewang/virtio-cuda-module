#include <cuda.h>
#include <stdio.h>

__global__ void kernel(int *a)
{
	int tx = threadIdx.x;
	a[tx] = a[tx] + 8;
}


int main()
{
	int *d_a;
	int *a;
	dim3 threads(10,1);
	dim3 blocks(1,1);
	int nstreams = 5;
	int i;
	int nreps = 10;
	// create CUDA event handles
    // use blocking sync
    cudaEvent_t start_event, stop_event;
	float elapsed_time;   // timing variables

    printf("\nStarting Test\n");

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));

    for (i = 0; i < nstreams; i++)
    {
        cudaStreamCreate(&(streams[i]));
    }


	a=(int*)malloc(sizeof(int)*2);
	a[0]=1;
	a[1]=2;
	cudaMalloc((void**)&d_a, sizeof(int)*2);
	cudaMemcpyAsync(d_a, a, sizeof(int)*2, cudaMemcpyHostToDevice, streams[0]);
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);

	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	// start
	cudaEventRecord(start_event, 0);
	for (i = 0; i < nreps; i++) {
		kernel<<<blocks, threads, 0, streams[0]>>>(d_a);
	}
	// end
	cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    printf("non-streamed:\t%.2f\n", elapsed_time / nreps);

	cudaMemcpy(a, d_a, sizeof(int)*2, cudaMemcpyDeviceToHost);
	printf("a[0] = %d, a[1] = %d\n", a[0], a[1]);

    // release resources
    for (i = 0; i < nstreams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	cudaFree(d_a);
	return 0;
}