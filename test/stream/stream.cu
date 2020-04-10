#include <cuda.h>
#include <stdio.h>

__global__ void kernel(float *g_data, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
}

int checkResult(float *data, const int n, const float x)
{
    for (int i = 0; i < n; i++)
    {
        if (data[i] != x)
        {
            printf("Error! data[%d] = %f, ref = %f\n", i, data[i], x);
            return 0;
        }
    }

    return 1;
}


int main()
{
	float *d_a;
	float *h_a;
	int nstreams = 2;
	int i,j;
	int nreps = 10;
    int num = 1 << 10;
    int nbytes = num * sizeof(float);
    float value = 16;
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

    h_a=(float*)malloc(nbytes);
    memset(h_a, 0, nbytes);

	dim3 block = dim3(32,1,1);
    dim3 grid  = dim3((num + block.x - 1) / block.x);

	cudaMalloc((void**)&d_a, nbytes);

	// cudaMemcpyAsync(d_a, a, sizeof(float)*2, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	// start
	cudaEventRecord(start_event, 0);
	for (i = 0; i < nreps; i++) {
        for (j=0; j<nstreams; j++) {
		  kernel<<<grid, block, 0, streams[j]>>>(d_a, value);
        }
	}
	// end
	cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    printf("non-streamed:\t%.2f\n", elapsed_time / nreps);

	cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);

    int bFinalResults = checkResult(h_a, num, nstreams * nreps * value);
    printf("result:%s\n", bFinalResults? "PASS" : "FAILED");
    // release resources
    for (i = 0; i < nstreams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	cudaFree(d_a);
    free(h_a);
	return 0;
}
