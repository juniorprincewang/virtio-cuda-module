#include <cuda.h>
#include <stdio.h>

__global__ void kernel(float *g_data, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
    printf("%f+g_data[%d]=%f\n", value, idx, g_data[idx]);
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
    int devID=1;
    int count = 0;
    struct cudaDeviceProp props;
    float *d_a=0;
    float *h_a=0;
    dim3 block, grid;
    int num = 1 << 4;
    int nbytes = num * sizeof(float);
    int value=16;
    int nStreams = 4;
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

    h_a=(float*)malloc(nbytes);
    memset(h_a, 0, nbytes);
    // start
    cudaMalloc((void**)&d_a, nbytes);
    cudaMemset(d_a, 0, nbytes);
    // set kernel launch configuration
    block = dim3(4);
    grid  = dim3((num + block.x - 1) / block.x);

    float ms; // elapsed time in milliseconds
    // create events and streams
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaStream_t stream[nStreams];
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&dummyEvent);
    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);

    cudaEventRecord(startEvent,0);
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
    kernel<<<grid, block>>>(d_a, value);
    cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Time for sequential transfer and execute (ms): %f\n", ms);

    bool bFinalResults = (bool) checkResult(h_a, num, value);
    printf("result:%d\n", bFinalResults);
    // end
    free(h_a);
    cudaFree(d_a);
     // cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(dummyEvent);
    for (int i = 0; i < nStreams; ++i)
        cudaStreamDestroy(stream[i]);
    return 0;
}
