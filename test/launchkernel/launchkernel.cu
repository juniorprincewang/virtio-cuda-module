#include <cuda.h>
#include <stdio.h>
__global__ void kernel2(int *a, int *b, int c)
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
	float *d_a=0;
	float *h_a=0;
	dim3 block, grid;
	int num = 1 << 4;
    int nbytes = num * sizeof(int);
    int value=16;
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
	cudaMemset(d_a, 255, nbytes);
	// set kernel launch configuration
    block = dim3(4);
    grid  = dim3((num + block.x - 1) / block.x);

	// cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
	// kernel<<<grid, block>>>(d_a, value);
	cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);
	// end
	free(h_a);
	cudaFree(d_a);
 	bool bFinalResults = (bool) checkResult(h_a, num, value);
	printf("result:%d\n", bFinalResults);
	return 0;
}
