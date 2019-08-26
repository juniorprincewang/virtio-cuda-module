#include <cuda.h>
#include <stdio.h>

#define CHECK(call) { \
	cudaError_t err; \
	if ( (err = (call)) != cudaSuccess) { \
		fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
				__FILE__, __LINE__); \
		exit(1); \
	} \
}

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
	/* test case 1
	* for 	__cudaRegisterFatBinary
			__cudaUnregisterFatBinary
			__cudaRegisterFunction
	*/
	// return 0;
	int devID=1;
	int count = 0;
	struct cudaDeviceProp props;
	float *d_a=0;
	float *h_a=0;
	dim3 block, grid;
	int num = 1 << 22;
    int nbytes = num * sizeof(float);
    int value=16;
	//test();
	/* test case 2
	* add 	cudaGetDeviceCount
			cudaGetDevice
			cudaGetDeviceProperties
	*/
/*
	cudaGetDeviceCount(&count);
	printf("cuda count=%d\n", count);
	
	printf("[=] Before devID is %d\n",  devID);
	cudaGetDevice(&devID);
	printf("[=] After devID is %d\n",  devID);
	printf("prop=%lu\n", sizeof(struct cudaDeviceProp));	
	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n",devID, props.name, props.major, props.minor);
	// return 0;
*/

    printf("sending 0x%x\n", nbytes);

	h_a=(float*)malloc(nbytes);
	memset(h_a, 0, nbytes);
	// h_a[0] = 1;
	// start
	CHECK(cudaMalloc((void**)&d_a, nbytes));
	CHECK(cudaMemset(d_a, 0, nbytes));
	
	// set kernel launch configuration
    block = dim3(4);
    grid  = dim3((num + block.x - 1) / block.x);

	
	CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
	kernel<<<grid, block>>>(d_a, value);
	CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
 	bool bFinalResults = (bool) checkResult(h_a, num, value);
	printf("result:%d\n", bFinalResults);
	// end
	free(h_a);
	CHECK(cudaFree(d_a));

	/* test case 3
	* add 	cudaMalloc
			cudaMemset
			cudaMemcpy
			cudaLaunch
			cudaFree
	*/

	return EXIT_SUCCESS;
}
