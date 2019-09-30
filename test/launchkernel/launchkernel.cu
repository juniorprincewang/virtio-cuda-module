#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

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
	// printf("%f+g_data[%d]=%f\n", value, idx, g_data[idx]);
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

#define  TIMING

static double tvsub(struct timeval start, struct timeval end)
{
	return (double)(end.tv_usec - start.tv_usec)/1000000 +
                        (double)(end.tv_sec - start.tv_sec);
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
	int num = 1 << 24;
    int nbytes = num * sizeof(float);
    int value=41;
    struct timeval malloc_start, malloc_end;
    struct timeval meminit_start, meminit_end;
    struct timeval free_start, free_end;
    struct timeval d_malloc_start, d_malloc_end;
    struct timeval d_meminit_start, d_meminit_end;
    struct timeval d_free_start, d_free_end;
    struct timeval HtoD_start, HtoD_end;
    struct timeval DtoH_start, DtoH_end;
    struct timeval kernel_start, kernel_end;
    struct timeval total_start, total_end;
	//test();
	/* test case 2
	* add 	cudaGetDeviceCount
			cudaGetDevice
			cudaGetDeviceProperties
	*/
	devID = 0;
	CHECK(cudaSetDevice(devID));
	CHECK(cudaGetDeviceCount(&count));
	printf("cuda count=%d\n", count);
	// CHECK(cudaGetDevice(&devID));
	CHECK(cudaGetDeviceProperties(&props, devID));
	printf("Device %d: \"%s\" with Compute %d.%d capability\n",devID, props.name, props.major, props.minor);

	// printf("num 0x%x\n", num);
	#ifdef  TIMING
		gettimeofday(&total_start, NULL);
	#endif

    printf("sending 0x%x\n", nbytes);
	printf("allocating 0x%x\n", nbytes);
    #ifdef  TIMING
		gettimeofday(&malloc_start, NULL);
	#endif
	h_a=(float*)malloc(nbytes);
	#ifdef  TIMING
		gettimeofday(&malloc_end, NULL);
	#endif

	printf("initing mem\n");
	#ifdef  TIMING
		gettimeofday(&meminit_start, NULL);
	#endif
	memset(h_a, 0, nbytes);
	#ifdef  TIMING
		gettimeofday(&meminit_end, NULL);
	#endif
	printf("h_a=%p\n", h_a);
	// h_a[0] = 1;
	// start

	#ifdef  TIMING
		gettimeofday(&d_malloc_start, NULL);
	#endif
	CHECK(cudaMalloc((void**)&d_a, nbytes));
	#ifdef  TIMING
		gettimeofday(&d_malloc_end, NULL);
	#endif
	printf("d_a address = %p\n", d_a);
	#ifdef  TIMING
		gettimeofday(&d_meminit_start, NULL);
	#endif
	CHECK(cudaMemset(d_a, 0, nbytes));
	#ifdef  TIMING
		gettimeofday(&d_meminit_end, NULL);
	#endif
	
	// set kernel launch configuration
    block = dim3(4);
    grid  = dim3((num + block.x - 1) / block.x);

	#ifdef  TIMING
		gettimeofday(&HtoD_start, NULL);
	#endif
	CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
	#ifdef  TIMING
		gettimeofday(&HtoD_end, NULL);
	#endif
	#ifdef  TIMING
		gettimeofday(&kernel_start, NULL);
	#endif
	kernel<<<grid, block>>>(d_a, value);
	#ifdef  TIMING
		gettimeofday(&kernel_end, NULL);
	#endif

	#ifdef  TIMING
		gettimeofday(&DtoH_start, NULL);
	#endif
	CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
	#ifdef  TIMING
		gettimeofday(&DtoH_end, NULL);
	#endif

 	bool bFinalResults = (bool) checkResult(h_a, num, value);
	printf("result:%s\n", bFinalResults? "PASS" : "FAILED");
	// end
	#ifdef  TIMING
		gettimeofday(&d_free_start, NULL);
	#endif
	CHECK(cudaFree(d_a));
	#ifdef  TIMING
		gettimeofday(&d_free_end, NULL);
	#endif

	#ifdef  TIMING
		gettimeofday(&free_start, NULL);
	#endif
	free(h_a);
	#ifdef  TIMING
		gettimeofday(&free_end, NULL);
	#endif

	/* test case 3
	* add 	cudaMalloc
			cudaMemset
			cudaMemcpy
			cudaLaunch
			cudaFree
	*/
	#ifdef  TIMING
	gettimeofday(&total_end, NULL);
	double total_time 	= tvsub(total_start, total_end);
	double malloc_time 	= tvsub(malloc_start, malloc_end);
	double meminit_time 	= tvsub(meminit_start, meminit_end);
	double free_time 	= tvsub(free_start, free_end);
	double d_malloc_time 	= tvsub(d_malloc_start, d_malloc_end);
	double d_meminit_time 	= tvsub(d_meminit_start, d_meminit_end);
	double d_free_time 	= tvsub(d_free_start, d_free_end);
	double HtoD_time 	= tvsub(HtoD_start, HtoD_end);
	double DtoH_time 	= tvsub(DtoH_start, DtoH_end);
	double kernel_time 	= tvsub(kernel_start, kernel_end);

	printf("================\n");
	printf("total_time : \t\t%f\n", total_time);
	printf("host malloc: \t\t%f\n", malloc_time);
	printf("host mem init: \t\t%f\n", meminit_time);
	printf("device malloc: \t\t%f\n", d_malloc_time);
	printf("device mem init: \t%f\n", d_meminit_time);
	printf("HtoD: \t\t\t%f\n", HtoD_time);
	printf("Exec: \t\t\t%f\n", kernel_time);
	printf("DtoH: \t\t\t%f\n", DtoH_time);
	printf("device free: \t\t%f\n", d_free_time);
	printf("host free: \t\t%f\n", free_time);
	printf("================\n");
	#endif

	return EXIT_SUCCESS;
}
