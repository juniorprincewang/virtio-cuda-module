// nvcc pthread_cuda.cu -o pthread_cuda --cudart=shared -lpthread
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <unistd.h>
#include <sys/syscall.h>

#define ARR_SIZE    10
#define NUM_DEVICE  2

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_CHECK(call)                                              \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (cudaSuccess != err) {                                         \
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                     __FILE__, __LINE__, cudaGetErrorString(err) );       \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

typedef struct {
   int *arr;
   int *dev_arr;
   int *dev_result;
   int *result;
   int num;
} cuda_st;

__global__ void kernel_fc(int *dev_arr, int *dev_result)
{
    int idx = threadIdx.x;
    printf("dev_arr[%d] = %d\n", idx, dev_arr[idx]);
    atomicAdd(dev_result, dev_arr[idx]);
}

void *thread_func(void* struc)
{
    cudaEvent_t start, stop;
    cuda_st * data = (cuda_st*)struc;
    printf("thread %d func start\n", data->num);
    printf("arr %d = ", data->num);
    for(int i=0; i<10; i++) {
        printf("%d ", data->arr[i]);
    }
    printf("\n");
    for(int i=0; i<2; i++) {
        CUDA_CHECK(cudaSetDevice(data->num));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaMemcpy(data->dev_arr, data->arr,  sizeof(int)*ARR_SIZE, cudaMemcpyHostToDevice));
        kernel_fc<<<1,ARR_SIZE>>>(data->dev_arr, data->dev_result);
        CUDA_CHECK(cudaMemcpy(data->result, data->dev_result, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    printf("thread %ld func exit\n", syscall(SYS_gettid));
    return NULL;
}

int main(void)
{
    // Make object
    cuda_st cuda[NUM_DEVICE];

    // Make thread
    pthread_t pthread[NUM_DEVICE];

    // Host array memory allocation
    int *arr[NUM_DEVICE];
    for(int i=0; i<NUM_DEVICE; i++) {
        arr[i] = (int*)malloc(sizeof(int)*ARR_SIZE);
    }

    // Fill this host array up with specified data
    for(int i=0; i<NUM_DEVICE; i++) {
        for(int j=0; j<ARR_SIZE; j++) {
            arr[i][j] = i*ARR_SIZE+j;
        }
    }

    // To confirm host array data
    for(int i=0; i<NUM_DEVICE; i++) {
        printf("arr[%d] = ", i);
        for(int j=0; j<ARR_SIZE; j++) {
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }

    // Result memory allocation
    int *result[NUM_DEVICE];
    for(int i=0; i<NUM_DEVICE; i++) {
        result[i] = (int*)malloc(sizeof(int));
        memset(result[i], 0, sizeof(int));
    }
    // Device array memory allocation
    int *dev_arr[NUM_DEVICE];
    for(int i=0; i<NUM_DEVICE; i++) {
        CUDA_CHECK(cudaMalloc(&dev_arr[i], sizeof(int)*ARR_SIZE));
    }

    // Device result memory allocation
    int *dev_result[NUM_DEVICE];
    for(int i=0; i<NUM_DEVICE; i++) {
        CUDA_CHECK(cudaMalloc(&dev_result[i], sizeof(int)));
        CUDA_CHECK(cudaMemset(dev_result[i], 0, sizeof(int)));
    }

    // Connect these pointers with object
    for(int i=0; i<NUM_DEVICE; i++) {
        cuda[i].arr = arr[i];
        cuda[i].dev_arr = dev_arr[i];
        cuda[i].result = result[i];
        cuda[i].dev_result = dev_result[i];
        cuda[i].num = 0;
     }

    // Create and excute pthread
    for(int i=0; i<NUM_DEVICE; i++) {
        pthread_create(&pthread[i], NULL, thread_func, (void*)&cuda[i]);
    }

    // Join pthread
    for(int i=0; i<NUM_DEVICE; i++) {
        pthread_join(pthread[i], NULL);
    }

    for(int i=0; i<NUM_DEVICE; i++) {
        printf("result[%d] = %d\n", i, (*cuda[i].result));
    }

    return 0;
}