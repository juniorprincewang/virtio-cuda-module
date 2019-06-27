/*
nvcc -arch=sm_35 -o delay delay.cu
*/
#include <stdio.h>
#include <stdlib.h>

#define MAX_DELAY 30

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)


#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

#define APPRX_CLKS_PER_SEC 1000000000ULL
__global__ void delay_kernel(unsigned seconds){

  unsigned long long dt = clock64();
  while (clock64() < (dt + (seconds*APPRX_CLKS_PER_SEC)));
}

int main(int argc, char *argv[]){

  unsigned delay_t = 2; // seconds, approximately
  unsigned delay_t_r;
  if (argc > 1) delay_t_r = atoi(argv[1]);
  if ((delay_t_r > 0) && (delay_t_r < MAX_DELAY)) delay_t = delay_t_r;
  unsigned long long difft = dtime_usec(0);
  delay_kernel<<<1,1>>>(delay_t);
  cudaDeviceSynchronize();
  //cudaCheckErrors("kernel fail");
  difft = dtime_usec(difft);
  printf("kernel duration: %fs\n", difft/(float)USECPSEC);
  return 0;
}
