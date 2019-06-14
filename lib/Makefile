CUDAPATH=/usr/local/cuda
COMMONFLAGS=-Wall -fpic -shared -ldl
HOMES=/home/maxwell/virtio-cuda-module/

all:
	nvcc --shared --compiler-options "$(COMMONFLAGS)" libvcuda.c -I$(CUDAPATH)/include -o libvcuda.so

libcc:
	gcc -I$(CUDAPATH)/include $(COMMONFLAGS) libvcuda.c -o libvcuda.so

exp:
	export LD_LIBRARY_PATH="$(HOME)/lib/:$LD_LIBRARY_PATH"

clean:
	rm -f *.so
