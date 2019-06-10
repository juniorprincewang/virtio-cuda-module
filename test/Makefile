CUDAPATH=/usr/local/cuda
COMMONFLAGS=-Wall -fpic -shared -ldl
HOST=10.10.81.192
LOCAL=192.168.151.171
USER=max
ROUTINES=/home/max/vgpu/vm
HOMES=/home/maxwell/virtio-cuda-module/

lib:
	nvcc --shared --compiler-options "$(COMMONFLAGS)" libvcuda.c -I$(CUDAPATH)/include -o libvcuda.so

libc:
	gcc -I$(CUDAPATH)/include $(COMMONFLAGS) libvcuda.c -o libvcuda.so

export:
	export LD_LIBRARY_PATH=$(HOME)/test

scp: fat
	scp ./run.fatbin $(USER)@$(HOST):$(ROUTINES)

fat:
	nvcc step2.cu --cudart=shared --fatbin -o run.fatbin
#	scp ./run.fatbin max@192.168.151.169:~/GPU/samples/hmacfatbin/

run: clean
#	nvcc step1.cu -L. -lcudatest -cudart=shared -o run
	nvcc step2.cu --cudart=shared -o run
	LD_PRELOAD=./libvcuda.so ./run

clean:
	rm -f ./run
	rm -f *.so
	rm -f *.fatbin
