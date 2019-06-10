CUDAPATH=/usr/local/cuda
COMMONFLAGS=-Wall -fpic -shared -ldl
HOST=192.168.151.169
LOCAL=192.168.151.171
ROUTINES=/home/max/GPU/samples/hmacfatbin/
USER=max
HOME=/root/virtiocuda

lib:
	nvcc --shared --compiler-options "$(COMMONFLAGS)" mylib.c -I$(CUDAPATH)/include -o libcudatest.so

libc:
	gcc -I$(CUDAPATH)/include $(COMMONFLAGS) mylib.c -o libcudatest.so

export:
	export LD_LIBRARY_PATH=$(HOME)/testcase

scp: fat
	scp ./run.fatbin $(USER)@$(HOST):$(ROUTINES)

fat:
	nvcc step2.cu --cudart=shared --fatbin -o run.fatbin
#	scp ./run.fatbin max@192.168.151.169:~/GPU/samples/hmacfatbin/

run: clean
#	nvcc step1.cu -L. -lcudatest -cudart=shared -o run
	nvcc step2.cu --cudart=shared -o run
	LD_PRELOAD=./libcudatest.so ./run

clean:
	rm -f ./run
