CUDAPATH=/usr/local/cuda
COMMONFLAGS=-Wall -fpic -shared -ldl -Wno-multichar
HOMES=/home/maxwell/virtio-cuda-module/

all:
	nvcc --shared --compiler-options "$(COMMONFLAGS)" libcudart.c -I$(CUDAPATH)/include -o libcudart.so
	sudo -u root cp libcudart.so /usr/local/vcuda/lib64/

libcc:
	gcc -I$(CUDAPATH)/include $(COMMONFLAGS) libcudart.c -o libcudart.so

clean:
	rm -f *.so
