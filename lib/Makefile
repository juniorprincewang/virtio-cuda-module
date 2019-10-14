TARGET 	= libcudart
CUDAPATH=/usr/local/cuda
COMMONFLAGS=-Wall -fpic -shared -ldl -Wno-multichar -Wl,-soname,$(TARGET).so.9.1 -nostdlib
HOMES=/home/maxwell/virtio-cuda-module/

all:
	# nvcc --shared --compiler-options "$(COMMONFLAGS)" libcudart.c -I$(CUDAPATH)/include -o $(TARGET).so.9.1.0
	gcc -I$(CUDAPATH)/include $(COMMONFLAGS) libcudart.c -o $(TARGET).so.9.1.0
	sudo -u root cp $(TARGET).so.9.1.0 /usr/local/vcuda/lib64/

libcc:
	gcc -I$(CUDAPATH)/include $(COMMONFLAGS) libcudart.c -o libcudart.so

clean:
	rm -f *.so
