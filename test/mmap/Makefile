all:
	nvcc mmap.cu --cudart=shared -o run
	LD_PRELOAD=../../lib/libcudart.so ./run

clean:
	rm -f ./run
