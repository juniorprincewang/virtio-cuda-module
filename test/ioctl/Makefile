all:
	nvcc ioctl.c --cudart=shared -o run
	LD_PRELOAD=../../lib/libvcuda.so ./run

clean:
	rm -f ./run
