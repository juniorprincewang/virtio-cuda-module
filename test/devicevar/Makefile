TARGET	= run
NVCC	= nvcc -arch sm_35 --cudart=shared
LIBS	= -lcudart
CFLAGS	= 

all:
#	nvcc launchkernel.cu --cudart=shared -o run
#	LD_PRELOAD=../../lib/libcudart.so ./run
	$(NVCC) $(CFLAGS) $(LIBS) -o $(TARGET) cudaMemcpyToSymbol.cu 

clean:
	rm -f ./run
	rm -f *.fatbin


fat:
	nvcc cudaMemcpyToSymbol.cu --cudart=shared --fatbin -o run.fatbin

cubin:
	$(NVCC) cudaMemcpyToSymbol.cu --cubin -o run.cubin
