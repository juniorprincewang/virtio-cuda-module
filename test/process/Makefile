TARGET	= delay
NVCC	= nvcc -arch sm_35 
LIBS	= -lcudart
CFLAGS	= -L /usr/local/vcuda/lib64/

all:
#	nvcc launchkernel.cu --cudart=shared -o run
#	LD_PRELOAD=../../lib/libcudart.so ./run
	$(NVCC) $(CFLAGS) $(LIBS) -o $(TARGET) delay.cu

clean:
	rm -f ./run
	rm -f *.fatbin