ROUTINES= /home/max/vgpu/vm
TARGET	= run
NVCC	= nvcc -arch sm_35 --cudart=shared
LIBS	= -lcudart
CFLAGS	= -L /usr/local/vcuda/lib64/

all:
#	nvcc launchkernel.cu --cudart=shared -o run
#	LD_PRELOAD=../../lib/libcudart.so ./run
	$(NVCC) $(CFLAGS) $(LIBS) -o $(TARGET) zerocopy.cu 

clean:
	rm -f ./run
	rm -f *.fatbin

fat:
	nvcc step2.cu --cudart=shared --fatbin -o run.fatbin
