HOST 	= 10.10.81.192
USER	= max
ROUTINES= /home/max/vgpu/vm
TARGET	= run
NVCC	= nvcc -arch sm_35 --cudart=shared
LIBS	= -lcudart
CFLAGS	= 

all:
#	nvcc launchkernel.cu --cudart=shared -o run
#	LD_PRELOAD=../../lib/libcudart.so ./run
	$(NVCC) $(CFLAGS) $(LIBS) -o $(TARGET) launchkernel.cu 

clean:
	rm -f ./run
	rm -f *.fatbin

scp: fat
	scp ./run.fatbin $(USER)@$(HOST):$(ROUTINES)

fat:
	nvcc launchkernel.cu --cudart=shared --fatbin -o run.fatbin

cubin:
	nvcc launchkernel.cu --cudart=shared --cubin -o run.cubin
