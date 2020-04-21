TARGET	= run
NVCC	= nvcc -arch sm_70 --cudart=shared
LIBS	= -lcudart
CFLAGS	= 

all:
	$(NVCC) $(CFLAGS) $(LIBS) -o $(TARGET) launchkernel.cu 

clean:
	rm -f ./run *.fatbin *.cubin

fat:
	$(NVCC) $(CFLAGS) $(LIBS)  launchkernel.cu --fatbin -o run.fatbin

cubin:
	$(NVCC) $(CFLAGS) $(LIBS)  launchkernel.cu --cubin -o run.cubin
