TARGET	= run
NVCC	= nvcc -arch sm_35 --cudart=shared
LIBS	= -lcudart
CFLAGS	= 

all:
	$(NVCC) $(CFLAGS) $(LIBS) -o $(TARGET) concurrent.cu 

clean:
	rm -f ./run *.fatbin *.cubin

fat:
	$(NVCC) concurrent.cu --fatbin -o run.fatbin

cubin:
	$(NVCC) concurrent.cu --cubin -o run.cubin
