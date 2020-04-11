TARGET	= run
NVCC	= nvcc -arch sm_35 --cudart=shared
LIBS	= -lcudart
CFLAGS	= 

all:
	$(NVCC) $(CFLAGS) $(LIBS) -o $(TARGET) stream.cu 

clean:
	rm -f ./run *.fatbin *.cubin

fat:
	$(NVCC) $(CFLAGS) stream.cu --fatbin -o run.fatbin

cubin:
	$(NVCC) $(CFLAGS) stream.cu --cubin -o run.cubin
