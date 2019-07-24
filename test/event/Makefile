TARGET	= run
NVCC	= nvcc -arch sm_35 
LIBS	= -lcudart
CFLAGS	= -L /usr/local/vcuda/lib64/

all:
#	nvcc step2.cu --cudart=shared -o run
#	LD_PRELOAD=../../lib/libcudart.so ./run
	$(NVCC) $(CFLAGS) $(LIBS) -o $(TARGET) event.cu 

shared:
	$(NVCC) --cudart=shared -o $(TARGET) event.cu 
	LD_PRELOAD=/usr/local/vcuda/lib64/libcudart.so ./run

clean:
	rm -f ./run

