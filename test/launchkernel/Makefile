
HOST=10.10.81.192
USER=max
ROUTINES=/home/max/vgpu/vm
HOMES=/home/maxwell/virtio-cuda-module/


scp: fat
	scp ./run.fatbin $(USER)@$(HOST):$(ROUTINES)

fat:
	nvcc step2.cu --cudart=shared --fatbin -o run.fatbin

all:
	nvcc step2.cu --cudart=shared -o run
	LD_PRELOAD=../../lib/libvcuda.so ./run

clean:
	rm -f ./run
	rm -f *.fatbin
