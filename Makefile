obj-m :=virtio_cuda.o

all:
	make -C /lib/modules/`uname -r`/build M=$(PWD) modules

clean:	
	make -C /lib/modules/`uname -r`/build M=$(PWD) clean 

install: clean all
	sudo insmod	virtio_cuda.ko

uninstall:
	# when next cmd runs into error, then run `depmod -a`.
	sudo rmmod virtio_cuda
