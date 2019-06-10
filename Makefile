obj-m :=virtio_cuda.o

all:
	make -C /lib/modules/`uname -r`/build M=$(PWD) modules

install:
	cp -f ./virtio_cuda.ko /lib/modules/`uname -r`/kernel/drivers/char/virtio_cuda.ko
	depmod -a
	modprobe virtio_cuda

uninstall:
	# when next cmd runs into error, then run `depmod -a`.
	modprobe -r virtio_cuda
	rm -f /lib/modules/`uname -r`/kernel/drivers/char/virtio_cuda.ko

clean:	
	make -C /lib/modules/`uname -r`/build M=$(PWD) clean 

pre:
	modprobe -r virtio_pci
	modprobe -r virtio_cuda
	modprobe virtio_pci
	modprobe virtio_cuda

