#include <stdio.h>
#include "../../virtio-ioc.h"
#include <string.h> // memset
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>	//open
#include <unistd.h>	// close
#include <sys/syscall.h> // SYS_gettid
#include <errno.h>

int fd=-1;
#define MODE O_RDWR
#define DEVICE_PATH "/dev/cudaport2p1"
#define error(fmt, arg...) printf("[ERROR]: %s->line : %d. "fmt, __FUNCTION__, __LINE__, ##arg)
#define debug(fmt, arg...) printf("[DEBUG]: "fmt, ##arg)
#define print(fmt, arg...) printf("[+]INFO: "fmt, ##arg)
#define func() printf("[FUNC] Now in %s\n", __FUNCTION__);
/*
 * open token
*/
void open_vdevice()
{
	func();
	if (fd == -1)
	{
		fd = open(DEVICE_PATH, MODE);
		if(fd == -1)
		{
			error("open device %s failed, %s (%d)\n", DEVICE_PATH, (char*)strerror(errno), errno);
			exit(EXIT_FAILURE);
		}
		debug("fd is %d\n", fd);
	}
}

void close_vdevice()
{
	func();
	close(fd);
	debug("closing fd\n");
}

void hello_write(int *var)
{
	VirtIOArg arg;
	open_vdevice();
	printf("_IOC_NR=%lu, _IOC_TYPE=%lu, _IOC_SIZE=%lu\n",\
		_IOC_NR(VIRTIO_IOC_HELLO), \
		_IOC_TYPE(VIRTIO_IOC_HELLO), \
		_IOC_SIZE(VIRTIO_IOC_HELLO) );
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_HELLO;
	arg.src = var;
	arg.srcSize = sizeof(int);
	arg.tid = syscall(SYS_gettid);

	if(ioctl(fd, VIRTIO_IOC_HELLO, &arg) == -1){
		error("ioctl when cmd is %d\n", VIRTIO_CUDA_HELLO);
	}
	printf("cmd = %d\n", arg.cmd);
	close_vdevice();
}

int main()
{
	int a=1;
	hello_write(&a);
	printf("a++=%d\n", a);
	return 0;
}