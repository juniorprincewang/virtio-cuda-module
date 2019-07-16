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
#include <sys/mman.h>
#include <malloc.h>


int fd=-1;
#define MODE O_RDWR
#define DEVICE_PATH "/dev/cudaport2p1"
#define error(fmt, arg...) printf("[ERROR]: %s->line : %d. "fmt, __FUNCTION__, __LINE__, ##arg)
#define debug(fmt, arg...) printf("[DEBUG]: "fmt, ##arg)
#define print(fmt, arg...) printf("[+]INFO: "fmt, ##arg)
#define func() printf("[FUNC] Now in %s\n", __FUNCTION__);


int minor=0;

// extern void *__libc_malloc(size_t);
// void *malloc(size_t size)
// {
// 	minor=1;
// 	return __libc_malloc(size);
// }

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

void mmap_test()
{
	int test_size=4096*1025;
	open_vdevice();
	char *p = (char *)mmap(0,
		test_size,
		PROT_READ | PROT_WRITE,
		MAP_SHARED,
		fd,
		0);
	if(p == MAP_FAILED) {
		error("mmap failed! reason is %s.\n", strerror(errno));
		return;
	}
	printf("%s\n", p);
	munmap(p, test_size);
	close_vdevice();
}

int main()
{
	
	unsigned int pagesize= getpagesize();
	int *p = (int *)malloc(4);
	char *s = "mmap";
	printf("minor=%d\n", minor);
	printf("page size = %u\n", pagesize);
	char *bp = (char *)malloc(pagesize*1025);
	memcpy(bp, s, 4);
	printf("%s\n", bp);
	free(bp);
	free(p);

	return 0;
}