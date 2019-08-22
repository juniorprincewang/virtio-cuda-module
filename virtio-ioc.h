#ifndef VIRTCR_IOC_H
#define VIRTCR_IOC_H

#define VIRTIO_CUDA_DEBUG
#define KMALLOC_SHIFT 22 // 4MB
#define KMALLOC_SIZE (1UL<<KMALLOC_SHIFT)



#ifndef __KERNEL__
#define __user

#include <stdint.h>
#include <sys/ioctl.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define VIRTIO_CUDA_HELLO 0
/** module control	**/
#define VIRTIO_CUDA_REGISTERFATBINARY 1
#define VIRTIO_CUDA_UNREGISTERFATBINARY	2
#define VIRTIO_CUDA_REGISTERFUNCTION 3
#define VIRTIO_CUDA_LAUNCH 4
/* memory management */
#define VIRTIO_CUDA_MALLOC 5
#define VIRTIO_CUDA_MEMCPY 6
#define VIRTIO_CUDA_FREE 7
/**	device management	**/
#define VIRTIO_CUDA_GETDEVICE 8
#define VIRTIO_CUDA_GETDEVICEPROPERTIES 9
#define VIRTIO_CUDA_CONFIGURECALL 10

#define VIRTIO_CUDA_SETUPARGUMENT 11
#define VIRTIO_CUDA_GETDEVICECOUNT 12
#define VIRTIO_CUDA_SETDEVICE 13
#define VIRTIO_CUDA_DEVICERESET 14
#define VIRTIO_CUDA_STREAMCREATE 15

#define VIRTIO_CUDA_STREAMDESTROY 16
#define VIRTIO_CUDA_EVENTCREATE 17
#define VIRTIO_CUDA_EVENTDESTROY 18
#define VIRTIO_CUDA_EVENTRECORD 19
#define VIRTIO_CUDA_EVENTSYNCHRONIZE 20

#define VIRTIO_CUDA_EVENTELAPSEDTIME 21
#define VIRTIO_CUDA_THREADSYNCHRONIZE 22
#define VIRTIO_CUDA_GETLASTERROR 23

#define VIRTIO_CUDA_MEMCPY_ASYNC 24
#define VIRTIO_CUDA_MEMSET 25
#define VIRTIO_CUDA_DEVICESYNCHRONIZE 26

#define VIRTIO_CUDA_EVENTCREATEWITHFLAGS 27
#define VIRTIO_CUDA_HOSTREGISTER 28
#define VIRTIO_CUDA_HOSTGETDEVICEPOINTER 29
#define VIRTIO_CUDA_HOSTUNREGISTER 30
#define VIRTIO_CUDA_SETDEVICEFLAGS 31
#define VIRTIO_CUDA_MEMGETINFO 32
#define VIRTIO_CUDA_MALLOCHOST 33


struct GPUDevice {
	uint32_t device_id;
	struct cudaDeviceProp prop;
};


#else

#include <linux/ioctl.h>


#endif //KERNEL



//for crypto_data_header, if these is no openssl header
#ifndef RSA_PKCS1_PADDING

#define RSA_PKCS1_PADDING	1
#define RSA_SSLV23_PADDING	2
#define RSA_NO_PADDING		3
#define RSA_PKCS1_OAEP_PADDING	4

#endif


/*
 * function arguments
*/
typedef struct VirtIOArg
{
	uint32_t cmd;
	uint32_t tid;
	uint64_t src;
	uint32_t srcSize;
	uint64_t dst;
	uint32_t dstSize;
	uint64_t flag;
	uint64_t param;
} VirtIOArg;
/* see ioctl-number in https://github.com/torvalds/
	linux/blob/master/Documentation/ioctl/ioctl-number.txt
*/
#define VIRTIO_IOC_ID '0xBB'

#define VIRTIO_IOC_HELLO \
	_IOWR(VIRTIO_IOC_ID,0,int)
/** module control	**/
#define VIRTIO_IOC_REGISTERFATBINARY \
	_IOWR(VIRTIO_IOC_ID,1, VirtIOArg)
#define VIRTIO_IOC_UNREGISTERFATBINARY	\
	_IOWR(VIRTIO_IOC_ID,2,VirtIOArg)
#define VIRTIO_IOC_REGISTERFUNCTION \
	_IOWR(VIRTIO_IOC_ID,3,VirtIOArg)
#define VIRTIO_IOC_LAUNCH \
	_IOWR(VIRTIO_IOC_ID,4,VirtIOArg)
/* memory management */
#define VIRTIO_IOC_MALLOC\
	_IOWR(VIRTIO_IOC_ID,5,VirtIOArg)
#define VIRTIO_IOC_MEMCPY \
	_IOWR(VIRTIO_IOC_ID,6,VirtIOArg)
#define VIRTIO_IOC_FREE \
	_IOWR(VIRTIO_IOC_ID,7,VirtIOArg)
/**	device management	**/
#define VIRTIO_IOC_GETDEVICE \
	_IOWR(VIRTIO_IOC_ID,8,VirtIOArg)
#define VIRTIO_IOC_GETDEVICEPROPERTIES \
	_IOWR(VIRTIO_IOC_ID,9,VirtIOArg)
#define VIRTIO_IOC_CONFIGURECALL \
	_IOWR(VIRTIO_IOC_ID,10,VirtIOArg)

#define VIRTIO_IOC_SETUPARGUMENT \
	_IOWR(VIRTIO_IOC_ID,11,VirtIOArg)
#define VIRTIO_IOC_GETDEVICECOUNT \
	_IOWR(VIRTIO_IOC_ID,12,VirtIOArg)
#define VIRTIO_IOC_SETDEVICE \
	_IOWR(VIRTIO_IOC_ID,13,VirtIOArg)
#define VIRTIO_IOC_DEVICERESET \
	_IOWR(VIRTIO_IOC_ID,14,VirtIOArg)
#define VIRTIO_IOC_STREAMCREATE \
	_IOWR(VIRTIO_IOC_ID,15,VirtIOArg)

#define VIRTIO_IOC_STREAMDESTROY \
	_IOWR(VIRTIO_IOC_ID,16,VirtIOArg)
#define VIRTIO_IOC_EVENTCREATE \
	_IOWR(VIRTIO_IOC_ID,17,VirtIOArg)
#define VIRTIO_IOC_EVENTDESTROY \
	_IOWR(VIRTIO_IOC_ID,18,VirtIOArg)
#define VIRTIO_IOC_EVENTRECORD \
	_IOWR(VIRTIO_IOC_ID,19,VirtIOArg)
#define VIRTIO_IOC_EVENTSYNCHRONIZE \
	_IOWR(VIRTIO_IOC_ID,20,VirtIOArg)

#define VIRTIO_IOC_EVENTELAPSEDTIME \
	_IOWR(VIRTIO_IOC_ID,21,VirtIOArg)
#define VIRTIO_IOC_THREADSYNCHRONIZE \
	_IOWR(VIRTIO_IOC_ID,22,VirtIOArg)
#define VIRTIO_IOC_GETLASTERROR \
	_IOWR(VIRTIO_IOC_ID,23,VirtIOArg)

#define VIRTIO_IOC_MEMCPY_ASYNC \
	_IOWR(VIRTIO_IOC_ID,24,VirtIOArg)
#define VIRTIO_IOC_MEMSET \
	_IOWR(VIRTIO_IOC_ID,25,VirtIOArg)
#define VIRTIO_IOC_DEVICESYNCHRONIZE \
	_IOWR(VIRTIO_IOC_ID,26,VirtIOArg)

#define VIRTIO_IOC_EVENTCREATEWITHFLAGS \
	_IOWR(VIRTIO_IOC_ID,27,VirtIOArg)
#define VIRTIO_IOC_HOSTREGISTER \
	_IOWR(VIRTIO_IOC_ID,28,VirtIOArg)
#define VIRTIO_IOC_HOSTGETDEVICEPOINTER \
	_IOWR(VIRTIO_IOC_ID,29,VirtIOArg)
#define VIRTIO_IOC_HOSTUNREGISTER \
	_IOWR(VIRTIO_IOC_ID,30,VirtIOArg)
#define VIRTIO_IOC_SETDEVICEFLAGS \
	_IOWR(VIRTIO_IOC_ID,31,VirtIOArg)
#define VIRTIO_IOC_MEMGETINFO \
	_IOWR(VIRTIO_IOC_ID,32,VirtIOArg)

#define VIRTIO_IOC_MALLOCHOST \
	_IOWR(VIRTIO_IOC_ID,33,VirtIOArg)


#endif