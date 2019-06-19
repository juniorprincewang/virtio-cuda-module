#define _GNU_SOURCE	// RTLD_NEXT
#include <dlfcn.h> // dlsym
#include <cuda.h>
#include <cuda_runtime.h>
#include <fatBinaryCtl.h>

#include <string.h> // memset
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>	//open
#include <unistd.h>	// close
#include <sys/syscall.h> // SYS_gettid
#include "../virtio-ioc.h"
#include <errno.h>

#define DEVICE_PATH "/dev/cudaport2p1"

// #define VIRTIO_CUDA_DEBUG

#ifdef VIRTIO_CUDA_DEBUG
#define error(fmt, arg...) printf("[ERROR]: %s->line : %d. "fmt, __FUNCTION__, __LINE__, ##arg)
#define debug(fmt, arg...) printf("[DEBUG]: "fmt, ##arg)
#define func() printf("[FUNC] Now in %s\n", __FUNCTION__);
#else

#define error(fmt, arg...) 
#define debug(fmt, arg...) 
#define func() 

#endif

#define MODE O_RDWR

#define ARG_LEN sizeof(VirtIOArg)

typedef struct KernelConf
{
	dim3 gridDim;
	dim3 blockDim;
	size_t sharedMem;
	cudaStream_t stream;
} KernelConf_t ;

static uint8_t cudaKernelPara[512];	// uint8_t === unsigned char
static uint32_t cudaParaSize;		// uint32_t == unsigned int
static KernelConf_t kernelConf;
static int fd=-1;

/*
 * ioctl
*/
void send_to_device(int cmd, VirtIOArg *arg)
{
	if(ioctl(fd, cmd, arg) == -1){
		error("ioctl when cmd is %d\n", _IOC_NR(cmd));
	}
}

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

void** __cudaRegisterFatBinary(void *fatCubin)
{
	VirtIOArg arg;
	unsigned int magic;
	unsigned long long **fatCubinHandle;

	func();
	open_vdevice();
	fatCubinHandle = (unsigned long long**)malloc(sizeof(unsigned long long*));
	magic = *(unsigned int*)fatCubin;
	if (magic == FATBINC_MAGIC)
	{
		__fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t*)fatCubin;
		debug("FatBin\n");
		debug("magic	=	0x%x\n", binary->magic);
		debug("version	=	0x%x\n", binary->version);
		debug("data	=	%p\n", binary->data);
		debug("filename_or_fatbins	=	%p\n", binary->filename_or_fatbins);
		*fatCubinHandle = (unsigned long long*)binary->data;
		struct fatBinaryHeader *fatHeader = (struct fatBinaryHeader*)binary->data;
		debug("FatBinHeader\n");
		debug("magic	=	0x%x\n", fatHeader->magic);
		debug("version	=	0x%x\n", fatHeader->version);
		debug("headerSize =	%d(0x%x)\n", fatHeader->headerSize, fatHeader->headerSize);
		debug("fatSize	=	%lld(0x%llx)\n", fatHeader->fatSize, fatHeader->fatSize);
		// initialize arguments
		memset(&arg, 0, ARG_LEN);
		arg.srcSize = fatHeader->headerSize + fatHeader->fatSize;
		arg.src = (uint64_t)(binary->data);
		arg.dstSize = 0;
		arg.cmd = (VIRTIO_CUDA_REGISTERFATBINARY);
		arg.tid = syscall(SYS_gettid);
		send_to_device(VIRTIO_IOC_REGISTERFATBINARY, &arg);
		if(arg.cmd != cudaSuccess)
		{
			error("	fatbin not registered\n");
			exit(-1);
		}
		return (void **)fatCubinHandle;
	}
	else
	{
		error("Unrecongnized CUDA FAT MAGIC 0x%x\n", magic);
		exit(-1);
	}
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, ARG_LEN);
	arg.cmd = VIRTIO_CUDA_UNREGISTERFATBINARY;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_UNREGISTERFATBINARY, &arg);
	close_vdevice();
	if (fatCubinHandle != NULL)
		free(fatCubinHandle);
}

void __cudaRegisterFunction(
	void 		**fatCubinHandle,
	const char	*hostFun,
	char		*deviceFun,
	const char	*deviceName,
	int		thread_limit,
	uint3		*tid,
	uint3		*bid,
	dim3		*bDim,
	dim3		*gDim,
	int		*wSize
)
{
	VirtIOArg arg;
	computeFatBinaryFormat_t fatBinHeader;
	func();

	fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);
//	debug("	fatbin magic= 0x%x\n", fatBinHeader->magic);
//	debug("	fatbin version= %d\n", fatBinHeader->version);
//	debug("	fatbin headerSize= 0x%x\n", fatBinHeader->headerSize);
//	debug("	fatbin fatSize= 0x%llx\n", fatBinHeader->fatSize);
	debug("	fatCubinHandle = %p, value =%p\n", fatCubinHandle, *fatCubinHandle);
	debug("	hostFun = %s, value =%p\n", hostFun, hostFun);
	debug("	deviceFun =%s, %p\n", deviceFun, deviceFun);
	debug("	deviceName = %s\n", deviceName);
	debug("	thread_limit = %d\n", thread_limit);
	memset(&arg, 0, ARG_LEN);
	arg.cmd = VIRTIO_CUDA_REGISTERFUNCTION;
	arg.src = (uint64_t)fatBinHeader;
	arg.srcSize = fatBinHeader->fatSize + fatBinHeader->headerSize;
	arg.dst = (uint64_t)deviceName;
	arg.dstSize = strlen(deviceName)+1; // +1 in order to keep \x00
	arg.tid = syscall(SYS_gettid);
	arg.flag = (uint64_t)hostFun;
	debug("	deviceName = %s\n", (char*)arg.dst);
	debug("	arg.srcSize = %d\n", arg.srcSize);
	debug("	len of deviceName = %d\n", arg.dstSize );
	send_to_device(VIRTIO_IOC_REGISTERFUNCTION, &arg);
	if(arg.cmd != cudaSuccess)
	{
		error("	functions are not registered successfully.\n");
		exit(-1);
	}
	return;
}

void __cudaRegisterVar(
	void **fatCubinHandle,
	char *hostVar,
	char *deviceAddress,
	const char *deviceName,
	int ext,
	int size,
	int constant,
	int global
)
{
	func();
	debug("Undefined\n");
}

char __cudaInitModule(void **fatCubinHandle)
{
	func();
	return 'U';
}

cudaError_t cudaConfigureCall(
	dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
	// VirtIOArg arg;
	func();
	debug("gridDim= %u %u %u\n", gridDim.x, gridDim.y, gridDim.z);	
	debug("blockDim= %u %u %u\n", blockDim.x, blockDim.y, blockDim.z);
	debug("sharedMem= %zu\n", sharedMem);
	debug("stream= %lu\n", (uint64_t)stream);
	
	memset(cudaKernelPara, 0, 512);
	cudaParaSize = sizeof(uint32_t);

	memset(&kernelConf, 0, sizeof(KernelConf_t));
	kernelConf.gridDim = gridDim;
	kernelConf.blockDim = blockDim;
	kernelConf.sharedMem = sharedMem;
	kernelConf.stream = stream;
	// Not invoke ioctl
	return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset)
{
	func();
	debug(" arg=%p\n", arg);
	debug(" size=%zu\n", size);
	debug(" offset=%zu\n", offset);
/*
	cudaKernelPara format
	| #arg | arg1 size| arg1 | arg2 size | arg2 ...|
*/
	memcpy(&cudaKernelPara[cudaParaSize], &size, sizeof(uint32_t));
	debug("size = %u\n", *(uint32_t*)&cudaKernelPara[cudaParaSize]);
	cudaParaSize += sizeof(uint32_t);
	
	memcpy(&cudaKernelPara[cudaParaSize], arg, size);
	debug("value = %llx\n", *(unsigned long long*)&cudaKernelPara[cudaParaSize]);
	cudaParaSize += size;
	(*((uint32_t*)cudaKernelPara))++;
	return cudaSuccess;
}

cudaError_t cudaLaunch(const void *entry)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_LAUNCH;
	arg.src = (uint64_t)&cudaKernelPara;
	arg.srcSize = cudaParaSize;
	arg.dst = (uint64_t)&kernelConf;
	arg.dstSize = sizeof(KernelConf_t);
	arg.flag = (uint64_t)entry;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_LAUNCH, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_MEMCPY;
	arg.flag = kind;
	arg.src = (uint64_t)src;
	arg.srcSize = count;
	arg.dst = (uint64_t)dst;
	arg.dstSize = count;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMCPY, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMemset(void *dst, int value, size_t count)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_MEMSET;
	arg.param = (uint64_t)value;
	arg.dst = (uint64_t)dst;
	arg.dstSize = count;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMSET, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMemcpyAsync(
			void *dst, 
			const void *src, 
			size_t count, 
			enum cudaMemcpyKind kind,
			cudaStream_t stream
			)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_MEMCPY_ASYNC;
	arg.flag = kind;
	arg.src = (uint64_t)src;
	arg.srcSize = count;
	arg.dst = (uint64_t)dst;
	arg.dstSize = count;
	arg.param = (uint64_t)stream;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMCPY_ASYNC, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_MALLOC;
	arg.src = (uint64_t)NULL;
	arg.srcSize = size;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MALLOC, &arg);
	*devPtr = (void *)arg.dst;
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaFree (void *devPtr)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_FREE;
	arg.src = (uint64_t)devPtr;
	arg.srcSize = 0;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_FREE, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaGetDevice (int *device)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_GETDEVICE;
	arg.dst = (uint64_t)device;
	arg.dstSize = sizeof(int);
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_GETDEVICE, &arg);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_GETDEVICEPROPERTIES;
	arg.dst = (uint64_t)prop;
	arg.dstSize = sizeof(struct cudaDeviceProp);
	arg.flag = device;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_GETDEVICEPROPERTIES, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaSetDevice(int device)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_SETDEVICE;
	arg.flag = device;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_SETDEVICE, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaGetDeviceCount(int *count)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_GETDEVICECOUNT;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_GETDEVICECOUNT, &arg);
	*count = arg.flag;
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaDeviceReset(void)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_DEVICERESET;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_DEVICERESET, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaDeviceSynchronize(void)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_DEVICESYNCHRONIZE;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_DEVICESYNCHRONIZE, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_STREAMCREATE;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_STREAMCREATE, &arg);
	*pStream = (cudaStream_t)arg.flag;
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_STREAMDESTROY;
	arg.flag = (uint64_t)stream;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_STREAMDESTROY, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTCREATE;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTCREATE, &arg);
	 *event = (void*)(arg.flag);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTCREATEWITHFLAGS;
	arg.flag = flags;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTCREATEWITHFLAGS, &arg);
	 *event = (void*)(arg.dst);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTDESTROY;
	arg.flag = (uint64_t)event;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTDESTROY, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTRECORD;
	debug("event = %lu\n", (uint64_t)event);
	arg.src = (uint64_t)event;
	if(NULL == stream)
		arg.dst = (uint64_t)(-1);
	else
		arg.dst = (uint64_t)stream;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTRECORD, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTSYNCHRONIZE;
	arg.flag = (uint64_t)event;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTSYNCHRONIZE, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTELAPSEDTIME;
	arg.tid = syscall(SYS_gettid);
	arg.src = (uint64_t)start;
	arg.dst = (uint64_t)end;
	send_to_device(VIRTIO_IOC_EVENTELAPSEDTIME, &arg);
	*ms = (float)arg.flag;
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaThreadSynchronize()
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_THREADSYNCHRONIZE;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_THREADSYNCHRONIZE, &arg);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaGetLastError(void)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_GETLASTERROR;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_GETLASTERROR, &arg);
	return (cudaError_t)arg.cmd;
}
