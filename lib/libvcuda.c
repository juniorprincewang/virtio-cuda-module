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
#define error(fmt, arg...) printf("[ERROR]: %s->line : %d. "fmt, __FUNCTION__, __LINE__, ##arg)
#define debug(fmt, arg...) printf("[DEBUG]: "fmt, ##arg)
#define print(fmt, arg...) printf("[+]INFO: "fmt, ##arg)
#define func() printf("[FUNC] Now in %s\n", __FUNCTION__);
#define MODE O_RDWR

#define ARG_LEN sizeof(VirtIOArg)

typedef struct KernelConf
{
	dim3 gridDim;
	dim3 blockDim;
	size_t sharedMem;
	cudaStream_t stream;
} KernelConf_t ;

uint32_t cudaKernelConf[8];
uint8_t cudaKernelPara[512];	// uint8_t === unsigned char
uint32_t cudaParaSize;		// uint32_t == unsigned int
KernelConf_t kernelConf;
int fd=-1;

/*
 * ioctl
*/
void send_to_device(int cmd, VirtIOArg *arg)
{
	func();
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
		debug("headerSize	=	%d(0x%x)\n", fatHeader->headerSize, fatHeader->headerSize);
		debug("fatSize	=	%lld(0x%llx)\n", fatHeader->fatSize, fatHeader->fatSize);
		// initialize arguments
		memset(&arg, 0, ARG_LEN);
		arg.srcSize = fatHeader->headerSize + fatHeader->fatSize;
		arg.src = (uint64_t)(binary->data);
		arg.dstSize = 0;
		arg.cmd = (VIRTIO_CUDA_REGISTERFATBINARY);
		arg.tid = syscall(SYS_gettid);
		send_to_device(VIRTIO_IOC_REGISTERFATBINARY, &arg);
		debug("	arg.cmd = %d\n", arg.cmd);
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

	send_to_device(VIRTIO_IOC_UNREGISTERFATBINARY, &arg);
	debug("	arg.cmd = %d\n", arg.cmd);
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
	arg.flag = (uint64_t)hostFun;
	// arg.totalSize = sizeof(VirtIOArg) + arg.srcSize + arg.dstSize;
	debug("	deviceName = %s\n", (char*)arg.dst);
	debug("	arg.srcSize = %d\n", arg.srcSize);
	debug("	arg.totalSize = %d\n", arg.totalSize);
	debug("	len of deviceName = %d\n", arg.dstSize );
	send_to_device(VIRTIO_IOC_REGISTERFUNCTION, &arg);
	debug("	arg.cmd = %d\n", arg.cmd);
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

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
	func();
	VirtIOArg arg;
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
// very important
	return cudaSuccess;
/*
	memset(cudaKernelConf, 0, sizeof(unsigned int)*8);
	cudaKernelConf[0] = gridDim.x;
	cudaKernelConf[1] = gridDim.y;
	cudaKernelConf[2] = gridDim.z;
	cudaKernelConf[3] = blockDim.x;
	cudaKernelConf[4] = blockDim.y;
	cudaKernelConf[5] = blockDim.z;
	cudaKernelConf[6] = sharedMem;
	cudaKernelConf[7] = (NULL == stream)? (unsigned int)0: (unsigned int)stream;
*/
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_CONFIGURECALL;
//	arg.src = (void*)cudaKernelConf;
//	arg.srcSize = sizeof(unsigned int)*8;
	arg.src = (void*)&kernelConf;
	arg.srcSize = sizeof(KernelConf_t);
	arg.totalSize = sizeof(VirtIOArg) + sizeof(KernelConf_t);

	// send fatbin to host
//	send_to_device(VIRTIO_IOC_CONFIGURECALL, &arg);



	debug("	arg.cmd = %d\n", arg.cmd);
	return arg.cmd;	
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
	//memcpy();
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
	func();
	VirtIOArg arg;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_LAUNCH;
	arg.src = (void*)&cudaKernelPara;
	arg.srcSize = cudaParaSize;
	arg.dst = (void*)&kernelConf;
	arg.dstSize = sizeof(KernelConf_t);
	arg.totalSize = sizeof(VirtIOArg) + arg.srcSize + arg.dstSize;
	arg.flag = (uint64_t)entry;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_LAUNCH, &arg);
	
	//return cudaSuccess;
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	func();
	VirtIOArg arg;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_MEMCPY;
	arg.flag = kind;
	arg.src = src;
	arg.srcSize = count;
	arg.dst = dst;
	arg.dstSize = 0;
	arg.totalSize = sizeof(VirtIOArg) + arg.srcSize + arg.dstSize;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMCPY, &arg);
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	func();
	VirtIOArg arg;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_MALLOC;
	arg.src = NULL;
	arg.srcSize = size;
	arg.totalSize = sizeof(VirtIOArg);
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MALLOC, &arg);
	*devPtr = arg.dst;
	debug("devPtr = %p\n", arg.dst);
	debug("	arg.cmd = %d\n", arg.cmd);
	
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaFree (void *devPtr)
{
	func();
	VirtIOArg arg;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_FREE;
	arg.src = devPtr;
	arg.srcSize = 0;
	arg.totalSize = sizeof(VirtIOArg);
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_FREE, &arg);
	debug("devPtr = %p\n", arg.src);
	debug("arg.cmd = %d\n", arg.cmd);
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
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_GETDEVICEPROPERTIES;
	arg.dst = (void *)prop;
	arg.dstSize = sizeof(struct cudaDeviceProp);
	arg.flag = device;
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg) + arg.dstSize;
	send_to_device(VIRTIO_IOC_GETDEVICEPROPERTIES, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaSetDevice(int device)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_SETDEVICE;
	arg.flag = device;
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg) ;
	send_to_device(VIRTIO_IOC_SETDEVICE, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaGetDeviceCount(int *count)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_GETDEVICECOUNT;
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg);
	send_to_device(VIRTIO_IOC_GETDEVICECOUNT, &arg);
	// do something
	*count = arg.flag;
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaDeviceReset(void)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_DEVICERESET;
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg) ;
	send_to_device(VIRTIO_IOC_DEVICERESET, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_STREAMCREATE;
	arg.tid = syscall(SYS_gettid);
	arg.src = (void*)pStream;
	arg.srcSize = sizeof(cudaStream_t);
	arg.totalSize = sizeof(VirtIOArg) + arg.srcSize;
	send_to_device(VIRTIO_IOC_STREAMCREATE, &arg);
	// do something
	//*pStream = (cudaStream_t)arg.flag;
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_STREAMDESTROY;
	arg.flag = (uint64_t)stream;
	arg.src = &stream;
	arg.srcSize = sizeof(cudaStream_t);
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg) + arg.srcSize;
	send_to_device(VIRTIO_IOC_STREAMDESTROY, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
cudaError_t cudaEventCreate(cudaEvent_t *event)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTCREATE;
	arg.src = event;
	arg.srcSize = sizeof(cudaEvent_t);
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg) + arg.srcSize;
	send_to_device(VIRTIO_IOC_EVENTCREATE, &arg);
	// do something
	 *event = (void*)(arg.flag);
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTDESTROY;
	arg.flag = (uint64_t)event;
	arg.tid = syscall(SYS_gettid);
	arg.src = &event;
	arg.srcSize = sizeof(cudaEvent_t);
	arg.totalSize = sizeof(VirtIOArg) +arg.srcSize;
	send_to_device(VIRTIO_IOC_EVENTDESTROY, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTRECORD;
	debug("event = %lu\n", (uint64_t)event);
	arg.src = &event;
	arg.param = (uint64_t)event;
	arg.srcSize = sizeof(cudaEvent_t); // sizeof(cudaEvent_t) = sizeof(uint64_t) = 8
	if(NULL == stream)
		arg.flag = (uint64_t)(-1);
	else
	{
		arg.flag = (uint64_t)stream;
		arg.dst = &stream;
		arg.dstSize = sizeof(cudaStream_t); // sizeof(cudaStream_t) = sizeof(uint64_t) = 8
	}
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg) +arg.srcSize + arg.dstSize;
	send_to_device(VIRTIO_IOC_EVENTRECORD, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
	VirtIOArg arg;
	func();
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTSYNCHRONIZE;
	arg.flag = (uint64_t)event;
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg) ;
	send_to_device(VIRTIO_IOC_EVENTSYNCHRONIZE, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
	VirtIOArg arg;
	func();
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTELAPSEDTIME;
	arg.tid = syscall(SYS_gettid);
	arg.flag = (uint64_t)start;
	arg.param = (uint64_t)end;
	arg.dst = (void*)ms;
	arg.dstSize = sizeof(float);
	arg.totalSize = sizeof(VirtIOArg) + arg.dstSize;
	send_to_device(VIRTIO_IOC_EVENTELAPSEDTIME, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
cudaError_t cudaThreadSynchronize()
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_THREADSYNCHRONIZE;
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg) ;
	send_to_device(VIRTIO_IOC_THREADSYNCHRONIZE, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
cudaError_t cudaGetLastError(void)
{
	func();
	VirtIOArg arg;
	// initialize arguments
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_GETLASTERROR;
	arg.tid = syscall(SYS_gettid);
	arg.totalSize = sizeof(VirtIOArg) ;
	send_to_device(VIRTIO_IOC_GETLASTERROR, &arg);
	// do something
	debug("	arg.cmd = %d\n", arg.cmd);
	return (cudaError_t)arg.cmd;	
}
