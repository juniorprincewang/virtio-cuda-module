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
#include <errno.h>	// errno
#include <sys/mman.h>	// mmap, PROT_READ, PROT_WRITE, MAP_SHARED
#include <assert.h> 	// assert

#define DEVICE_FILE "/dev/cudaport2p%d"

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
#define DEVICE_COUNT 32

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
static int fd=-1;			// fd of device file
static int device_count=0;	// virtual device number
static int minor = 0;		// get current device

extern void *__libc_malloc(size_t);
static unsigned int map_offset=0;
static size_t const BLOCK_MAGIC = 0xdeadbeaf;

typedef struct block_header
{
    void* address;
    size_t total_size;
    size_t data_size;
    size_t magic;
} BlockHeader;

void __attribute__ ((constructor)) my_init(void);
void __attribute__ ((destructor)) my_fini(void);

BlockHeader* get_block_by_ptr(void* p)
{
    void* ptr = (char*)p - sizeof(BlockHeader);
    BlockHeader* blk = (BlockHeader*)ptr;

    if (blk->magic != BLOCK_MAGIC)
    {
        // error("bad magic in block %p\n", p);
        return NULL;
    }

    return blk;
}

static size_t roundup(size_t n, size_t alignment)
{
    return (n + alignment - 1) / alignment * alignment;
}

static void *__mmalloc(size_t size)
{
	int alignment = 8;
    size_t data_start_offset = roundup(sizeof(BlockHeader), alignment);
    size_t header_start_offset = data_start_offset - sizeof(BlockHeader);
    size_t total_size = data_start_offset + size;

	unsigned int blocks_num = (total_size+KMALLOC_SIZE-1)/KMALLOC_SIZE;
	void *ptr = mmap(0, blocks_num * KMALLOC_SIZE, PROT_READ|PROT_WRITE, 
						MAP_SHARED, fd, map_offset);
	if(ptr == MAP_FAILED) {
		error("mmap failed, error: %s.\n", strerror(errno));
		return NULL;
	}
	func();
	map_offset += blocks_num*KMALLOC_SIZE;

	BlockHeader* blk 	= (BlockHeader*)((char*)ptr + header_start_offset);
    blk->address 		= ptr;
    blk->total_size    	= blocks_num * KMALLOC_SIZE;
    blk->data_size     	= size;
    blk->magic         	= BLOCK_MAGIC;

	msync(ptr, blocks_num*KMALLOC_SIZE, MS_ASYNC);
    return (char*)ptr + data_start_offset;
}

void *malloc(size_t size)
{
	if (size > KMALLOC_SIZE)
		return __mmalloc(size);
	return __libc_malloc(size);
}

void free(void *ptr)
{
	if (ptr == NULL)
        return;
    
    BlockHeader* blk = get_block_by_ptr(ptr);
    if(!blk) {
		__libc_free(ptr);
		return;
    }
    munmap(blk->address, blk->total_size);
}

/*
 * ioctl
*/
void send_to_device(int cmd, VirtIOArg *arg)
{
	if(ioctl(fd, cmd, arg) == -1){
		error("ioctl when cmd is %d\n", _IOC_NR(cmd));
	}
}

int get_vdevice_count(int *result)
{
	char fname[128]="/proc/virtio-cuda/virtual_device_count";
	char buf[15];
	int size=0;
	int fdv=0;
	int count=0;
	fdv=open(fname, O_RDONLY);
	if (fdv<0) {
		error("open device %s failed, %s (%d)\n", 
			fname, (char*)strerror(errno), errno);
		return -ENODEV;
	}
	if((size=read(fdv, buf, 16))<0) {
		error("read error!\n");
		return -ENODEV;
	}
	close(fdv);
	sscanf(buf, "%d", &count);
	*result = count;
	return 0;
}

/*
 * open token
*/
int open_vdevice()
{
	char devname[32];
	int i=0;
	if(get_vdevice_count(&device_count) < 0) {
		error("Cannot find valid device.\n");
		return -ENODEV;
	}
	debug("device_count=%d\n", device_count);
	for(i=1; i<=device_count; i++) {
		sprintf(devname, DEVICE_FILE, i);
		fd = open(devname, MODE);
		if(fd>= 0)
			break;
		else if(errno==EBUSY) {
			debug("device %d is busy\n", i);
			continue;
		}
		else
			error("open device "DEVICE_FILE" failed, %s (%d)", 
				i, (char *)strerror(errno), errno);
	}
	if(i > device_count) {
		error("Failed to find valid device file.\n");
		return -EINVAL;
	}
	minor = i;
	debug("fd is %d\n", fd);
	return 0;
}

void close_vdevice()
{
	func();
	close(fd);
	debug("closing fd\n");
}


void my_init(void) {
	debug("Init dynamic library.\n");
	if(open_vdevice() < 0)
		exit(-1);
}

void my_fini(void) {
	debug("deinit dynamic library\n");
	close_vdevice();
}

void** __cudaRegisterFatBinary(void *fatCubin)
{
	VirtIOArg arg;
	unsigned int magic;
	unsigned long long **fatCubinHandle;

	func();
	
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
		debug("FatBinHeader = %p\n", fatHeader);
		debug("magic	=	0x%x\n", fatHeader->magic);
		debug("version	=	0x%x\n", fatHeader->version);
		debug("headerSize =	%d(0x%x)\n", fatHeader->headerSize, fatHeader->headerSize);
		debug("fatSize	=	%lld(0x%llx)\n", fatHeader->fatSize, fatHeader->fatSize);
		// initialize arguments
		memset(&arg, 0, ARG_LEN);
		arg.src 	= (uint64_t)(binary->data);
		arg.srcSize = fatHeader->headerSize + fatHeader->fatSize;
		arg.dstSize = 0;
		arg.cmd 	= VIRTIO_CUDA_REGISTERFATBINARY;
		arg.tid 	= syscall(SYS_gettid);
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
	if (fatCubinHandle != NULL)
		free(fatCubinHandle);
}

void __cudaRegisterFatBinaryEnd(
  void **fatCubinHandle
)
{
	func();
	return ;
}


void __cudaRegisterFunction(
	void 		**fatCubinHandle,
	const char	*hostFun,
	char		*deviceFun,
	const char	*deviceName,
	int			thread_limit,
	uint3		*tid,
	uint3		*bid,
	dim3		*bDim,
	dim3		*gDim,
	int			*wSize
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
	arg.cmd 	= VIRTIO_CUDA_REGISTERFUNCTION;
	arg.src 	= (uint64_t)fatBinHeader;
	arg.srcSize = fatBinHeader->fatSize + fatBinHeader->headerSize;
	arg.dst 	= (uint64_t)deviceName;
	arg.dstSize = strlen(deviceName)+1; // +1 in order to keep \x00
	arg.tid 	= syscall(SYS_gettid);
	arg.flag 	= (uint64_t)hostFun;
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
	VirtIOArg arg;
	computeFatBinaryFormat_t fatBinHeader;
	func();

	fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);
	debug("	fatCubinHandle = %p, value =%p\n", fatCubinHandle, *fatCubinHandle);
	debug("	hostVar = %p, value =%p\n", hostVar, *hostVar);
	debug("	deviceAddress = %p, value = %s\n", deviceAddress, deviceAddress);
	debug("	deviceName = %s\n", deviceName);
	debug("	ext = %d\n", ext);
	debug("	size = %d\n", size);
	debug("	constant = %d\n", constant);
	debug("	global = %d\n", global);
	memset(&arg, 0, ARG_LEN);
	arg.cmd 	= VIRTIO_CUDA_REGISTERVAR;
	arg.src 	= (uint64_t)fatBinHeader;
	arg.srcSize = fatBinHeader->fatSize + fatBinHeader->headerSize;
	arg.dst 	= (uint64_t)deviceName;
	arg.dstSize = strlen(deviceName)+1; // +1 in order to keep \x00
	arg.tid 	= syscall(SYS_gettid);
	arg.flag 	= (uint64_t)hostVar;
	arg.param 	= (uint64_t)constant;
	arg.param2 	= (uint64_t)global;
	send_to_device(VIRTIO_IOC_REGISTERVAR, &arg);
	if(arg.cmd != cudaSuccess)
	{
		error("	functions are not registered successfully.\n");
	}
}

void __cudaRegisterManagedVar(
	void **fatCubinHandle,
	void **hostVarPtrAddress,
	char  *deviceAddress,
	const char  *deviceName,
	int    ext,
	size_t size,
	int    constant,
	int    global
)
{
	func();
	debug("Undefined\n");
}

void __cudaRegisterTexture(
	void                    **fatCubinHandle,
	const struct textureReference  *hostVar,
	const void                    **deviceAddress,
	const char                     *deviceName,
	int                       dim,
	int                       norm,
	int                        ext
)
{
	func();
	debug("Undefined\n");
}

void __cudaRegisterSurface(
	void                    **fatCubinHandle,
	const struct surfaceReference  *hostVar,
	const void                    **deviceAddress,
	const char                     *deviceName,
	int                       dim,
	int                       ext
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

cudaError_t  __cudaPopCallConfiguration(
  dim3         *gridDim,
  dim3         *blockDim,
  size_t       *sharedMem,
  void         *stream
)
{
	func();
	debug("Undefined\n");
	return cudaSuccess;
}

unsigned  __cudaPushCallConfiguration(
	dim3         gridDim,
	dim3         blockDim,
	size_t sharedMem,
	struct CUstream_st *stream
)
{
	func();
	debug("gridDim= %u %u %u\n", gridDim.x, gridDim.y, gridDim.z);	
	debug("blockDim= %u %u %u\n", blockDim.x, blockDim.y, blockDim.z);
	debug("sharedMem= %zu\n", sharedMem);
	// debug("stream= %lu\n", (cudaStream_t)(stream));
	
	memset(cudaKernelPara, 0, 512);
	cudaParaSize = sizeof(uint32_t);

	memset(&kernelConf, 0, sizeof(KernelConf_t));
	kernelConf.gridDim 		= gridDim;
	kernelConf.blockDim 	= blockDim;
	kernelConf.sharedMem 	= sharedMem;
	kernelConf.stream 		= (cudaStream_t)stream;
	return 0;
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
	kernelConf.gridDim 		= gridDim;
	kernelConf.blockDim 	= blockDim;
	kernelConf.sharedMem 	= sharedMem;
	kernelConf.stream 		= stream;
	// Do not invoke ioctl
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
	debug("value = 0x%llx\n", *(unsigned long long*)&cudaKernelPara[cudaParaSize]);
	cudaParaSize += size;
	(*((uint32_t*)cudaKernelPara))++;
	return cudaSuccess;
}

cudaError_t cudaLaunchKernel(
	const void *func,
	dim3 gridDim,
	dim3 blockDim,
	void **args,
	size_t sharedMem,
	cudaStream_t stream
)
{
	uint32_t fid;
	func();
	fid=(uint32_t)func;
	debug("func id = %u\n", fid);
	debug("szieof(args)=%lu\n", sizeof(args));
	debug("szieof(args[0])=%lu\n", sizeof(args[0]));

	return cudaSuccess;
}

cudaError_t cudaLaunch(const void *entry)
{
	VirtIOArg arg;
	void *para;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_LAUNCH;
	arg.src = (uint64_t)&cudaKernelPara;
	para = (void*)arg.src;
	int para_idx = sizeof(uint32_t);
	int para_num = *(uint32_t*)para;
	debug("para_num=%d\n", para_num);
	for(int i=0; i<para_num; i++) {
		debug("i=%d\n", i);
		debug("size = %u\n", *(uint32_t*)&para[para_idx]);
		if (*(uint32_t*)&para[para_idx]==8)
			debug("value=%llx\n",*(unsigned long long*)&para[para_idx+sizeof(uint32_t)]);
		else
			debug("value=%llx\n",*(unsigned int*)&para[para_idx+sizeof(uint32_t)]);
		para_idx += *(uint32_t*)&para[para_idx] + sizeof(uint32_t);
	}

	arg.srcSize = cudaParaSize;
	arg.dst 	= (uint64_t)&kernelConf;
	arg.dstSize = sizeof(KernelConf_t);
	arg.flag 	= (uint64_t)entry;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_LAUNCH, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_MEMCPY;
	arg.flag 	= kind;
	arg.src 	= (uint64_t)src;
	arg.srcSize = count;
	arg.dst 	= (uint64_t)dst;
	arg.dstSize = count;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMCPY, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMemcpyToSymbol(	const void *symbol, const void *src, 
								size_t count, size_t offset, enum cudaMemcpyKind kind)
{
	VirtIOArg arg;
	func();
	assert(kind == cudaMemcpyHostToDevice);
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_MEMCPYTOSYMBOL;
	arg.flag 	= kind;
	arg.src 	= (uint64_t)src;
	arg.srcSize = count;
	debug("symbol is %p\n", symbol);
	arg.dst 	= (uint64_t)symbol;
	arg.dstSize = 0;
	arg.param 	= offset;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMCPYTOSYMBOL, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMemcpyFromSymbol(	void *dst, const void *symbol, 
									size_t count, size_t offset, enum cudaMemcpyKind kind)
{
	VirtIOArg arg;
	func();
	assert(kind == cudaMemcpyDeviceToHost);
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_MEMCPYFROMSYMBOL;
	arg.flag 	= kind;
	debug("symbol is %p\n", symbol);
	arg.src 	= (uint64_t)symbol;
	arg.dst 	= (uint64_t)dst;
	arg.dstSize = count;
	arg.param 	= offset;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMCPYFROMSYMBOL, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMemset(void *dst, int value, size_t count)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_MEMSET;
	arg.param 	= (uint64_t)value;
	arg.dst 	= (uint64_t)dst;
	arg.dstSize = count;
	arg.tid 	= syscall(SYS_gettid);
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
	arg.cmd 	= VIRTIO_CUDA_MEMCPY_ASYNC;
	arg.flag 	= kind;
	arg.src 	= (uint64_t)src;
	arg.srcSize = count;
	arg.dst 	= (uint64_t)dst;
	arg.dstSize = (uint32_t)stream;
	arg.param 	= (uint64_t)stream;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMCPY_ASYNC, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_MALLOC;
	arg.src 	= (uint64_t)NULL;
	arg.srcSize = size;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MALLOC, &arg);
	*devPtr = (void *)arg.dst;
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_HOSTREGISTER;
	arg.tid 	= syscall(SYS_gettid);
	arg.src 	= (uint64_t)ptr;
	arg.srcSize = size;
	arg.flag 	= flags;
	send_to_device(VIRTIO_IOC_HOSTREGISTER, &arg);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaHostUnregister(void *ptr)
{
	VirtIOArg arg;
	func();

	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_HOSTUNREGISTER;
	arg.tid = syscall(SYS_gettid);
	arg.src = (uint64_t)ptr;
	send_to_device(VIRTIO_IOC_HOSTUNREGISTER, &arg);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	func();
	*pHost = __mmalloc(size);
	return cudaHostRegister(*pHost, size, flags);
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
	func();
	return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

cudaError_t cudaFreeHost(void *ptr)
{
	BlockHeader* blk = NULL;
	func();
	cudaError_t err = cudaHostUnregister(ptr);
	blk = get_block_by_ptr(ptr);
    if(!blk) {
		return cudaErrorInitializationError;
    }
    munmap(blk->address, blk->total_size);
	return err;
}

cudaError_t cudaFree(void *devPtr)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_FREE;
	arg.src 	= (uint64_t)devPtr;
	arg.srcSize = 0;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_FREE, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaGetDevice(int *device)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_GETDEVICE;
	arg.dst 	= (uint64_t)device;
	arg.dstSize = sizeof(int);
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_GETDEVICE, &arg);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_GETDEVICEPROPERTIES;
	arg.dst 	= (uint64_t)prop;
	arg.dstSize = sizeof(struct cudaDeviceProp);
	arg.flag 	= device;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_GETDEVICEPROPERTIES, &arg);
	if(!prop)
		return cudaErrorInvalidDevice;
	return cudaSuccess;
}

cudaError_t cudaSetDevice(int device)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_SETDEVICE;
	arg.flag 	= device;
	arg.tid 	= syscall(SYS_gettid);
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
	*count  = arg.flag;
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
	return cudaSuccess;	
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
	arg.cmd 	= VIRTIO_CUDA_STREAMDESTROY;
	arg.flag 	= (uint64_t)stream;
	arg.tid 	= syscall(SYS_gettid);
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
	arg.cmd 	= VIRTIO_CUDA_EVENTCREATEWITHFLAGS;
	arg.flag 	= flags;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTCREATEWITHFLAGS, &arg);
	 *event 	= (void*)(arg.dst);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_EVENTDESTROY;
	arg.flag 	= (uint64_t)event;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTDESTROY, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTRECORD;
	debug("event  = 0x%lx\n", (uint64_t)event);
	debug("stream = 0x%lx\n", (uint64_t)stream);
	arg.src = (uint64_t)event;
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
	arg.cmd 	= VIRTIO_CUDA_EVENTSYNCHRONIZE;
	arg.flag 	= (uint64_t)event;
	arg.tid 	= syscall(SYS_gettid);
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
	*ms 	= (float)arg.flag;
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

cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_MEMGETINFO;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMGETINFO, &arg);
	*free  	= (size_t)arg.srcSize;
	*total 	= (size_t)arg.dstSize;
	return (cudaError_t)arg.cmd;
}


cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_SETDEVICEFLAGS;
	arg.tid 	= syscall(SYS_gettid);
	arg.flag 	= (uint64_t)flags;
	send_to_device(VIRTIO_IOC_SETDEVICEFLAGS, &arg);
	return (cudaError_t)arg.cmd;
}

const char *cudaGetErrorString(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";

        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";

        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";

        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";

        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";

        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";

        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";

        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";

        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";

        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";

        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";

        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";

        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";

        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";

        case cudaErrorMapBufferObjectFailed:
            return "cudaErrorMapBufferObjectFailed";

        case cudaErrorUnmapBufferObjectFailed:
            return "cudaErrorUnmapBufferObjectFailed";

        case cudaErrorInvalidHostPointer:
            return "cudaErrorInvalidHostPointer";

        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";

        case cudaErrorInvalidTexture:
            return "cudaErrorInvalidTexture";

        case cudaErrorInvalidTextureBinding:
            return "cudaErrorInvalidTextureBinding";

        case cudaErrorInvalidChannelDescriptor:
            return "cudaErrorInvalidChannelDescriptor";

        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";

        case cudaErrorAddressOfConstant:
            return "cudaErrorAddressOfConstant";

        case cudaErrorTextureFetchFailed:
            return "cudaErrorTextureFetchFailed";

        case cudaErrorTextureNotBound:
            return "cudaErrorTextureNotBound";

        case cudaErrorSynchronizationError:
            return "cudaErrorSynchronizationError";

        case cudaErrorInvalidFilterSetting:
            return "cudaErrorInvalidFilterSetting";

        case cudaErrorInvalidNormSetting:
            return "cudaErrorInvalidNormSetting";

        case cudaErrorMixedDeviceExecution:
            return "cudaErrorMixedDeviceExecution";

        case cudaErrorCudartUnloading:
            return "cudaErrorCudartUnloading";

        case cudaErrorUnknown:
            return "cudaErrorUnknown";

        case cudaErrorNotYetImplemented:
            return "cudaErrorNotYetImplemented";

        case cudaErrorMemoryValueTooLarge:
            return "cudaErrorMemoryValueTooLarge";

        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";

        case cudaErrorNotReady:
            return "cudaErrorNotReady";

        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver";

        case cudaErrorSetOnActiveProcess:
            return "cudaErrorSetOnActiveProcess";

        case cudaErrorInvalidSurface:
            return "cudaErrorInvalidSurface";

        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";

        case cudaErrorECCUncorrectable:
            return "cudaErrorECCUncorrectable";

        case cudaErrorSharedObjectSymbolNotFound:
            return "cudaErrorSharedObjectSymbolNotFound";

        case cudaErrorSharedObjectInitFailed:
            return "cudaErrorSharedObjectInitFailed";

        case cudaErrorUnsupportedLimit:
            return "cudaErrorUnsupportedLimit";

        case cudaErrorDuplicateVariableName:
            return "cudaErrorDuplicateVariableName";

        case cudaErrorDuplicateTextureName:
            return "cudaErrorDuplicateTextureName";

        case cudaErrorDuplicateSurfaceName:
            return "cudaErrorDuplicateSurfaceName";

        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";

        case cudaErrorInvalidKernelImage:
            return "cudaErrorInvalidKernelImage";

        case cudaErrorNoKernelImageForDevice:
            return "cudaErrorNoKernelImageForDevice";

        case cudaErrorIncompatibleDriverContext:
            return "cudaErrorIncompatibleDriverContext";

        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";

        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";

        case cudaErrorDeviceAlreadyInUse:
            return "cudaErrorDeviceAlreadyInUse";

        case cudaErrorProfilerDisabled:
            return "cudaErrorProfilerDisabled";

        case cudaErrorProfilerNotInitialized:
            return "cudaErrorProfilerNotInitialized";

        case cudaErrorProfilerAlreadyStarted:
            return "cudaErrorProfilerAlreadyStarted";

        case cudaErrorProfilerAlreadyStopped:
            return "cudaErrorProfilerAlreadyStopped";

        /* Since CUDA 4.0*/
        case cudaErrorAssert:
            return "cudaErrorAssert";

        case cudaErrorTooManyPeers:
            return "cudaErrorTooManyPeers";

        case cudaErrorHostMemoryAlreadyRegistered:
            return "cudaErrorHostMemoryAlreadyRegistered";

        case cudaErrorHostMemoryNotRegistered:
            return "cudaErrorHostMemoryNotRegistered";

        /* Since CUDA 5.0 */
        case cudaErrorOperatingSystem:
            return "cudaErrorOperatingSystem";

        case cudaErrorPeerAccessUnsupported:
            return "cudaErrorPeerAccessUnsupported";

        case cudaErrorLaunchMaxDepthExceeded:
            return "cudaErrorLaunchMaxDepthExceeded";

        case cudaErrorLaunchFileScopedTex:
            return "cudaErrorLaunchFileScopedTex";

        case cudaErrorLaunchFileScopedSurf:
            return "cudaErrorLaunchFileScopedSurf";

        case cudaErrorSyncDepthExceeded:
            return "cudaErrorSyncDepthExceeded";

        case cudaErrorLaunchPendingCountExceeded:
            return "cudaErrorLaunchPendingCountExceeded";

        case cudaErrorNotPermitted:
            return "cudaErrorNotPermitted";

        case cudaErrorNotSupported:
            return "cudaErrorNotSupported";

        /* Since CUDA 6.0 */
        case cudaErrorHardwareStackError:
            return "cudaErrorHardwareStackError";

        case cudaErrorIllegalInstruction:
            return "cudaErrorIllegalInstruction";

        case cudaErrorMisalignedAddress:
            return "cudaErrorMisalignedAddress";

        case cudaErrorInvalidAddressSpace:
            return "cudaErrorInvalidAddressSpace";

        case cudaErrorInvalidPc:
            return "cudaErrorInvalidPc";

        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";

        /* Since CUDA 6.5*/
        case cudaErrorInvalidPtx:
            return "cudaErrorInvalidPtx";

        case cudaErrorInvalidGraphicsContext:
            return "cudaErrorInvalidGraphicsContext";

        case cudaErrorStartupFailure:
            return "cudaErrorStartupFailure";

        case cudaErrorApiFailureBase:
            return "cudaErrorApiFailureBase";

        /* Since CUDA 8.0*/        
        case cudaErrorNvlinkUncorrectable :   
            return "cudaErrorNvlinkUncorrectable";

        /* Since CUDA 8.5*/        
        case cudaErrorJitCompilerNotFound :   
            return "cudaErrorJitCompilerNotFound";

        /* Since CUDA 9.0*/
        case cudaErrorCooperativeLaunchTooLarge :
            return "cudaErrorCooperativeLaunchTooLarge";

    }

    return "<unknown>";
}