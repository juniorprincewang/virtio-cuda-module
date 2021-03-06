#define _GNU_SOURCE	// RTLD_NEXT
#include <dlfcn.h> // dlsym
#include <cuda.h>
#include <cuda_runtime.h>
#include <fatBinaryCtl.h>
#include <cublas_v2.h>
#include <curand.h>

#include <string.h> // memset
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>	//open
#include <unistd.h>	// close, syscall
#include <sys/syscall.h> // SYS_gettid
#include "../virtio-ioc.h"
#include <errno.h>	// errno
#include <sys/mman.h>	// mmap, PROT_READ, PROT_WRITE, MAP_SHARED
#include <assert.h> 	// assert
#include <pthread.h>

static int global = 0;

#define DEVICE_FILE "/dev/cudaport2p%d"

// #define VIRTIO_CUDA_DEBUG

#ifdef VIRTIO_CUDA_DEBUG
#define debug(fmt, arg...) printf("[DEBUG]: "fmt, ##arg)
#define func() printf("[FUNC] Now in %s\n", __FUNCTION__);
#else
	#define debug(fmt, arg...) 
	#define func() 
#endif

#define error(fmt, arg...) printf("[ERROR]: %s->line : %d. "fmt, __FUNCTION__, __LINE__, ##arg)

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
extern void __libc_free(void *ptr);
static unsigned int map_offset=0;
static size_t const BLOCK_MAGIC = 0xdeadbeaf;

typedef struct block_header
{
	void* address;
	size_t total_size;
	size_t data_size;
	size_t magic;
} BlockHeader;

__attribute__ ((constructor)) void my_init(void);
__attribute__ ((destructor)) void my_fini(void);

static pthread_spinlock_t lock;

/*
 * ioctl
*/
void send_to_device(int cmd, void *arg)
{
	#ifdef VIRTIO_LOCK_USER
	pthread_spin_lock(&lock);
	#endif
	if(ioctl(fd, cmd, arg) == -1){
		error("ioctl when cmd is %d\n", _IOC_NR(cmd));
	}
	#ifdef VIRTIO_LOCK_USER
	pthread_spin_unlock(&lock);
	#endif
}

BlockHeader* get_block_by_ptr(void* p)
{
	void* ptr = (char*)p - sizeof(BlockHeader);
	BlockHeader* blk = (BlockHeader*)ptr;

	if (blk->magic != BLOCK_MAGIC)
	{
		// debug("no magic 0x%lx\n", BLOCK_MAGIC);
		return NULL;
	}
	return blk;
}

static size_t roundup(size_t n, size_t alignment)
{
	return (n+(alignment-1))/alignment * alignment;
}
/*
static void mmapctl(void *ptr)
{
	VirtIOArg arg;
	BlockHeader* blk = NULL;
	func();
	blk = get_block_by_ptr(ptr);
	if(!blk) {
		return;
	}
	void *origin_addr 	= blk->address;
	size_t total_size 	= blk->total_size;
	size_t data_size 	= blk->data_size;
	memset(&arg, 0, ARG_LEN);
	arg.cmd 	= VIRTIO_CUDA_MMAPCTL;
	arg.src 	= (uint64_t)origin_addr;
	arg.srcSize = total_size;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MMAPCTL, &arg);
	
	blk->address 		= origin_addr;
	blk->total_size    	= total_size;
	blk->data_size     	= data_size;
	blk->magic         	= BLOCK_MAGIC;
}
*/

static void *__mmalloc(size_t size)
{
	VirtIOArg arg;
	void *src = NULL;
	int alignment = 8;
	// size_t page_size = sysconf(_SC_PAGESIZE);
	// debug("page size = %d\n", page_size);
	size_t data_start_offset = roundup(sizeof(BlockHeader), alignment);
	size_t header_start_offset = data_start_offset - sizeof(BlockHeader);
	size_t total_size = data_start_offset + size;
	size_t blocks_size = roundup(total_size, KMALLOC_SIZE);

	void *ptr = mmap(0, blocks_size, PROT_READ|PROT_WRITE, 
						MAP_SHARED, fd, map_offset);
	if(ptr == MAP_FAILED) {
		error("mmap failed, error: %s.\n", strerror(errno));
		return NULL;
	}
	map_offset += blocks_size;
	BlockHeader* blk 	= (BlockHeader*)((char*)ptr + header_start_offset);

	msync(ptr, blocks_size, MS_ASYNC);
	src = (char*)ptr + data_start_offset;
	debug("get ptr =%p, size=%lx\n", ptr, blocks_size);
	debug("return src =%p\n", src);
	// mmapctl(src);

	memset(&arg, 0, ARG_LEN);
	arg.cmd 	= VIRTIO_CUDA_MMAPCTL;
	arg.src 	= (uint64_t)ptr;
	arg.srcSize = blocks_size;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MMAPCTL, &arg);
	
	blk->address 		= ptr;
	blk->total_size    	= blocks_size;
	blk->data_size     	= size;
	blk->magic         	= BLOCK_MAGIC;
	return src;
}

static void munmapctl(void *ptr)
{
	VirtIOArg arg;
	BlockHeader* blk = NULL;
	// func();
	blk = get_block_by_ptr(ptr);
	if(!blk) {
		return;
	}
	memset(&arg, 0, ARG_LEN);
	arg.cmd 	= VIRTIO_CUDA_MUNMAPCTL;
	arg.src 	= (uint64_t)blk->address;
	arg.srcSize = blk->total_size;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MUNMAPCTL, &arg);
}


void *malloc(size_t size)
{
	if (global==1) {
		debug("malloc 0x%lx\n", size);
		if (size > (KMALLOC_SIZE)<<2)
			return __mmalloc(size);
	}
	return __libc_malloc(size);
}

void free(void *ptr)
{
	if(global == 1) {
		BlockHeader* blk = NULL;
		// func();
		if (ptr == NULL)
			return;
		blk = get_block_by_ptr(ptr);
		if(!blk) {
			__libc_free(ptr);
			return;
		}
		debug("blk->address   =0x%lx\n",(uint64_t)blk->address);
		debug("blk->total_size=0x%lx\n",(uint64_t)blk->total_size);
		debug("magic 0x%lx\n", blk->magic);
	    munmapctl(ptr);
	    munmap((void*)blk->address, blk->total_size);
	    return;
	}
	return __libc_free(ptr);
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
		error("read error! error %s\n", strerror(errno));
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
	global = 1;
	map_offset=0;
	if(open_vdevice() < 0)
		exit(-1);
	pthread_spin_init(&lock, 0);
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
	
	fatCubinHandle = (unsigned long long**)__libc_malloc(sizeof(unsigned long long*));
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
	// VirtIOArg arg;
	// func();
	// memset(&arg, 0, ARG_LEN);
	// arg.cmd = VIRTIO_CUDA_UNREGISTERFATBINARY;
	// arg.src = (uint64_t)(*fatCubinHandle);
	// arg.tid = syscall(SYS_gettid);
	// send_to_device(VIRTIO_IOC_UNREGISTERFATBINARY, &arg);
	if (fatCubinHandle != NULL)
		__libc_free(fatCubinHandle);
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
	debug("	hostVar = %p, value =%s\n", hostVar, hostVar);
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
	// uint64_t fid;
	func();
	// fid=(uint64_t)func;
	// debug("func id = 0x%lx\n", fid);
	debug("szieof(args)   =0x%lx\n", sizeof(args));
	debug("szieof(args[0])=0x%lx\n", sizeof(args[0]));

	return cudaSuccess;
}

cudaError_t cudaLaunch(const void *entry)
{
	VirtIOArg arg;
	unsigned char *para;
	func();
	if(!entry)
		return cudaSuccess;
	if (kernelConf.gridDim.x<=0 || kernelConf.gridDim.y<=0 || 
		kernelConf.gridDim.z<=0 ||
		kernelConf.blockDim.x<=0 || kernelConf.blockDim.y<=0 || 
		kernelConf.blockDim.z<=0 )
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_LAUNCH;
	arg.src = (uint64_t)&cudaKernelPara;
	para = (unsigned char*)arg.src;
	int para_idx = sizeof(uint32_t);
	int para_num = *(uint32_t*)para;
	debug("para_num=%d\n", para_num);
	for(int i=0; i<para_num; i++) {
		debug("i=%d\n", i);
		debug("size = %u\n", *(uint32_t*)&para[para_idx]);
		if (*(uint32_t*)&para[para_idx]==8)
			debug("value=%llx\n",*(unsigned long long*)&para[para_idx+sizeof(uint32_t)]);
		else
			debug("value=%x\n",*(unsigned int*)&para[para_idx+sizeof(uint32_t)]);
		para_idx += *(uint32_t*)(&para[para_idx]) + sizeof(uint32_t);
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
	if (kind <0 || kind >4) {
		return cudaErrorInvalidMemcpyDirection;
	}
	if(!dst || !src || count<=0)
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_MEMCPY;
	arg.flag 	= kind;
	arg.src 	= (uint64_t)src;
	arg.srcSize = count;
	arg.dst 	= (uint64_t)dst;
	arg.dstSize = count;
	arg.tid 	= syscall(SYS_gettid);
	debug("gettid %d\n", arg.tid);
	send_to_device(VIRTIO_IOC_MEMCPY, &arg);
	if (arg.flag == cudaMemcpyHostToHost) {
		memcpy(dst, src, count);
		return cudaSuccess;
	}
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaMemcpyToSymbol(	const void *symbol, const void *src, 
								size_t count, size_t offset, enum cudaMemcpyKind kind)
{
	VirtIOArg arg;
	func();
	if(!symbol || !src || count<=0)
		return cudaSuccess;
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
	if(!dst || !symbol || count<=0)
		return cudaSuccess;
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
	if(!dst || count<=0)
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_MEMSET;
	arg.dst 	= (uint64_t)dst;
	arg.param 	= count;
	arg.param2  = (uint64_t)value;
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
	if (kind <0 || kind >4) {
		debug("direction is %d\n", kind);
		return cudaErrorInvalidMemcpyDirection;
	}
	if(!dst || !src || count<=0)
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_MEMCPY_ASYNC;
	arg.flag 	= kind;
	arg.src 	= (uint64_t)src;
	arg.srcSize = count;
	arg.dst 	= (uint64_t)dst;
	arg.src2 	= (uint64_t)stream;
	arg.param 	= (uint64_t)stream;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_MEMCPY_ASYNC, &arg);
	if (arg.flag == cudaMemcpyHostToHost) {
		memcpy(dst, src, count);
		return cudaSuccess;
	}
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
	debug("tid %d\n", arg.tid);
	send_to_device(VIRTIO_IOC_MALLOC, &arg);
	*devPtr = (void *)arg.dst;
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
	VirtIOArg arg;
	func();
	if(size<=0)
		return cudaSuccess;
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
	if(!ptr)
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_HOSTUNREGISTER;
	arg.tid = syscall(SYS_gettid);
	arg.src = (uint64_t)ptr;
	arg.srcSize = 0;
	send_to_device(VIRTIO_IOC_HOSTUNREGISTER, &arg);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	func();
	if(size <=0)
		return cudaSuccess;
	// *pHost = malloc(size);
	if(size < KMALLOC_SIZE)
		*pHost = malloc(size);
	else
		*pHost = __mmalloc(size);
	debug("*pHost = %p\n", *pHost);
	return cudaHostRegister(*pHost, size, flags);
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
	func();
	debug("allocating size 0x%lx\n", size);
	return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

cudaError_t cudaFreeHost(void *ptr)
{
	BlockHeader* blk = NULL;
	func();
	if(!ptr)
		return cudaSuccess;
	cudaError_t err = cudaHostUnregister(ptr);
	blk = get_block_by_ptr(ptr);
    if(!blk) {
		return err;
    }
    debug("blk->address   =0x%lx\n",(uint64_t)blk->address);
	debug("blk->total_size=0x%lx\n",(uint64_t)blk->total_size);
	debug("magic 0x%lx\n", blk->magic);
    munmapctl(ptr);
    munmap(blk->address, blk->total_size);
	return err;
}

cudaError_t cudaFree(void *devPtr)
{
	VirtIOArg arg;
	func();
	if(!devPtr)
		return cudaSuccess;
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
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_GETDEVICE, &arg);
	*device 	= (int)arg.flag;
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
	debug("gettid %d\n", arg.tid);
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
	debug("stream = 0x%lx\n", (uint64_t)(*pStream));
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_STREAMCREATEWITHFLAGS;
	arg.flag 	= flags;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_STREAMCREATEWITHFLAGS, &arg);
	 *pStream 	= (cudaStream_t)arg.dst;
	 debug("stream = 0x%lx\n", (uint64_t)(*pStream));
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	VirtIOArg arg;
	func();
	debug("stream = 0x%lx\n", (uint64_t)stream);
	if(stream==0)
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_STREAMDESTROY;
	arg.flag 	= (uint64_t)stream;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_STREAMDESTROY, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_STREAMSYNCHRONIZE;
	arg.flag 	= (uint64_t)stream;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_STREAMSYNCHRONIZE, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
								cudaEvent_t event, unsigned int flags)
{
	VirtIOArg arg;
	func();
	if(event == 0)
		return cudaSuccess;
	assert(flags == 0);
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_STREAMWAITEVENT;
	arg.tid = syscall(SYS_gettid);
	arg.src = (uint64_t)stream;
	arg.dst = (uint64_t)event;
	send_to_device(VIRTIO_IOC_STREAMWAITEVENT, &arg);
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
	*event = (cudaEvent_t)arg.flag;
	debug("tid %d create event is 0x%lx\n", arg.tid, (uint64_t)(*event));
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_EVENTCREATEWITHFLAGS;
	arg.flag 	= (uint64_t)flags;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTCREATEWITHFLAGS, &arg);
	*event 	= (cudaEvent_t)arg.dst;
	debug("event is 0x%lx\n", (uint64_t)(*event));
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	VirtIOArg arg;
	func();
	if (event == 0)
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_EVENTDESTROY;
	arg.flag 	= (uint64_t)event;
	arg.tid 	= syscall(SYS_gettid);
	debug("tid %d destroy event is 0x%lx\n", arg.tid, (uint64_t)(event));
	send_to_device(VIRTIO_IOC_EVENTDESTROY, &arg);
	return (cudaError_t)arg.cmd;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
	VirtIOArg arg;
	func();
	debug("event is 0x%lx\n", (uint64_t)event);
	if(event==0)
		return cudaSuccess;
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
	if(event==0)
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_EVENTSYNCHRONIZE;
	arg.flag 	= (uint64_t)event;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTSYNCHRONIZE, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
	VirtIOArg arg;
	func();
	if(event==0)
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUDA_EVENTQUERY;
	arg.flag 	= (uint64_t)event;
	arg.tid 	= syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_EVENTQUERY, &arg);
	return (cudaError_t)arg.cmd;	
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
	VirtIOArg arg;
	func();
	if(start==0 || end==0)
		return cudaSuccess;
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_EVENTELAPSEDTIME;
	arg.tid = syscall(SYS_gettid);
	arg.src = (uint64_t)start;
	arg.dst = (uint64_t)end;
	arg.param 		= (uint64_t)ms;
	arg.paramSize 	= sizeof(float);
	send_to_device(VIRTIO_IOC_EVENTELAPSEDTIME, &arg);
	debug("elapsed time is %g\n", *ms);
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

cudaError_t cudaPeekAtLastError(void)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUDA_PEEKATLASTERROR;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_PEEKATLASTERROR, &arg);
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

CUBLASAPI cublasStatus_t cublasCreate_v2 (cublasHandle_t *handle)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_CREATE;
	arg.srcSize = sizeof(cublasHandle_t);
	arg.src 	= (uint64_t)handle;
	// debug("sizeof(cublasHandle_t)=%lx\n", sizeof(cublasHandle_t));
	send_to_device(VIRTIO_IOC_CUBLAS_CREATE, &arg);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasDestroy_v2 (cublasHandle_t handle)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_DESTROY;
	arg.srcSize = sizeof(cublasHandle_t);
	arg.src 	= (uint64_t)&handle;
	send_to_device(VIRTIO_IOC_CUBLAS_DESTROY, &arg);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasSetVector (int n, int elemSize, const void *x, 
                                             int incx, void *devicePtr, int incy)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	int int_size = sizeof(int);
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_SETVECTOR;
	arg.srcSize = n * elemSize;
	arg.src 	= (uint64_t)x;
	arg.dst 	= (uint64_t)devicePtr;
	len = int_size * 4 ;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &n, int_size);
	idx += int_size;
	memcpy(buf+idx, &elemSize, int_size);
	idx += int_size;
	memcpy(buf+idx, &incx, int_size);
	idx += int_size;
	memcpy(buf+idx, &incy, int_size);
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= len;
	send_to_device(VIRTIO_IOC_CUBLAS_SETVECTOR, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasGetVector (int n, int elemSize, const void *x, 
                                             int incx, void *y, int incy)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	int int_size = sizeof(int);
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_GETVECTOR;
	arg.srcSize = n * elemSize;
	arg.src 	= (uint64_t)x;
	arg.dst 	= (uint64_t)y;
	len = int_size * 4 ;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &n, int_size);
	idx += int_size;
	memcpy(buf+idx, &elemSize, int_size);
	idx += int_size;
	memcpy(buf+idx, &incx, int_size);
	idx += int_size;
	memcpy(buf+idx, &incy, int_size);
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= len;
	send_to_device(VIRTIO_IOC_CUBLAS_GETVECTOR, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasSetStream_v2 (cublasHandle_t handle, cudaStream_t streamId)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUBLAS_SETSTREAM;
	debug("stream = 0x%lx\n", (uint64_t)streamId);
	arg.src = (uint64_t)handle;
	arg.dst = (uint64_t)streamId;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_CUBLAS_SETSTREAM, &arg);
	return (cublasStatus_t)arg.cmd;	
}


CUBLASAPI cublasStatus_t cublasGetStream_v2 (cublasHandle_t handle, cudaStream_t *streamId)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd = VIRTIO_CUBLAS_GETSTREAM;
	debug("stream = 0x%lx\n", (uint64_t)streamId);
	arg.src = (uint64_t)handle;
	arg.tid = syscall(SYS_gettid);
	send_to_device(VIRTIO_IOC_CUBLAS_GETSTREAM, &arg);
	*streamId = (cudaStream_t)arg.flag;
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasSasum_v2(cublasHandle_t handle, 
                                         int n, 
                                         const float *x, 
                                         int incx, 
                                         float *result) /* host or device pointer */
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_SASUM;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.srcSize2 = (uint32_t)incx;
	arg.dst 	= (uint64_t)handle;
	arg.param 		= (uint64_t)result;
	arg.paramSize 	= sizeof(float);
	send_to_device(VIRTIO_IOC_CUBLAS_SASUM, &arg);
	debug("result = %g\n", *result);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasDasum_v2(cublasHandle_t handle, 
                                     int n, 
                                     const double *x, 
                                     int incx, 
                                     double *result) /* host or device pointer */
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_DASUM;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.srcSize2 	= (uint32_t)incx;
	arg.dst 	= (uint64_t)handle;
	arg.param 		= (uint64_t)result;
	arg.paramSize 	= sizeof(double);
	send_to_device(VIRTIO_IOC_CUBLAS_DASUM, &arg);
	debug("result = %g\n", *result);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasScopy_v2 (cublasHandle_t handle,
                                          int n, 
                                          const float *x, 
                                          int incx, 
                                          float *y, 
                                          int incy)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_SCOPY;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.dst 	= (uint64_t)y;
	arg.src2 	= (uint64_t)handle;
	arg.srcSize2 	= (uint32_t)incx;
	arg.dstSize 	= (uint32_t)incy;
	send_to_device(VIRTIO_IOC_CUBLAS_SCOPY, &arg);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasDcopy_v2 (cublasHandle_t handle,
                                          int n, 
                                          const double *x, 
                                          int incx, 
                                          double *y, 
                                          int incy)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_DCOPY;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.dst 	= (uint64_t)y;
	arg.src2 	= (uint64_t)handle;
	arg.srcSize2 	= (uint32_t)incx;
	arg.dstSize 	= (uint32_t)incy;
	send_to_device(VIRTIO_IOC_CUBLAS_DCOPY, &arg);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasSdot_v2 (cublasHandle_t handle,
                                         int n, 
                                         const float *x, 
                                         int incx, 
                                         const float *y, 
                                         int incy,
                                         float *result)  /* host or device pointer */
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_SDOT;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.dst 	= (uint64_t)y;
	arg.src2 	= (uint64_t)handle;
	arg.srcSize2 	= (uint32_t)incx;
	arg.dstSize 	= (uint32_t)incy;
	arg.param 		= (uint64_t)result;
	arg.paramSize 	= sizeof(float);
	send_to_device(VIRTIO_IOC_CUBLAS_SDOT, &arg);
	debug("result = %g\n", *result);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasDdot_v2 (cublasHandle_t handle,
                                         int n, 
                                         const double *x, 
                                         int incx, 
                                         const double *y,
                                         int incy,
                                         double *result)  /* host or device pointer */
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_DDOT;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.src2 	= (uint64_t)handle;
	arg.dst 	= (uint64_t)y;
	arg.srcSize2 	= (uint32_t)incx;
	arg.dstSize 	= (uint32_t)incy;
	arg.param 		= (uint64_t)result;
	arg.paramSize 	= sizeof(double);
	send_to_device(VIRTIO_IOC_CUBLAS_DDOT, &arg);
	debug("result = %g\n", *result);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasSaxpy_v2 (cublasHandle_t handle,
                                          int n, 
                                          const float *alpha, /* host or device pointer */
                                          const float *x, 
                                          int incx, 
                                          float *y, 
                                          int incy)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_SAXPY;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.dst 	= (uint64_t)y;
	arg.src2 	= (uint64_t)handle;
	arg.srcSize2 	= (uint32_t)incx;
	arg.dstSize 	= (uint32_t)incy;
	len = sizeof(float);
	buf = __libc_malloc(len);
	memcpy(buf+idx, alpha, sizeof(float));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= (uint32_t)len;
	send_to_device(VIRTIO_IOC_CUBLAS_SAXPY, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasDaxpy_v2 (cublasHandle_t handle,
                                          int n, 
                                          const double *alpha, /* host or device pointer */
                                          const double *x, 
                                          int incx, 
                                          double *y, 
                                          int incy)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_DAXPY;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.dst 	= (uint64_t)y;
	arg.src2 	= (uint64_t)handle;
	arg.srcSize2 	= (uint32_t)incx;
	arg.dstSize = (uint32_t)incy;
	len = sizeof(double);
	buf = __libc_malloc(len);
	memcpy(buf+idx, alpha, sizeof(double));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= (uint32_t)len;
	send_to_device(VIRTIO_IOC_CUBLAS_DAXPY, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasSscal_v2(cublasHandle_t handle, 
                                         int n, 
                                         const float *alpha,  /* host or device pointer */
                                         float *x, 
                                         int incx)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;

	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_SSCAL;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.src2 	= (uint64_t)handle;
	arg.srcSize2 = (uint32_t)incx;
	len = sizeof(float);
	buf = __libc_malloc(len);
	memcpy(buf+idx, alpha, sizeof(float));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= (uint32_t)len;
	send_to_device(VIRTIO_IOC_CUBLAS_SSCAL, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}
    
CUBLASAPI cublasStatus_t cublasDscal_v2(cublasHandle_t handle, 
                                         int n, 
                                         const double *alpha,  /* host or device pointer */
                                         double *x, 
                                         int incx)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;

	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_DSCAL;
	arg.src 	= (uint64_t)x;
	arg.srcSize = (uint32_t)n;
	arg.src2 	= (uint64_t)handle;
	arg.srcSize2 = (uint32_t)incx;
	len = sizeof(double);
	buf = __libc_malloc(len);
	memcpy(buf+idx, alpha, sizeof(double));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= (uint32_t)len;
	send_to_device(VIRTIO_IOC_CUBLAS_DSCAL, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}

/* GEMV */
CUBLASAPI cublasStatus_t cublasSgemv_v2 (cublasHandle_t handle, 
                                          cublasOperation_t trans, 
                                          int m, 
                                          int n, 
                                          const float *alpha, /* host or device pointer */
                                          const float *A, 
                                          int lda, 
                                          const float *x, 
                                          int incx, 
                                          const float *beta,  /* host or device pointer */
                                          float *y, 
                                          int incy)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	int int_size = sizeof(int);

	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_SGEMV;
	arg.src 	= (uint64_t)A;
	arg.srcSize = m * n;
	arg.src2 	= (uint64_t)x;
	arg.srcSize2 = n;
	arg.dst 	= (uint64_t)y;
	arg.dstSize = m;

	len = int_size * 3 + sizeof(cublasHandle_t) + 
			sizeof(cublasOperation_t) + sizeof(float)*2;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &handle, sizeof(cublasHandle_t));
	idx += sizeof(cublasHandle_t);
	memcpy(buf+idx, &trans, sizeof(cublasOperation_t));
	idx += sizeof(cublasOperation_t);
	memcpy(buf+idx, &lda, int_size);
	idx += int_size;
	memcpy(buf+idx, &incx, int_size);
	idx += int_size;
	memcpy(buf+idx, &incy, int_size);
	idx += int_size;
	memcpy(buf+idx, alpha, sizeof(float));
	idx += sizeof(float);
	memcpy(buf+idx, beta, sizeof(float));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= (uint32_t)len;
	send_to_device(VIRTIO_IOC_CUBLAS_SGEMV, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}
 
CUBLASAPI cublasStatus_t cublasDgemv_v2 (cublasHandle_t handle, 
                                          cublasOperation_t trans, 
                                          int m,
                                          int n,
                                          const double *alpha, /* host or device pointer */ 
                                          const double *A,
                                          int lda,
                                          const double *x,
                                          int incx,
                                          const double *beta, /* host or device pointer */
                                          double *y, 
                                          int incy)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	int int_size = sizeof(int);

	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_DGEMV;
	arg.src 	= (uint64_t)A;
	arg.srcSize = (uint32_t)(m * n);
	arg.src2 	= (uint64_t)x;
	arg.srcSize2 = (uint32_t)n;
	arg.dst 	= (uint64_t)y;
	arg.dstSize = (uint32_t)m;
	len = int_size * 3 + sizeof(cublasHandle_t) + 
			sizeof(cublasOperation_t)+ sizeof(double)*2;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &handle, sizeof(cublasHandle_t));
	idx += sizeof(cublasHandle_t);
	memcpy(buf+idx, &trans, sizeof(cublasOperation_t));
	idx += sizeof(cublasOperation_t);
	memcpy(buf+idx, &lda, int_size);
	idx += int_size;
	memcpy(buf+idx, &incx, int_size);
	idx += int_size;
	memcpy(buf+idx, &incy, int_size);
	idx += int_size;
	memcpy(buf+idx, alpha, sizeof(double));
	idx += sizeof(double);
	memcpy(buf+idx, beta, sizeof(double));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= len;
	send_to_device(VIRTIO_IOC_CUBLAS_DGEMV, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}

/* GEMM */
CUBLASAPI cublasStatus_t cublasSgemm_v2 (cublasHandle_t handle, 
	                                      cublasOperation_t transa,
	                                      cublasOperation_t transb, 
	                                      int m,
	                                      int n,
	                                      int k,
	                                      const float *alpha, /* host or device pointer */  
	                                      const float *A, 
	                                      int lda,
	                                      const float *B,
	                                      int ldb, 
	                                      const float *beta, /* host or device pointer */  
	                                      float *C,
	                                      int ldc)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	int int_size = sizeof(int);

	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_SGEMM;
	arg.src 	= (uint64_t)A;
	arg.srcSize = m * k;
	arg.src2 	= (uint64_t)B;
	arg.srcSize2 = k * n;
	arg.dst 	= (uint64_t)C;
	arg.dstSize = m * n;
	len = int_size * 6 + sizeof(cublasHandle_t) + 
			sizeof(cublasOperation_t)*2 + sizeof(float)*2;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &handle, sizeof(cublasHandle_t));
	idx += sizeof(cublasHandle_t);
	memcpy(buf+idx, &transa, sizeof(cublasOperation_t));
	idx += sizeof(cublasOperation_t);
	memcpy(buf+idx, &transb, sizeof(cublasOperation_t));
	idx += sizeof(cublasOperation_t);
	memcpy(buf+idx, &m, int_size);
	idx += int_size;
	memcpy(buf+idx, &n, int_size);
	idx += int_size;
	memcpy(buf+idx, &k, int_size);
	idx += int_size;
	memcpy(buf+idx, &lda, int_size);
	idx += int_size;
	memcpy(buf+idx, &ldb, int_size);
	idx += int_size;
	memcpy(buf+idx, &ldc, int_size);
	idx += int_size;
	memcpy(buf+idx, alpha, sizeof(float));
	idx += sizeof(float);
	memcpy(buf+idx, beta, sizeof(float));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= len;
	send_to_device(VIRTIO_IOC_CUBLAS_SGEMM, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}

CUBLASAPI cublasStatus_t cublasDgemm_v2 (cublasHandle_t handle, 
                                          cublasOperation_t transa,
                                          cublasOperation_t transb, 
                                          int m,
                                          int n,
                                          int k,
                                          const double *alpha, /* host or device pointer */  
                                          const double *A, 
                                          int lda,
                                          const double *B,
                                          int ldb, 
                                          const double *beta, /* host or device pointer */  
                                          double *C,
                                          int ldc)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	int int_size = sizeof(int);

	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_DGEMM;
	arg.src 	= (uint64_t)A;
	arg.srcSize = m * k;
	arg.src2 	= (uint64_t)B;
	arg.srcSize2 = k * n;
	arg.dst 	= (uint64_t)C;
	arg.dstSize = m * n;
	len = int_size * 6 + sizeof(cublasHandle_t) + 
			sizeof(cublasOperation_t)*2 + sizeof(double)*2;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &handle, sizeof(cublasHandle_t));
	idx += sizeof(cublasHandle_t);
	memcpy(buf+idx, &transa, sizeof(cublasOperation_t));
	idx += sizeof(cublasOperation_t);
	memcpy(buf+idx, &transb, sizeof(cublasOperation_t));
	idx += sizeof(cublasOperation_t);
	memcpy(buf+idx, &m, int_size);
	idx += int_size;
	memcpy(buf+idx, &n, int_size);
	idx += int_size;
	memcpy(buf+idx, &k, int_size);
	idx += int_size;
	memcpy(buf+idx, &lda, int_size);
	idx += int_size;
	memcpy(buf+idx, &ldb, int_size);
	idx += int_size;
	memcpy(buf+idx, &ldc, int_size);
	idx += int_size;
	memcpy(buf+idx, alpha, sizeof(double));
	idx += sizeof(double);
	memcpy(buf+idx, beta, sizeof(double));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= len;
	send_to_device(VIRTIO_IOC_CUBLAS_DGEMM, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}

cublasStatus_t cublasSetMatrix (int rows, int cols, int elemSize, 
                                 const void *A, int lda, void *B, 
                                 int ldb)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	int int_size = sizeof(int);
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_SETMATRIX;
	arg.src 	= (uint64_t)A;
	arg.srcSize = (uint32_t)(rows * cols * elemSize);
	arg.dst 	= (uint64_t)B;
	len = int_size * 5;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &rows, int_size);
	idx += int_size;
	memcpy(buf+idx, &cols, int_size);
	idx += int_size;
	memcpy(buf+idx, &elemSize, int_size);
	idx += int_size;
	memcpy(buf+idx, &lda, int_size);
	idx += int_size;
	memcpy(buf+idx, &ldb, int_size);
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= len;
	send_to_device(VIRTIO_IOC_CUBLAS_SETMATRIX, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}

cublasStatus_t cublasGetMatrix (int rows, int cols, int elemSize, 
                                 const void *A, int lda, void *B,
                                 int ldb)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;
	int int_size = sizeof(int);
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CUBLAS_GETMATRIX;
	arg.src 	= (uint64_t)A;
	arg.srcSize = (uint32_t)(rows * cols * elemSize);
	arg.dst 	= (uint64_t)B;
	len = int_size * 5;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &rows, int_size);
	idx += int_size;
	memcpy(buf+idx, &cols, int_size);
	idx += int_size;
	memcpy(buf+idx, &elemSize, int_size);
	idx += int_size;
	memcpy(buf+idx, &lda, int_size);
	idx += int_size;
	memcpy(buf+idx, &ldb, int_size);
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= len;
	send_to_device(VIRTIO_IOC_CUBLAS_GETMATRIX, &arg);
	__libc_free(buf);
	return (cublasStatus_t)arg.cmd;
}
/*****************************************************************************/
/******CURAND***********/
/*****************************************************************************/

curandStatus_t CURANDAPI 
curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	debug("sizeof(curandGenerator_t) = %lx\n", sizeof(curandGenerator_t));
	arg.cmd 	= VIRTIO_CURAND_CREATEGENERATOR;
	arg.dst 	= (uint64_t)rng_type;
	send_to_device(VIRTIO_IOC_CURAND_CREATEGENERATOR, &arg);
	*generator 	= (curandGenerator_t)arg.flag;
	return (curandStatus_t)arg.cmd;
}

curandStatus_t CURANDAPI 
curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CURAND_CREATEGENERATORHOST;
	arg.dst 	= (uint64_t)rng_type;
	send_to_device(VIRTIO_IOC_CURAND_CREATEGENERATORHOST, &arg);
	*generator 	= (curandGenerator_t)arg.flag;
	return (curandStatus_t)arg.cmd;
}

curandStatus_t CURANDAPI 
curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CURAND_GENERATE;
	arg.src 	= (uint64_t)generator;
	arg.dst 	= (uint64_t)outputPtr;
	arg.dstSize	= (uint32_t)(num*sizeof(unsigned int));
	arg.param	= (uint64_t)num;
	send_to_device(VIRTIO_IOC_CURAND_GENERATE, &arg);
	return (curandStatus_t)arg.cmd;
}

curandStatus_t CURANDAPI 
curandGenerateNormal(curandGenerator_t generator, float *outputPtr, 
                     size_t n, float mean, float stddev)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;

	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CURAND_GENERATENORMAL;
	arg.src 	= (uint64_t)generator;
	arg.dst 	= (uint64_t)outputPtr;
	arg.dstSize	= (uint32_t)(n*sizeof(float));
	arg.src2	= (uint64_t)n;
	len = sizeof(float)*2;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &mean, sizeof(float));
	idx += sizeof(float);
	memcpy(buf+idx, &stddev, sizeof(float));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= len;
	send_to_device(VIRTIO_IOC_CURAND_GENERATENORMAL, &arg);
	__libc_free(buf);
	return (curandStatus_t)arg.cmd;
}

curandStatus_t CURANDAPI 
curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr, 
                     size_t n, double mean, double stddev)
{
	VirtIOArg arg;
	uint8_t *buf = NULL;
	int len = 0;
	int idx = 0;

	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CURAND_GENERATENORMALDOUBLE;
	arg.src 	= (uint64_t)generator;
	arg.dst 	= (uint64_t)outputPtr;
	arg.dstSize	= (uint32_t)(n*sizeof(double));
	arg.src2	= (uint64_t)n;
	len = sizeof(double)*2;
	buf = __libc_malloc(len);
	memcpy(buf+idx, &mean, sizeof(double));
	idx += sizeof(double);
	memcpy(buf+idx, &stddev, sizeof(double));
	arg.param 		= (uint64_t)buf;
	arg.paramSize 	= len;
	send_to_device(VIRTIO_IOC_CURAND_GENERATENORMALDOUBLE, &arg);
	__libc_free(buf);
	return (curandStatus_t)arg.cmd;
}

curandStatus_t CURANDAPI 
curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CURAND_GENERATEUNIFORM;
	arg.src 	= (uint64_t)generator;
	arg.dst 	= (uint64_t)outputPtr;
	arg.dstSize	= (uint32_t)(num*sizeof(float));
	arg.param	= (uint64_t)num;
	send_to_device(VIRTIO_IOC_CURAND_GENERATEUNIFORM, &arg);
	return (curandStatus_t)arg.cmd;
}

curandStatus_t CURANDAPI 
curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CURAND_GENERATEUNIFORMDOUBLE;
	arg.src 	= (uint64_t)generator;
	arg.dst 	= (uint64_t)outputPtr;
	arg.dstSize	= (uint32_t)(num*sizeof(double));
	arg.param	= (uint64_t)num;
	send_to_device(VIRTIO_IOC_CURAND_GENERATEUNIFORMDOUBLE, &arg);
	return (curandStatus_t)arg.cmd;
}

curandStatus_t CURANDAPI 
curandDestroyGenerator(curandGenerator_t generator)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CURAND_DESTROYGENERATOR;
	arg.src 	= (uint64_t)generator;
	send_to_device(VIRTIO_IOC_CURAND_DESTROYGENERATOR, &arg);
	return (curandStatus_t)arg.cmd;
}

curandStatus_t CURANDAPI 
curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CURAND_SETGENERATOROFFSET;
	arg.src 	= (uint64_t)generator;
	arg.param 	= (uint64_t)offset;
	send_to_device(VIRTIO_IOC_CURAND_SETGENERATOROFFSET, &arg);
	return (curandStatus_t)arg.cmd;
}

curandStatus_t CURANDAPI 
curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed)
{
	VirtIOArg arg;
	func();
	memset(&arg, 0, sizeof(VirtIOArg));
	arg.cmd 	= VIRTIO_CURAND_SETPSEUDORANDOMSEED;
	arg.src 	= (uint64_t)generator;
	arg.param 	= (uint64_t)seed;
	send_to_device(VIRTIO_IOC_CURAND_SETPSEUDORANDOMSEED, &arg);
	return (curandStatus_t)arg.cmd;
}