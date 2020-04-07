#ifndef _GNU_SOURCE
#define _GNU_SOURCE // RTLD_NEXT
#endif
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
#include <fcntl.h>  //open
#include <unistd.h> // close, syscall
#include <sys/syscall.h> // SYS_gettid
#include "../../virtio-ioc.h"
#include <errno.h>  // errno
#include <sys/mman.h>   // mmap, PROT_READ, PROT_WRITE, MAP_SHARED
#include <assert.h>     // assert
#include <pthread.h>
#include <stdbool.h>
#include <malloc.h>
#include <sys/time.h>
#include <limits.h>
// Needed for definition of remote attestation messages.
#include "remote_attestation_result.h"
#include "isv_enclave_u.h"
// Needed to call untrusted key exchange library APIs, i.e. sgx_ra_proc_msg2.
#include "sgx_ukey_exchange.h"
// Needed to create enclave and do ecall.
#include "sgx_urts.h"
// Needed to query extended epid group id.
#include "sgx_uae_service.h"
#include <sgx_uswitchless.h>
#include "../utils/cudump.h"

#ifndef SAFE_FREE
#define SAFE_FREE(ptr) {if (NULL != (ptr)) {free(ptr); (ptr) = NULL;}}
#endif

static char enclave_path[256];
#define ENCLAVE_PATH    "/usr/local/cuda/lib64/isv_enclave.signed.so"
#define DEVICE_FILE     "/dev/cudaport2p%d"

#ifdef VIRTIO_CUDA_DEBUG
    #define debug(fmt, arg...) do{printf("[DEBUG]: " fmt, ##arg);} while(0)
    #define func() do { printf("[FUNC] PID %ld Now in %s\n", syscall(SYS_gettid), __FUNCTION__);} while(0)
    #define debug_clean(fmt, arg...) do{printf("" fmt, ##arg);}while(0)
    #define api_inc(n) do {cuda_api_count[n]++; } while(0)
    #define mem_inc(n, size) do {accum_memcpy_size[n]+=size; } while(0)
#else
    #define debug(fmt, arg...) 
    #define func() 
    #define debug_clean(fmt, arg...)  
    #define api_inc(n) 
    #define mem_inc(n, size) 
#endif

#define error(fmt, arg...) printf("[ERROR]: %s->line : %d. " fmt, __FUNCTION__, __LINE__, ##arg)

#define MODE O_RDWR

#define ARG_LEN sizeof(VirtIOArg)
#define DEVICE_COUNT 32

/*
* Global Variable
*/

static struct CUctx_st *ctx;
// store single kernel configuration and params
static struct CUkernel_st *kernel_param;
static int global_init = 0;
// accumulate api number in different types
static int cuda_api_count[API_TYPE_SIZE] = {0};
// accumulate memory size in 5 types
/*
0   host 2 host
1   host 2 device
2   device 2 host
3   device 2 device
4   default
*/
static long accum_memcpy_size[5]={0};



typedef struct KernelConf
{
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    void * stream;
} KernelConf_t ;

static uint8_t cudaKernelPara[512]; // uint8_t === unsigned char
static uint32_t cudaParaSize;       // uint32_t == unsigned int
static KernelConf_t kernelConf;
static int device_count=0;  // virtual device number
static int minor = 0;       // get current device

extern "C" void *__libc_malloc(size_t);
extern "C" void __libc_free(void *ptr);
static size_t map_offset=0;
static size_t const BLOCK_MAGIC = 0xdeadbeaf;

typedef struct block_header
{
    void* address;
    size_t total_size;
    size_t data_size;
    size_t magic;
} BlockHeader;

typedef struct sgx_ra_env
{
    sgx_ra_context_t context;
    sgx_enclave_id_t enclave_id;
} SGX_RA_ENV;


typedef struct fatbin_buf
{
    uint32_t block_size;
    uint32_t total_size;
    uint32_t size;
    uint32_t nr_binary;
    uint8_t  buf[];
} fatbin_buf_t;

static fatbin_buf_t *p_binary;

typedef struct cubin_buf
{
    uint32_t size;
    uint32_t nr_var;
    uint32_t nr_func;
#ifdef ENABLE_MAC
    uint8_t payload_tag[SAMPLE_SP_TAG_SIZE];
#endif
    uint8_t  buf[];
} cubin_buf_t;
static cubin_buf_t *p_last_binary;

typedef struct function_buf
{
    uint64_t hostFun;
    uint32_t size;
    uint8_t  buf[];
} function_buf_t;

typedef struct var_buf
{
    uint64_t hostVar;
    int constant;
    int global;
    uint32_t size;
    uint8_t  buf[];
} var_buf_t;

__attribute__ ((constructor)) void my_library_init(void);
__attribute__ ((destructor)) void my_library_fini(void);
static int init_sgx_ecdh(SGX_RA_ENV *sgx_ctx);
static int fini_sgx_ecdh(SGX_RA_ENV sgx_ctx);

static pthread_spinlock_t lock;
static SGX_RA_ENV sgx_env;
/*
 * ioctl
*/
void send_to_device(int cmd, void *arg)
{
    #ifdef VIRTIO_LOCK_USER
    pthread_spin_lock(&lock);
    #endif
    if(ioctl(ctx->fd, cmd, arg) == -1){
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
    void *origin_addr   = blk->address;
    size_t total_size   = blk->total_size;
    size_t data_size    = blk->data_size;
    memset(&arg, 0, ARG_LEN);
    arg.cmd     = VIRTIO_CUDA_MMAPCTL;
    arg.src     = (uint64_t)origin_addr;
    arg.srcSize = total_size;
    arg.tid     = syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_MMAPCTL, &arg);
    
    blk->address        = origin_addr;
    blk->total_size     = total_size;
    blk->data_size      = data_size;
    blk->magic          = BLOCK_MAGIC;
}
*/

static void *__mmalloc(size_t size)
{
    VirtIOArg arg;
    void *src = NULL;
    int alignment = 8;
    func();
    // size_t page_size = sysconf(_SC_PAGESIZE);
    // debug("page size = %d\n", page_size);
    size_t data_start_offset = roundup(sizeof(BlockHeader), alignment);
    size_t header_start_offset = data_start_offset - sizeof(BlockHeader);
    size_t total_size = data_start_offset + size;
    size_t blocks_size = roundup(total_size, KMALLOC_SIZE);

    void *ptr = mmap(0, blocks_size, PROT_READ|PROT_WRITE, 
                        MAP_SHARED, ctx->fd, map_offset);
    if(ptr == MAP_FAILED) {
        error("mmap failed, error: %s.\n", strerror(errno));
        return NULL;
    }
    map_offset += blocks_size;
    BlockHeader* blk    = (BlockHeader*)((char*)ptr + header_start_offset);

    msync(ptr, blocks_size, MS_ASYNC);
    src = (char*)ptr + data_start_offset;
    debug("get ptr =%p, size=%lx\n", ptr, blocks_size);
    debug("return src =%p\n", src);
    // mmapctl(src);

    memset(&arg, 0, ARG_LEN);
    arg.cmd     = VIRTIO_CUDA_MMAPCTL;
    arg.src     = (uint64_t)ptr;
    arg.srcSize = (uint32_t)blocks_size;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_MMAPCTL, &arg);
    
    blk->address        = ptr;
    blk->total_size     = blocks_size;
    blk->data_size      = size;
    blk->magic          = BLOCK_MAGIC;
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
    arg.cmd     = VIRTIO_CUDA_MUNMAPCTL;
    arg.src     = (uint64_t)blk->address;
    arg.srcSize = (uint32_t)blk->total_size;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_MUNMAPCTL, &arg);
}

extern "C" void *my_malloc(size_t size)
{
    debug("malloc 0x%lx\n", size);
    if (size >= KMALLOC_SIZE)
        return __mmalloc(size);
    return __libc_malloc(size);
}

extern "C" void my_free(void *ptr)
{
    BlockHeader* blk = NULL;
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
}

int get_vdevice_count(int *result)
{
    char fname[128]="/proc/virtio-cuda/virtual_device_count";
    char buf[15];
    ssize_t size=0;
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
static int open_vdevice()
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
        ctx->fd = open(devname, MODE);
        if(ctx->fd>= 0) {
            sprintf(enclave_path, ENCLAVE_PATH, i);
            break;
        }
        else if(errno==EBUSY) {
            debug("device %d is busy\n", i);
            continue;
        }
        else
            error("open device %d failed, %s (%d)", 
                i, (char *)strerror(errno), errno);
    }
    if(i > device_count) {
        error("Failed to find valid device file.\n");
        return -EINVAL;
    }
    minor = i;
    debug("fd is %d\n", ctx->fd);
    return 0;
}

static void close_vdevice()
{
    func();
    close(ctx->fd);
    debug("closing fd\n");
}

static void ctx_new()
{
    ctx = (CUctx_st *)malloc(sizeof(*ctx));
    ctx->fd = 0;
    ctx->primary_context_initialized = 0;
    ctx->nr_mod = 0;
    ctx->head_mod = NULL;
}

static void ctx_del()
{
    free(ctx);
}

void my_library_init(void) {
    #ifdef  TIMING
        struct timeval total_start, total_end;
        struct timeval ecdh_start, ecdh_end;
    #endif
    #ifdef  TIMING
        gettimeofday(&total_start, NULL);
    #endif
    size_t page_size = sysconf(_SC_PAGESIZE);
    debug("Init dynamic library.\n");
    ctx_new();

    map_offset=0;
    if(open_vdevice() < 0)
        exit(-1);
    pthread_spin_init(&lock, 0);

    #ifdef  TIMING
        gettimeofday(&ecdh_start, NULL);
    #endif
#ifdef ENABLE_SGX
    init_sgx_ecdh(&sgx_env);
#endif
    #ifdef  TIMING
        gettimeofday(&ecdh_end, NULL);
    #endif
    #ifdef  TIMING
        double ecdh_time   = (double)(ecdh_end.tv_usec - ecdh_start.tv_usec)/1000000 +
                        (double)(ecdh_end.tv_sec - ecdh_start.tv_sec);
        printf("ecdh time: \t\t%f\n", ecdh_time);
    #endif

    p_binary = (fatbin_buf_t *)__libc_malloc(page_size<<5);
    if (!p_binary) {
        error("Failed to allocate primary context buffer.\n");
        exit(-1);
    }
    debug("p_binary address %p\n", p_binary);
    p_binary->size = 0;
    p_binary->nr_binary = 0;
    p_binary->total_size = page_size<<5;
    p_binary->block_size = page_size<<3;
    #ifdef  TIMING
        gettimeofday(&total_end, NULL);
    #endif
    #ifdef  TIMING
        double total_time   = (double)(total_end.tv_usec - total_start.tv_usec)/1000000 +
                        (double)(total_end.tv_sec - total_start.tv_sec);
        printf("init library time: \t\t%f\n", total_time);
    #endif
}

static void count_cuda_api(void)
{
    int cnt = 0;
    printf("API count:\n");
    for(int i=0; i<API_TYPE_SIZE; i++) {
        cnt += cuda_api_count[i];
        printf(" %d : %d \n", i, cuda_api_count[i]);
    }
    printf("total #api is %d\n", cnt);
}

static void format_size(long size)
{
    long n=size;
    if(n>>30) {
        printf(" %d GB\n", n>>30);
        return;
    } else if(n>>20) {
        printf(" %d MB\n", n>>20);
        return;
    } else if(n>>10) {
        printf(" %d KB\n", n>>10);
        return;
    } else {
        printf(" %d B\n",  n);
        return;
    }
}

static void count_memcpy_size(void)
{
    printf("accumulate memcpy size:\n");
    for(int i=0; i<5; i++) {
        printf(" %d : 0x%lx(%d) \t", i, accum_memcpy_size[i], accum_memcpy_size[i]);
        format_size(accum_memcpy_size[i]);
    }
}

void my_library_fini(void)
{
    debug("deinit dynamic library\n");
    __libc_free(p_binary);
#ifdef ENABLE_SGX
    fini_sgx_ecdh(sgx_env);
#endif
    close_vdevice();
    ctx_del();
    count_cuda_api();
    count_memcpy_size();
}

static void get_mac(uint8_t *data, uint32_t size, uint8_t *payload_tag)
{
    sgx_status_t status;
    int ret;
    ret = mac_data(sgx_env.enclave_id, &status, sgx_env.context, 
                            data, size, payload_tag);
    if((SGX_SUCCESS != ret)  || (SGX_SUCCESS != status))
    {
        error("\nError, get mac using SK based AESGCM failed. ret = "
                        "0x%0x. status = 0x%0x", ret, status);
        return;
    }
    debug("payload tag\n");
    for (int k=0; k<SAMPLE_SP_TAG_SIZE; k++) {
        debug_clean("%x ", payload_tag[k]);
    }
    debug_clean("\n\n");
}

static void get_decrypted_data(uint8_t *src, uint32_t size, uint8_t *dst, uint8_t *payload_tag)
{
    sgx_status_t status;
    int ret;
    debug("payload tag\n");
    for (int k=0; k<SAMPLE_SP_TAG_SIZE; k++) {
        debug_clean("%x ", payload_tag[k]);
    }
    debug_clean("\n\n");
    ret = decrypt_data(sgx_env.enclave_id, &status, sgx_env.context, 
                            src, size, dst, payload_tag);
    if((SGX_SUCCESS != ret)  || (SGX_SUCCESS != status))
    {
        error("\nError, decrypt data using SK based AESGCM failed. ret = "
                        "0x%0x. status = 0x%0x\n", ret, status);
        return;
    }
}

static void get_encrypted_data(uint8_t *src, uint32_t size, uint8_t *dst, uint8_t *payload_tag)
{
    sgx_status_t status;
    int ret;
    ret = encrypt_data(sgx_env.enclave_id, &status, sgx_env.context, 
                            src, size, dst, payload_tag);
    if((SGX_SUCCESS != ret)  || (SGX_SUCCESS != status))
    {
        error("Error, encrypt data using SK based AESGCM failed. ret = "
                        "0x%0x. status = 0x%0x\n", ret, status);
        return;
    }
    debug("payload tag\n");
    for (int k=0; k<SAMPLE_SP_TAG_SIZE; k++) {
        debug_clean("%x ", payload_tag[k]);
    }
    debug_clean("\n\n");
}

static void init_primary_context()
{
    VirtIOArg arg;
#ifdef ENABLE_MAC
    uint8_t payload_tag[SAMPLE_SP_TAG_SIZE];
#endif
    if(!ctx->primary_context_initialized) {
        ctx->primary_context_initialized = 1;
        memset(&arg, 0, ARG_LEN);
        debug("nr_binary %x\n", p_binary->nr_binary);
#ifdef ENABLE_MAC
        get_mac((uint8_t *)p_binary, p_binary->total_size, payload_tag);
        memcpy(arg.mac, payload_tag, SAMPLE_SP_TAG_SIZE);
#endif
        arg.src     = (uint64_t)p_binary;
        arg.srcSize = p_binary->total_size;
        arg.cmd     = VIRTIO_CUDA_PRIMARYCONTEXT;
        arg.tid     = (uint32_t)syscall(SYS_gettid);
        send_to_device(VIRTIO_IOC_PRIMARYCONTEXT, &arg);
        if(arg.cmd != cudaSuccess)
        {
            error("Failed to initialize primary context\n");
            exit(-1);
        }
    }
}

static void dump_fatbin_to_file(char *fatbin, int size)
{
    int fd = open("/tmp/test.cubin", O_WRONLY| O_TRUNC | O_CREAT, 0666);
    write(fd, (void*)fatbin, size);
    close(fd);
}

static CUmod_st * mod_add(const char *cubin)
{
    CUmod_st *mod = (CUmod_st*)malloc(sizeof(*mod));
    if(cuda_load_cubin(mod, cubin)) {
        error("Failed to load cubin\n");
        return NULL;
    }
    mod->next = ctx->head_mod;
    ctx->head_mod = mod;
    ctx->nr_mod++;
    return mod;
}

static void mod_remove(const char *cubin)
{

}

static void mod_clear()
{
    CUmod_st *mod=NULL;
    while (ctx->head_mod) {
        mod = ctx->head_mod->next;
        ctx->head_mod = ctx->head_mod->next;
        free(mod);
    }
}

static void dump_cuda_symbol(struct CUmod_st *mod)
{
    dump_symbol(mod);
}

static void dump_cuda_kernel(struct CUmod_st *mod)
{
    dump_kernel(mod);
}

extern "C" void** __cudaRegisterFatBinary(void *fatCubin)
{
    VirtIOArg arg;
    unsigned int magic;
    void **fatCubinHandle;
    uint32_t size;
    struct fatBinaryHeader *fatHeader;
#ifdef ENABLE_MAC
    uint8_t payload_tag[SAMPLE_SP_TAG_SIZE];
#endif
 
    func();
    magic = *(unsigned int*)fatCubin;
    if (magic == FATBINC_MAGIC)
    {
        __fatBinC_Wrapper_t *binary = (__fatBinC_Wrapper_t*)fatCubin;
        debug("FatBin\n");
        debug("magic    =   0x%x\n", binary->magic);
        debug("version  =   0x%x\n", binary->version);
        debug("data =   %p\n", binary->data);
        debug("filename_or_fatbins  =   %p\n", binary->filename_or_fatbins);
        fatCubinHandle = (void **)&binary->data;
        fatHeader = (struct fatBinaryHeader*)binary->data;
        debug("FatBinHeader = %p\n", fatHeader);
        debug("magic    =   0x%x\n", fatHeader->magic);
        debug("version  =   0x%x\n", fatHeader->version);
        debug("headerSize = %d(0x%x)\n", fatHeader->headerSize, fatHeader->headerSize);
        debug("fatSize  =   %lld(0x%llx)\n", fatHeader->fatSize, fatHeader->fatSize);
        // dump_fatbin_to_file((char *)binary->data, fatHeader->headerSize + fatHeader->fatSize);
        // Note that fatcubin includes meta header and cubin data
        // meta header occupies first 0x50 bytes.
        CUmod_st *mod = mod_add((char*)binary->data+0x50);
        dump_cuda_symbol(mod);
        dump_cuda_kernel(mod);


        // initialize arguments
        memset(&arg, 0, ARG_LEN);
        size = (uint32_t)(fatHeader->headerSize + fatHeader->fatSize);
        arg.src     = (uint64_t)(binary->data);
        arg.src2    = (uint64_t)(binary->data);
        arg.srcSize = size;
        arg.dstSize = 0;
        arg.cmd     = VIRTIO_CUDA_REGISTERFATBINARY;
        arg.tid     = (uint32_t)syscall(SYS_gettid);
        if(fatHeader->fatSize == 0) {
            debug("Invalid fatbin.\n");
            p_last_binary = NULL;
            return fatCubinHandle;
        }
        // 
        if(p_binary->total_size < sizeof(fatbin_buf_t) + p_binary->size + size + sizeof(cubin_buf_t)) {
            debug("realloc size %x to %lx\n", p_binary->total_size, 
                sizeof(fatbin_buf_t) + p_binary->size + size + sizeof(cubin_buf_t) );
            p_binary->total_size = roundup(sizeof(fatbin_buf_t) + p_binary->size + size + sizeof(cubin_buf_t), 
                                    p_binary->block_size);
            p_binary = (fatbin_buf_t *)realloc(p_binary, p_binary->total_size);
            debug("realloc total_size %x\n", p_binary->total_size);
        }
        debug("p_binary->nr_binary %x\n", p_binary->nr_binary);
        p_last_binary = (cubin_buf_t *)(p_binary->buf + p_binary->size);
        p_last_binary->size     = size;
        p_last_binary->nr_var   = 0;
        p_last_binary->nr_func  = 0;
#ifdef ENABLE_ENC
        debug("binary_data first 4 bytes %x\n", *(int*)binary->data);
        get_encrypted_data((uint8_t *)binary->data, size, p_last_binary->buf, payload_tag);
        memcpy(p_last_binary->payload_tag, payload_tag, SAMPLE_SP_TAG_SIZE);
#else
        memcpy(p_last_binary->buf, binary->data, size);
#endif
        p_binary->size += size + sizeof(cubin_buf_t);
        p_binary->nr_binary++;
        debug("p_binary->size is %x\n", p_binary->size);
        // send_to_device(VIRTIO_IOC_REGISTERFATBINARY, &arg);
        // if(arg.cmd != cudaSuccess)
        // {
        //     error(" fatbin not registered\n");
        //     exit(-1);
        // }
        return fatCubinHandle;
    }
    else
    {
        error("Unrecongnized CUDA FAT MAGIC 0x%x\n", magic);
        exit(-1);
    }
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    func();
    debug("fatcubinhandle %p\n", fatCubinHandle[0]);
    return ;
}

extern "C" void __cudaRegisterFatBinaryEnd(
  void **fatCubinHandle
)
{
    func();
    debug("fatcubinhandle %p\n", fatCubinHandle[0]);
    return ;
}


extern "C" void __cudaRegisterFunction(
    void        **fatCubinHandle,
    const char  *hostFun,
    char        *deviceFun,
    const char  *deviceName,
    int         thread_limit,
    uint3       *tid,
    uint3       *bid,
    dim3        *bDim,
    dim3        *gDim,
    int         *wSize
)
{
    VirtIOArg arg;
    computeFatBinaryFormat_t fatBinHeader;
    uint32_t buf_size = 0;
    unsigned long offset = 0;
    func();
    if(!p_last_binary)
        return;

    fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);
//  debug(" fatbin magic= 0x%x\n", fatBinHeader->magic);
//  debug(" fatbin version= %d\n", fatBinHeader->version);
//  debug(" fatbin headerSize= 0x%x\n", fatBinHeader->headerSize);
//  debug(" fatbin fatSize= 0x%llx\n", fatBinHeader->fatSize);
    debug(" fatCubinHandle = %p, value =%p\n", fatCubinHandle, fatCubinHandle[0]);
    debug(" hostFun = %s, value =%p\n", hostFun, hostFun);
    debug(" deviceFun =%s, %p\n", deviceFun, deviceFun);
    debug(" deviceName = %s\n", deviceName);
    debug(" thread_limit = %d\n", thread_limit);
    debug(" tid = {%d, %d, %d} \n", tid?tid->x:0, tid?tid->y:0, tid?tid->z:0);
    debug(" bid = {%d, %d, %d} \n", bid?bid->x:0, bid?bid->y:0, bid?bid->z:0);
    debug(" bDim = {%d, %d, %d} \n", bDim?bDim->x:0, bDim?bDim->y:0, bDim?bDim->z:0);
    debug(" gDim = {%d, %d, %d} \n", gDim?gDim->x:0, gDim?gDim->y:0, gDim?gDim->z:0);
    buf_size = (uint32_t)(strlen(deviceName)+1);

    struct CUfunc_st* func;
    if( (func = lookup_func_by_name(ctx->head_mod, deviceName)) == NULL) {
        error("Failed to lookup kernel name %s in cubin\n", deviceName);
        return;
    }
    func->raw_func.host_func = (void *)hostFun;

    memset(&arg, 0, ARG_LEN);
    arg.cmd     = VIRTIO_CUDA_REGISTERFUNCTION;
    arg.src     = (uint64_t)fatBinHeader;
    arg.srcSize = (uint32_t)(fatBinHeader->fatSize + fatBinHeader->headerSize);
    arg.dst     = (uint64_t)deviceName;
    arg.dstSize = (uint32_t)(strlen(deviceName)+1); // +1 in order to keep \x00
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    arg.flag    = (uint64_t)hostFun;
    // batch 
    if(p_binary->total_size < sizeof(fatbin_buf_t) + p_binary->size + buf_size + sizeof(function_buf_t)) {
        debug("realloc size %x to %lx\n", p_binary->total_size, 
                sizeof(fatbin_buf_t) + p_binary->size + buf_size + sizeof(function_buf_t) );
        p_binary->total_size = roundup(sizeof(fatbin_buf_t) + p_binary->size + buf_size + sizeof(function_buf_t), 
                                    p_binary->block_size);
        offset = (unsigned long)p_last_binary - (unsigned long)p_binary;
        p_binary = (fatbin_buf_t *)realloc(p_binary, p_binary->total_size);
        debug("realloc total_size %x\n", p_binary->total_size);
        p_last_binary = (cubin_buf_t *)((void*)p_binary + offset);
    }
    function_buf_t *p_func = (function_buf_t *)(p_binary->buf + p_binary->size);
    p_func->size    = buf_size;
    p_func->hostFun = (uint64_t)hostFun;
    memcpy(p_func->buf, deviceName, buf_size);
    p_binary->size += buf_size + sizeof(function_buf_t);
    debug("p_binary->size = %x\n", p_binary->size);
    debug("last binary nr_func = %x\n", p_last_binary->nr_func);
    p_last_binary->nr_func += 1;
    // send_to_device(VIRTIO_IOC_REGISTERFUNCTION, &arg);
    // if(arg.cmd != cudaSuccess)
    // {
    //     error(" functions are not registered successfully.\n");
    //     exit(-1);
    // }
    return;
}

extern "C" void __cudaRegisterVar(
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
    uint32_t buf_size = 0;
    unsigned long offset=0;
    func();
    if(!p_last_binary)
        return;

    fatBinHeader = (computeFatBinaryFormat_t)(*fatCubinHandle);
    debug(" fatCubinHandle = %p, value =%p\n", fatCubinHandle, *fatCubinHandle);
    debug(" hostVar = %p, value =%s\n", hostVar, hostVar);
    debug(" deviceAddress = %p, value = %s\n", deviceAddress, deviceAddress);
    debug(" deviceName = %s\n", deviceName);
    debug(" ext = %d\n", ext);
    debug(" size = %d\n", size);
    debug(" constant = %d\n", constant);
    debug(" global = %d\n", global);
    buf_size = (uint32_t)(strlen(deviceName)+1);

    struct cuda_const_symbol* cs;
    if( (cs = lookup_symbol_by_name(ctx->head_mod, deviceName)) == NULL) {
        error("Failed to lookup symbol name %s in cubin\n", deviceName);
        return;
    }
    cs->host_var = (void *)hostVar;

    memset(&arg, 0, ARG_LEN);
    arg.cmd     = VIRTIO_CUDA_REGISTERVAR;
    arg.src     = (uint64_t)fatBinHeader;
    arg.srcSize = (uint32_t)(fatBinHeader->fatSize + fatBinHeader->headerSize);
    arg.dst     = (uint64_t)deviceName;
    arg.dstSize = (uint32_t)(strlen(deviceName)+1); // +1 in order to keep \x00
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    arg.flag    = (uint64_t)hostVar;
    arg.param   = (uint64_t)constant;
    arg.param2  = (uint64_t)global;
    // batch 
    if(p_binary->total_size < sizeof(fatbin_buf_t) + p_binary->size + buf_size + sizeof(var_buf_t)) {
        debug("realloc size %lx to %lx\n", p_binary->total_size, 
               sizeof(fatbin_buf_t) + p_binary->size + buf_size + sizeof(var_buf_t) );
        p_binary->total_size = roundup(sizeof(fatbin_buf_t) + p_binary->size + buf_size + sizeof(var_buf_t), 
                                    p_binary->block_size);
        offset = (unsigned long)p_last_binary - (unsigned long)p_binary;
        p_binary = (fatbin_buf_t *)realloc(p_binary, p_binary->total_size);
        debug("realloc size %x\n", p_binary->total_size);
        p_last_binary = (cubin_buf_t *)((void*)p_binary + offset);
    }
    var_buf_t *p_var = (var_buf_t *)(p_binary->buf + p_binary->size);
    p_var->size     = buf_size;
    p_var->hostVar  = (uint64_t)hostVar;
    p_var->constant = constant;
    p_var->global   = global;
    memcpy(p_var->buf, deviceName, buf_size);
    p_binary->size += buf_size + sizeof(var_buf_t);
    p_last_binary->nr_var += 1;
    // send_to_device(VIRTIO_IOC_REGISTERVAR, &arg);
    // if(arg.cmd != cudaSuccess)
    // {
    //     error(" functions are not registered successfully.\n");
    // }
}

/*extern "C" void __cudaRegisterManagedVar(
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
}*/

/*extern "C" void __cudaRegisterTexture(
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
}*/

/*extern "C" void __cudaRegisterSurface(
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
}*/

extern "C" char __cudaInitModule(void **fatCubinHandle)
{
    func();
    return 'U';
}

extern "C" unsigned  __cudaPushCallConfiguration(
    dim3 gridDim,
    dim3 blockDim, 
    size_t sharedMem = 0, 
    void *stream = 0)
{
    func();
    debug("gridDim= %u %u %u\n", gridDim.x, gridDim.y, gridDim.z);  
    debug("blockDim= %u %u %u\n", blockDim.x, blockDim.y, blockDim.z);
    debug("sharedMem= %zu\n", sharedMem);
    debug("stream= %lu\n", (cudaStream_t)(stream));
    

    memset(&kernelConf, 0, sizeof(KernelConf_t));
    kernelConf.gridDim      = gridDim;
    kernelConf.blockDim     = blockDim;
    kernelConf.sharedMem    = sharedMem;
    kernelConf.stream       = stream;
    return 0;
}

extern "C" cudaError_t  __cudaPopCallConfiguration(
    dim3         *gridDim,
    dim3         *blockDim,
    size_t       *sharedMem,
    void         *stream
)
{
    func();
    *gridDim = kernelConf.gridDim;
    *blockDim = kernelConf.blockDim;
    *sharedMem = kernelConf.sharedMem;
    stream = kernelConf.stream;
    return cudaSuccess;
}

/*
CUDA10
*/
extern "C" cudaError_t cudaLaunchKernel(
    const void *hostFunc,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream
)
{
    struct cuda_param *param_data = NULL;
    struct CUkernel_st *kernel_param;
    func();
    debug("hostFunc %lx\n", (uint64_t)hostFunc);
    debug("gridDim= %u %u %u\n", gridDim.x, gridDim.y, gridDim.z);  
    debug("blockDim= %u %u %u\n", blockDim.x, blockDim.y, blockDim.z);
    debug("sharedMem= %zu\n", sharedMem);
    debug("stream= %lu\n", stream);

    struct CUfunc_st* func;
    if( (func = lookup_func_by_hostfunc(ctx->head_mod, hostFunc)) == NULL) {
        error("Failed to lookup kernel host func %p in cubin\n", hostFunc);
        return cudaErrorUnknown;
    }
    struct cuda_raw_func *raw_func = &func->raw_func;
    debug("kernel name %s\n", raw_func->name);
    debug("kernel param count %d\n", raw_func->param_count);
    debug("kernel param size %d\n", raw_func->param_size);
    kernel_param = (struct CUkernel_st*)malloc(sizeof(struct CUkernel_st) + raw_func->param_size);
    kernel_param->grid_x = gridDim.x;
    kernel_param->grid_y = gridDim.y;
    kernel_param->grid_z = gridDim.z;
    kernel_param->block_x = blockDim.x;
    kernel_param->block_y = blockDim.y;
    kernel_param->block_z = blockDim.z;
    kernel_param->smem_size = sharedMem;
    kernel_param->stream = (uint64_t)stream;
    kernel_param->param_nr = raw_func->param_count;
    kernel_param->param_size = raw_func->param_size;
    param_data = raw_func->param_data;
    while (param_data) {
        debug("\tparam{%d, 0x%x, 0x%x},\n", 
                   param_data->idx, 
                   param_data->offset, 
                   param_data->size);
        memcpy(&kernel_param->param_buf[param_data->offset], 
                    args[param_data->idx], param_data->size);
        param_data = param_data->next;
    }
    /*
    args = {&arg0, &arg1};
    debug("ptr=%p, content=%p\n", args[0], *((void**)args[0]));
    debug("ptr=%p, content=%f\n", args[1], *(float*)args[1]);
    if (args[2] == NULL)
        debug("NULL\n");
    */
    return cudaSuccess;
}

/*
CUDA9
*/
extern "C" cudaError_t cudaConfigureCall(
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
    kernelConf.gridDim      = gridDim;
    kernelConf.blockDim     = blockDim;
    kernelConf.sharedMem    = sharedMem;
    kernelConf.stream       = stream;
    // Do not invoke ioctl
    return cudaSuccess;
}

/*
CUDA9
*/
extern "C" cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset)
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
    cudaParaSize += (uint32_t)sizeof(uint32_t);
    
    memcpy(&cudaKernelPara[cudaParaSize], arg, size);
    debug("value = 0x%llx\n", *(unsigned long long*)&cudaKernelPara[cudaParaSize]);
    cudaParaSize += (uint32_t)size;
    (*((uint32_t*)cudaKernelPara))++;
    return cudaSuccess;
}

extern "C" cudaError_t cudaLaunch(const void *entry)
{
    VirtIOArg arg;
    unsigned char *para;
    func();
    api_inc(API_KERNEL);
    init_primary_context();
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
        para_idx += (uint32_t)(*(uint32_t*)(&para[para_idx]) + sizeof(uint32_t));
    }
    debug("entry %lx\n", (uint64_t)entry);
    arg.srcSize = (uint32_t)cudaParaSize;
    arg.dst     = (uint64_t)&kernelConf;
    arg.dstSize = (uint32_t)sizeof(KernelConf_t);
    arg.flag    = (uint64_t)entry;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_LAUNCH, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    VirtIOArg arg;
    func();
    api_inc(API_MEMCPY);
    init_primary_context();
    if (kind <0 || kind >4) {
        return cudaErrorInvalidMemcpyDirection;
    }
    debug("cudaMemcpyKind %d\n", kind);
    if(!dst || !src || count<=0)
        return cudaSuccess;
    mem_inc(kind, count);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_MEMCPY;
    arg.flag    = kind;
    arg.src     = (uint64_t)src;
    arg.srcSize = (uint32_t)count;
    arg.dst     = (uint64_t)dst;
    arg.dstSize = (uint32_t)count;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    debug("gettid %d\n", arg.tid);
    send_to_device(VIRTIO_IOC_MEMCPY, &arg);
    if (arg.flag == cudaMemcpyHostToHost) {
        memcpy(dst, src, count);
        return cudaSuccess;
    }
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaMemcpyToSymbol( const void *symbol, const void *src, 
                                size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    VirtIOArg arg;
    func();
    api_inc(API_MEMCPY);
    init_primary_context();
    if(!symbol || !src || count<=0)
        return cudaSuccess;
    assert(kind == cudaMemcpyHostToDevice);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_MEMCPYTOSYMBOL;
    arg.flag    = kind;
    arg.src     = (uint64_t)src;
    arg.srcSize = (uint32_t)count;
    debug("symbol is %p\n", symbol);
    arg.dst     = (uint64_t)symbol;
    arg.dstSize = 0;
    arg.param   = offset;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_MEMCPYTOSYMBOL, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaMemcpyFromSymbol(   void *dst, const void *symbol, 
                                    size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    VirtIOArg arg;
    func();
    api_inc(API_MEMCPY);
    init_primary_context();
    if(!dst || !symbol || count<=0)
        return cudaSuccess;
    assert(kind == cudaMemcpyDeviceToHost);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_MEMCPYFROMSYMBOL;
    arg.flag    = kind;
    debug("symbol is %p\n", symbol);
    arg.src     = (uint64_t)symbol;
    arg.dst     = (uint64_t)dst;
    arg.dstSize = (uint32_t)count;
    arg.param   = (uint64_t)offset;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_MEMCPYFROMSYMBOL, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaMemset(void *dst, int value, size_t count)
{
    VirtIOArg arg;
    func();
    api_inc(API_MEM);
    init_primary_context();
    if(!dst || count<=0)
        return cudaSuccess;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_MEMSET;
    arg.dst     = (uint64_t)dst;
    arg.param   = (uint32_t)count;
    arg.param2  = (uint64_t)value;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_MEMSET, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaMemcpyAsync(
            void *dst, 
            const void *src, 
            size_t count, 
            enum cudaMemcpyKind kind,
            cudaStream_t stream
            )
{
    VirtIOArg arg;
    func();
    api_inc(API_MEMCPY);
    if (kind <0 || kind >4) {
        debug("direction is %d\n", kind);
        return cudaErrorInvalidMemcpyDirection;
    }
    if(!dst || !src || count<=0)
        return cudaSuccess;
    mem_inc(kind, count);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_MEMCPY_ASYNC;
    arg.flag    = kind;
    arg.src     = (uint64_t)src;
    arg.srcSize = (uint32_t)count;
    arg.dst     = (uint64_t)dst;
    arg.src2    = (uint64_t)stream;
    arg.param   = (uint64_t)stream;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_MEMCPY_ASYNC, &arg);
    if (arg.flag == cudaMemcpyHostToHost) {
        memcpy(dst, src, count);
        return cudaSuccess;
    }
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    VirtIOArg arg;
    func();
    api_inc(API_MEM);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_MALLOC;
    arg.src     = (uint64_t)NULL;
    arg.srcSize = (uint32_t)size;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    debug("tid %d\n", arg.tid);
    send_to_device(VIRTIO_IOC_MALLOC, &arg);
    *devPtr = (void *)arg.dst;
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
    VirtIOArg arg;
    func();
    api_inc(API_MEM);
    if(size<=0)
        return cudaSuccess;
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_HOSTREGISTER;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    arg.src     = (uint64_t)ptr;
    arg.srcSize = (uint32_t)size;
    arg.flag    = (uint64_t)flags;
    send_to_device(VIRTIO_IOC_HOSTREGISTER, &arg);
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaHostUnregister(void *ptr)
{
    VirtIOArg arg;
    func();
    api_inc(API_MEM);
    if(!ptr)
        return cudaSuccess;
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_HOSTUNREGISTER;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    arg.src = (uint64_t)ptr;
    send_to_device(VIRTIO_IOC_HOSTUNREGISTER, &arg);
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    func();
    api_inc(API_MEM);
    init_primary_context();
    if(size <=0)
        return cudaSuccess;
    *pHost = my_malloc(size);
    debug("*pHost = %p\n", *pHost);
    return cudaHostRegister(*pHost, size, flags);
}

extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    func();
    api_inc(API_MEM);
    debug("allocating size 0x%lx\n", size);
    return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

extern "C" cudaError_t cudaFreeHost(void *ptr)
{
    BlockHeader* blk = NULL;
    func();
    api_inc(API_MEM);
    init_primary_context();
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

extern "C" cudaError_t cudaFree(void *devPtr)
{
    VirtIOArg arg;
    func();
    api_inc(API_MEM);
    init_primary_context();
    if(!devPtr)
        return cudaSuccess;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_FREE;
    arg.src     = (uint64_t)devPtr;
    arg.srcSize = 0;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_FREE, &arg);
    return (cudaError_t)arg.cmd;    
}

/**************************************************/
// start of device management

extern "C" cudaError_t cudaGetDevice(int *device)
{
    VirtIOArg arg;
    func();
    api_inc(API_DEVICE);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_GETDEVICE;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_GETDEVICE, &arg);
    *device     = (int)arg.flag;
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    VirtIOArg arg;
    func();
    api_inc(API_DEVICE);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_GETDEVICEPROPERTIES;
    arg.dst     = (uint64_t)prop;
    arg.dstSize = sizeof(struct cudaDeviceProp);
    arg.flag    = device;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_GETDEVICEPROPERTIES, &arg);
    if(!prop)
        return cudaErrorInvalidDevice;
    return cudaSuccess;
}

extern "C" cudaError_t cudaSetDevice(int device)
{
    VirtIOArg arg;
    func();
    api_inc(API_DEVICE);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_SETDEVICE;
    arg.flag    = device;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    debug("gettid %d\n", arg.tid);
    send_to_device(VIRTIO_IOC_SETDEVICE, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaGetDeviceCount(int *count)
{
    VirtIOArg arg;
    func();
    api_inc(API_DEVICE);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_GETDEVICECOUNT;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_GETDEVICECOUNT, &arg);
    *count  = (int)arg.flag;
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    VirtIOArg arg;
    func();
    api_inc(API_DEVICE);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_DEVICESETCACHECONFIG;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    arg.flag = (uint64_t)cacheConfig;
    send_to_device(VIRTIO_IOC_DEVICESETCACHECONFIG, &arg);
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaDeviceReset(void)
{
    VirtIOArg arg;
    func();
    api_inc(API_DEVICE);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_DEVICERESET;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_DEVICERESET, &arg);
    return cudaSuccess; 
}

extern "C" cudaError_t cudaDeviceSynchronize(void)
{
    VirtIOArg arg;
    func();
    api_inc(API_DEVICE);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_DEVICESYNCHRONIZE;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_DEVICESYNCHRONIZE, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
    VirtIOArg arg;
    func();
    api_inc(API_DEVICE);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_SETDEVICEFLAGS;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    arg.flag    = (uint64_t)flags;
    send_to_device(VIRTIO_IOC_SETDEVICEFLAGS, &arg);
    return (cudaError_t)arg.cmd;
}
// end of device management
/**************************************************/

extern "C" cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
    VirtIOArg arg;
    func();
    api_inc(API_STREAM);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_STREAMCREATE;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_STREAMCREATE, &arg);
    *pStream = (cudaStream_t)arg.flag;
    debug("stream = 0x%lx\n", (uint64_t)(*pStream));
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    VirtIOArg arg;
    func();
    api_inc(API_STREAM);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_STREAMCREATEWITHFLAGS;
    arg.flag    = flags;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_STREAMCREATEWITHFLAGS, &arg);
     *pStream   = (cudaStream_t)arg.dst;
     debug("stream = 0x%lx\n", (uint64_t)(*pStream));
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    VirtIOArg arg;
    func();
    api_inc(API_STREAM);
    init_primary_context();
    debug("stream = 0x%lx\n", (uint64_t)stream);
    if(stream==0)
        return cudaSuccess;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_STREAMDESTROY;
    arg.flag    = (uint64_t)stream;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_STREAMDESTROY, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    VirtIOArg arg;
    func();
    api_inc(API_STREAM);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_STREAMSYNCHRONIZE;
    arg.flag    = (uint64_t)stream;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_STREAMSYNCHRONIZE, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                cudaEvent_t event, unsigned int flags)
{
    VirtIOArg arg;
    func();
    api_inc(API_STREAM);
    init_primary_context();
    if(event == 0)
        return cudaSuccess;
    assert(flags == 0);
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_STREAMWAITEVENT;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    arg.src = (uint64_t)stream;
    arg.dst = (uint64_t)event;
    send_to_device(VIRTIO_IOC_STREAMWAITEVENT, &arg);
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    VirtIOArg arg;
    func();
    api_inc(API_EVENT);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_EVENTCREATE;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_EVENTCREATE, &arg);
    *event = (cudaEvent_t)arg.flag;
    debug("tid %d create event is 0x%lx\n", arg.tid, (uint64_t)(*event));
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    VirtIOArg arg;
    func();
    api_inc(API_EVENT);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_EVENTCREATEWITHFLAGS;
    arg.flag    = (uint64_t)flags;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_EVENTCREATEWITHFLAGS, &arg);
    *event  = (cudaEvent_t)arg.dst;
    debug("event is 0x%lx\n", (uint64_t)(*event));
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    VirtIOArg arg;
    func();
    api_inc(API_EVENT);
    init_primary_context();
    if (event == 0)
        return cudaSuccess;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_EVENTDESTROY;
    arg.flag    = (uint64_t)event;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    debug("tid %d destroy event is 0x%lx\n", arg.tid, (uint64_t)(event));
    send_to_device(VIRTIO_IOC_EVENTDESTROY, &arg);
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    VirtIOArg arg;
    func();
    api_inc(API_EVENT);
    init_primary_context();
    debug("event is 0x%lx\n", (uint64_t)event);
    if(event==0)
        return cudaSuccess;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_EVENTRECORD;
    debug("event  = 0x%lx\n", (uint64_t)event);
    debug("stream = 0x%lx\n", (uint64_t)stream);
    arg.src = (uint64_t)event;
    arg.dst = (uint64_t)stream;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_EVENTRECORD, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    VirtIOArg arg;
    func();
    api_inc(API_EVENT);
    init_primary_context();
    if(event==0)
        return cudaSuccess;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_EVENTSYNCHRONIZE;
    arg.flag    = (uint64_t)event;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_EVENTSYNCHRONIZE, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event)
{
    VirtIOArg arg;
    func();
    api_inc(API_EVENT);
    init_primary_context();
    if(event==0)
        return cudaSuccess;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUDA_EVENTQUERY;
    arg.flag    = (uint64_t)event;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_EVENTQUERY, &arg);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    VirtIOArg arg;
    func();
    api_inc(API_EVENT);
    init_primary_context();
    if(start==0 || end==0)
        return cudaSuccess;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_EVENTELAPSEDTIME;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    arg.src = (uint64_t)start;
    arg.dst = (uint64_t)end;
    arg.param       = (uint64_t)ms;
    arg.paramSize   = sizeof(float);
    send_to_device(VIRTIO_IOC_EVENTELAPSEDTIME, &arg);
    debug("elapsed time is %g\n", *ms);
    return (cudaError_t)arg.cmd;    
}

extern "C" cudaError_t cudaThreadSynchronize()
{
    VirtIOArg arg;
    func();
    api_inc(API_THREAD);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_THREADSYNCHRONIZE;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_THREADSYNCHRONIZE, &arg);
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaGetLastError(void)
{
    VirtIOArg arg;
    func();
    api_inc(API_ERROR);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_GETLASTERROR;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_GETLASTERROR, &arg);
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaPeekAtLastError(void)
{
    VirtIOArg arg;
    func();
    api_inc(API_ERROR);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_PEEKATLASTERROR;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_PEEKATLASTERROR, &arg);
    return (cudaError_t)arg.cmd;
}

extern "C" cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
    VirtIOArg arg;
    func();
    api_inc(API_MEM);
    init_primary_context();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUDA_MEMGETINFO;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_MEMGETINFO, &arg);
    *free   = (size_t)arg.srcSize;
    *total  = (size_t)arg.dstSize;
    return (cudaError_t)arg.cmd;
}


extern "C" const char *cudaGetErrorString(cudaError_t error)
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

extern "C" CUBLASAPI cublasStatus_t cublasCreate_v2 (cublasHandle_t *handle)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_CREATE;
    arg.srcSize = sizeof(cublasHandle_t);
    arg.src     = (uint64_t)handle;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    // debug("sizeof(cublasHandle_t)=%lx\n", sizeof(cublasHandle_t));
    send_to_device(VIRTIO_IOC_CUBLAS_CREATE, &arg);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasDestroy_v2 (cublasHandle_t handle)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_DESTROY;
    arg.srcSize = sizeof(cublasHandle_t);
    arg.src     = (uint64_t)&handle;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_CUBLAS_DESTROY, &arg);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasSetVector (int n, int elemSize, const void *x, 
                                             int incx, void *devicePtr, int incy)
{
    VirtIOArg arg;
    uint8_t *buf = NULL;
    int len = 0;
    int idx = 0;
    int int_size = sizeof(int);
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_SETVECTOR;
    arg.srcSize = n * elemSize;
    arg.src     = (uint64_t)x;
    arg.dst     = (uint64_t)devicePtr;
    len = int_size * 4 ;
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, &n, int_size);
    idx += int_size;
    memcpy(buf+idx, &elemSize, int_size);
    idx += int_size;
    memcpy(buf+idx, &incx, int_size);
    idx += int_size;
    memcpy(buf+idx, &incy, int_size);
    arg.param       = (uint64_t)buf;
    arg.paramSize   = len;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_CUBLAS_SETVECTOR, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasGetVector (int n, int elemSize, const void *x, 
                                             int incx, void *y, int incy)
{
    VirtIOArg arg;
    uint8_t *buf = NULL;
    int len = 0;
    int idx = 0;
    int int_size = sizeof(int);
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_GETVECTOR;
    arg.srcSize = n * elemSize;
    arg.src     = (uint64_t)x;
    arg.dst     = (uint64_t)y;
    len = int_size * 4 ;
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, &n, int_size);
    idx += int_size;
    memcpy(buf+idx, &elemSize, int_size);
    idx += int_size;
    memcpy(buf+idx, &incx, int_size);
    idx += int_size;
    memcpy(buf+idx, &incy, int_size);
    arg.param       = (uint64_t)buf;
    arg.paramSize   = len;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_CUBLAS_GETVECTOR, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasSetStream_v2 (cublasHandle_t handle, cudaStream_t streamId)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUBLAS_SETSTREAM;
    debug("stream = 0x%lx\n", (uint64_t)streamId);
    arg.src = (uint64_t)handle;
    arg.dst = (uint64_t)streamId;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_CUBLAS_SETSTREAM, &arg);
    return (cublasStatus_t)arg.cmd; 
}


extern "C" CUBLASAPI cublasStatus_t cublasGetStream_v2 (cublasHandle_t handle, cudaStream_t *streamId)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd = VIRTIO_CUBLAS_GETSTREAM;
    debug("stream = 0x%lx\n", (uint64_t)streamId);
    arg.src = (uint64_t)handle;
    arg.tid = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_CUBLAS_GETSTREAM, &arg);
    *streamId = (cudaStream_t)arg.flag;
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasSasum_v2(cublasHandle_t handle, 
                                         int n, 
                                         const float *x, 
                                         int incx, 
                                         float *result) /* host or device pointer */
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_SASUM;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.srcSize2 = (uint32_t)incx;
    arg.dst     = (uint64_t)handle;
    arg.param       = (uint64_t)result;
    arg.paramSize   = sizeof(float);
    send_to_device(VIRTIO_IOC_CUBLAS_SASUM, &arg);
    debug("result = %g\n", *result);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasDasum_v2(cublasHandle_t handle, 
                                     int n, 
                                     const double *x, 
                                     int incx, 
                                     double *result) /* host or device pointer */
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_DASUM;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.srcSize2    = (uint32_t)incx;
    arg.dst     = (uint64_t)handle;
    arg.param       = (uint64_t)result;
    arg.paramSize   = sizeof(double);
    send_to_device(VIRTIO_IOC_CUBLAS_DASUM, &arg);
    debug("result = %g\n", *result);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasScopy_v2 (cublasHandle_t handle,
                                          int n, 
                                          const float *x, 
                                          int incx, 
                                          float *y, 
                                          int incy)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_SCOPY;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.dst     = (uint64_t)y;
    arg.src2    = (uint64_t)handle;
    arg.srcSize2    = (uint32_t)incx;
    arg.dstSize     = (uint32_t)incy;
    send_to_device(VIRTIO_IOC_CUBLAS_SCOPY, &arg);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasDcopy_v2 (cublasHandle_t handle,
                                          int n, 
                                          const double *x, 
                                          int incx, 
                                          double *y, 
                                          int incy)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_DCOPY;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.dst     = (uint64_t)y;
    arg.src2    = (uint64_t)handle;
    arg.srcSize2    = (uint32_t)incx;
    arg.dstSize     = (uint32_t)incy;
    send_to_device(VIRTIO_IOC_CUBLAS_DCOPY, &arg);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasSdot_v2 (cublasHandle_t handle,
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
    arg.cmd     = VIRTIO_CUBLAS_SDOT;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.dst     = (uint64_t)y;
    arg.src2    = (uint64_t)handle;
    arg.srcSize2    = (uint32_t)incx;
    arg.dstSize     = (uint32_t)incy;
    arg.param       = (uint64_t)result;
    arg.paramSize   = sizeof(float);
    send_to_device(VIRTIO_IOC_CUBLAS_SDOT, &arg);
    debug("result = %g\n", *result);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasDdot_v2 (cublasHandle_t handle,
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
    arg.cmd     = VIRTIO_CUBLAS_DDOT;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.src2    = (uint64_t)handle;
    arg.dst     = (uint64_t)y;
    arg.srcSize2    = (uint32_t)incx;
    arg.dstSize     = (uint32_t)incy;
    arg.param       = (uint64_t)result;
    arg.paramSize   = sizeof(double);
    send_to_device(VIRTIO_IOC_CUBLAS_DDOT, &arg);
    debug("result = %g\n", *result);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasSaxpy_v2 (cublasHandle_t handle,
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
    arg.cmd     = VIRTIO_CUBLAS_SAXPY;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.dst     = (uint64_t)y;
    arg.src2    = (uint64_t)handle;
    arg.srcSize2    = (uint32_t)incx;
    arg.dstSize     = (uint32_t)incy;
    len = sizeof(float);
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, alpha, sizeof(float));
    arg.param       = (uint64_t)buf;
    arg.paramSize   = (uint32_t)len;
    send_to_device(VIRTIO_IOC_CUBLAS_SAXPY, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasDaxpy_v2 (cublasHandle_t handle,
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
    arg.cmd     = VIRTIO_CUBLAS_DAXPY;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.dst     = (uint64_t)y;
    arg.src2    = (uint64_t)handle;
    arg.srcSize2    = (uint32_t)incx;
    arg.dstSize = (uint32_t)incy;
    len = sizeof(double);
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, alpha, sizeof(double));
    arg.param       = (uint64_t)buf;
    arg.paramSize   = (uint32_t)len;
    send_to_device(VIRTIO_IOC_CUBLAS_DAXPY, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasSscal_v2(cublasHandle_t handle, 
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
    arg.cmd     = VIRTIO_CUBLAS_SSCAL;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.src2    = (uint64_t)handle;
    arg.srcSize2 = (uint32_t)incx;
    len = sizeof(float);
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, alpha, sizeof(float));
    arg.param       = (uint64_t)buf;
    arg.paramSize   = (uint32_t)len;
    send_to_device(VIRTIO_IOC_CUBLAS_SSCAL, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}
    
extern "C" CUBLASAPI cublasStatus_t cublasDscal_v2(cublasHandle_t handle, 
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
    arg.cmd     = VIRTIO_CUBLAS_DSCAL;
    arg.src     = (uint64_t)x;
    arg.srcSize = (uint32_t)n;
    arg.src2    = (uint64_t)handle;
    arg.srcSize2 = (uint32_t)incx;
    len = sizeof(double);
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, alpha, sizeof(double));
    arg.param       = (uint64_t)buf;
    arg.paramSize   = (uint32_t)len;
    send_to_device(VIRTIO_IOC_CUBLAS_DSCAL, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}

/* GEMV */
extern "C" CUBLASAPI cublasStatus_t cublasSgemv_v2 (cublasHandle_t handle, 
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
    arg.cmd     = VIRTIO_CUBLAS_SGEMV;
    arg.src     = (uint64_t)A;
    arg.srcSize = m * n;
    arg.src2    = (uint64_t)x;
    arg.srcSize2 = n;
    arg.dst     = (uint64_t)y;
    arg.dstSize = m;

    len = (uint32_t)(int_size * 3 + sizeof(cublasHandle_t) + 
            sizeof(cublasOperation_t) + sizeof(float)*2);
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, &handle, sizeof(cublasHandle_t));
    idx += (uint32_t)sizeof(cublasHandle_t);
    memcpy(buf+idx, &trans, sizeof(cublasOperation_t));
    idx += (uint32_t)sizeof(cublasOperation_t);
    memcpy(buf+idx, &lda, int_size);
    idx += int_size;
    memcpy(buf+idx, &incx, int_size);
    idx += int_size;
    memcpy(buf+idx, &incy, int_size);
    idx += int_size;
    memcpy(buf+idx, alpha, sizeof(float));
    idx += (uint32_t)sizeof(float);
    memcpy(buf+idx, beta, sizeof(float));
    arg.param       = (uint64_t)buf;
    arg.paramSize   = (uint32_t)len;
    send_to_device(VIRTIO_IOC_CUBLAS_SGEMV, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}
 
extern "C" CUBLASAPI cublasStatus_t cublasDgemv_v2 (cublasHandle_t handle, 
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
    uint32_t len = 0;
    uint32_t idx = 0;
    uint32_t int_size = sizeof(int);

    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_DGEMV;
    arg.src     = (uint64_t)A;
    arg.srcSize = (uint32_t)(m * n);
    arg.src2    = (uint64_t)x;
    arg.srcSize2 = (uint32_t)n;
    arg.dst     = (uint64_t)y;
    arg.dstSize = (uint32_t)m;
    len = (uint32_t)(int_size * 3 + sizeof(cublasHandle_t) + 
            sizeof(cublasOperation_t)+ sizeof(double)*2);
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, &handle, sizeof(cublasHandle_t));
    idx += (uint32_t)sizeof(cublasHandle_t);
    memcpy(buf+idx, &trans, sizeof(cublasOperation_t));
    idx += (uint32_t)sizeof(cublasOperation_t);
    memcpy(buf+idx, &lda, int_size);
    idx += int_size;
    memcpy(buf+idx, &incx, int_size);
    idx += int_size;
    memcpy(buf+idx, &incy, int_size);
    idx += int_size;
    memcpy(buf+idx, alpha, sizeof(double));
    idx += (uint32_t)sizeof(double);
    memcpy(buf+idx, beta, sizeof(double));
    arg.param       = (uint64_t)buf;
    arg.paramSize   = len;
    send_to_device(VIRTIO_IOC_CUBLAS_DGEMV, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}

/* GEMM */
extern "C" CUBLASAPI cublasStatus_t cublasSgemm_v2 (cublasHandle_t handle, 
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
    uint32_t len = 0;
    uint32_t idx = 0;
    uint32_t int_size = sizeof(int);

    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_SGEMM;
    arg.src     = (uint64_t)A;
    arg.srcSize = m * k;
    arg.src2    = (uint64_t)B;
    arg.srcSize2 = k * n;
    arg.dst     = (uint64_t)C;
    arg.dstSize = m * n;
    len = int_size * 6 + (uint32_t)sizeof(cublasHandle_t) + 
            (uint32_t)sizeof(cublasOperation_t)*2 + (uint32_t)sizeof(float)*2;
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, &handle, sizeof(cublasHandle_t));
    idx += (uint32_t)sizeof(cublasHandle_t);
    memcpy(buf+idx, &transa, sizeof(cublasOperation_t));
    idx += (uint32_t)sizeof(cublasOperation_t);
    memcpy(buf+idx, &transb, sizeof(cublasOperation_t));
    idx += (uint32_t)sizeof(cublasOperation_t);
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
    idx += (uint32_t)sizeof(float);
    memcpy(buf+idx, beta, sizeof(float));
    arg.param       = (uint64_t)buf;
    arg.paramSize   = len;
    send_to_device(VIRTIO_IOC_CUBLAS_SGEMM, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}

extern "C" CUBLASAPI cublasStatus_t cublasDgemm_v2 (cublasHandle_t handle, 
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
    uint64_t len = 0;
    uint64_t idx = 0;
    uint32_t int_size = sizeof(int);

    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_DGEMM;
    arg.src     = (uint64_t)A;
    arg.srcSize = m * k;
    arg.src2    = (uint64_t)B;
    arg.srcSize2 = k * n;
    arg.dst     = (uint64_t)C;
    arg.dstSize = m * n;
    len = int_size * 6 + sizeof(cublasHandle_t) + 
            sizeof(cublasOperation_t)*2 + sizeof(double)*2;
    buf = (uint8_t *)__libc_malloc(len);
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
    arg.param       = (uint64_t)buf;
    arg.paramSize   = (uint32_t)len;
    send_to_device(VIRTIO_IOC_CUBLAS_DGEMM, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}

extern "C" cublasStatus_t cublasSetMatrix (int rows, int cols, int elemSize, 
                                 const void *A, int lda, void *B, 
                                 int ldb)
{
    VirtIOArg arg;
    uint8_t *buf = NULL;
    uint32_t len = 0;
    uint32_t idx = 0;
    uint32_t int_size = sizeof(int);
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_SETMATRIX;
    arg.src     = (uint64_t)A;
    arg.srcSize = (uint32_t)(rows * cols * elemSize);
    arg.dst     = (uint64_t)B;
    len = int_size * 5;
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, &rows, int_size);
    idx += int_size;
    memcpy(buf+idx, &cols, int_size);
    idx += int_size;
    memcpy(buf+idx, &elemSize, int_size);
    idx += int_size;
    memcpy(buf+idx, &lda, int_size);
    idx += int_size;
    memcpy(buf+idx, &ldb, int_size);
    arg.param       = (uint64_t)buf;
    arg.paramSize   = len;
    send_to_device(VIRTIO_IOC_CUBLAS_SETMATRIX, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}

extern "C" cublasStatus_t cublasGetMatrix (int rows, int cols, int elemSize, 
                                 const void *A, int lda, void *B,
                                 int ldb)
{
    VirtIOArg arg;
    uint8_t *buf = NULL;
    uint32_t len = 0;
    uint32_t idx = 0;
    uint32_t int_size = sizeof(int);
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CUBLAS_GETMATRIX;
    arg.src     = (uint64_t)A;
    arg.srcSize = (uint32_t)(rows * cols * elemSize);
    arg.dst     = (uint64_t)B;
    len = int_size * 5;
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, &rows, int_size);
    idx += int_size;
    memcpy(buf+idx, &cols, int_size);
    idx += int_size;
    memcpy(buf+idx, &elemSize, int_size);
    idx += int_size;
    memcpy(buf+idx, &lda, int_size);
    idx += int_size;
    memcpy(buf+idx, &ldb, int_size);
    arg.param       = (uint64_t)buf;
    arg.paramSize   = len;
    send_to_device(VIRTIO_IOC_CUBLAS_GETMATRIX, &arg);
    __libc_free(buf);
    return (cublasStatus_t)arg.cmd;
}
/*****************************************************************************/
/******CURAND***********/
/*****************************************************************************/


extern "C" curandStatus_t CURANDAPI 
curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CURAND_CREATEGENERATORHOST;
    arg.dst     = (uint64_t)rng_type;
    send_to_device(VIRTIO_IOC_CURAND_CREATEGENERATORHOST, &arg);
    *generator  = (curandGenerator_t)arg.flag;
    return (curandStatus_t)arg.cmd;
}

extern "C" curandStatus_t CURANDAPI 
curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CURAND_GENERATE;
    arg.src     = (uint64_t)generator;
    arg.dst     = (uint64_t)outputPtr;
    arg.dstSize = (uint32_t)(num*sizeof(unsigned int));
    arg.param   = (uint64_t)num;
    send_to_device(VIRTIO_IOC_CURAND_GENERATE, &arg);
    return (curandStatus_t)arg.cmd;
}

extern "C" curandStatus_t CURANDAPI 
curandGenerateNormal(curandGenerator_t generator, float *outputPtr, 
                     size_t n, float mean, float stddev)
{
    VirtIOArg arg;
    uint8_t *buf = NULL;
    uint64_t len = 0;
    uint64_t idx = 0;

    func();
    if(!generator)
        return CURAND_STATUS_SUCCESS;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CURAND_GENERATENORMAL;
    arg.src     = (uint64_t)generator;
    arg.dst     = (uint64_t)outputPtr;
    arg.dstSize = (uint32_t)(n*sizeof(float));
    arg.src2    = (uint64_t)n;
    len = sizeof(float)*2;
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, &mean, sizeof(float));
    idx += sizeof(float);
    memcpy(buf+idx, &stddev, sizeof(float));
    arg.param       = (uint64_t)buf;
    arg.paramSize   = (uint32_t)len;
    send_to_device(VIRTIO_IOC_CURAND_GENERATENORMAL, &arg);
    __libc_free(buf);
    return (curandStatus_t)arg.cmd;
}

extern "C" curandStatus_t CURANDAPI 
curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr, 
                     size_t n, double mean, double stddev)
{
    VirtIOArg arg;
    uint8_t *buf = NULL;
    uint64_t len = 0;
    uint64_t idx = 0;

    func();
    if(!generator)
        return CURAND_STATUS_SUCCESS;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CURAND_GENERATENORMALDOUBLE;
    arg.src     = (uint64_t)generator;
    arg.dst     = (uint64_t)outputPtr;
    arg.dstSize = (uint32_t)(n*sizeof(double));
    arg.src2    = (uint64_t)n;
    len = sizeof(double)*2;
    buf = (uint8_t *)__libc_malloc(len);
    memcpy(buf+idx, &mean, sizeof(double));
    idx += sizeof(double);
    memcpy(buf+idx, &stddev, sizeof(double));
    arg.param       = (uint64_t)buf;
    arg.paramSize   = (uint32_t)len;
    send_to_device(VIRTIO_IOC_CURAND_GENERATENORMALDOUBLE, &arg);
    __libc_free(buf);
    return (curandStatus_t)arg.cmd;
}

extern "C" curandStatus_t CURANDAPI 
curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num)
{
    VirtIOArg arg;
    func();
    if(!generator)
        return CURAND_STATUS_SUCCESS;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CURAND_GENERATEUNIFORM;
    arg.src     = (uint64_t)generator;
    arg.dst     = (uint64_t)outputPtr;
    arg.dstSize = (uint32_t)(num*sizeof(float));
    arg.param   = (uint64_t)num;
    send_to_device(VIRTIO_IOC_CURAND_GENERATEUNIFORM, &arg);
    return (curandStatus_t)arg.cmd;
}

extern "C" curandStatus_t CURANDAPI 
curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num)
{
    VirtIOArg arg;
    func();
    if(!generator)
        return CURAND_STATUS_SUCCESS;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CURAND_GENERATEUNIFORMDOUBLE;
    arg.src     = (uint64_t)generator;
    arg.dst     = (uint64_t)outputPtr;
    arg.dstSize = (uint32_t)(num*sizeof(double));
    arg.param   = (uint64_t)num;
    send_to_device(VIRTIO_IOC_CURAND_GENERATEUNIFORMDOUBLE, &arg);
    return (curandStatus_t)arg.cmd;
}

extern "C" curandStatus_t CURANDAPI 
curandDestroyGenerator(curandGenerator_t generator)
{
    VirtIOArg arg;
    func();
    if(!generator)
        return CURAND_STATUS_SUCCESS;
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CURAND_DESTROYGENERATOR;
    arg.src     = (uint64_t)generator;
    send_to_device(VIRTIO_IOC_CURAND_DESTROYGENERATOR, &arg);
    return (curandStatus_t)arg.cmd;
}

typedef struct generator_buf
{
    curandGenerator_t generator;
    unsigned long long seed;
    unsigned long long offset;
} generator_buf_t;
static generator_buf_t p_last_generator;
static generator_buf_t p_generator;

extern "C" curandStatus_t CURANDAPI 
curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    // debug("sizeof(curandGenerator_t) = %lx\n", sizeof(curandGenerator_t));
    arg.cmd     = VIRTIO_CURAND_CREATEGENERATOR;
    arg.dst     = (uint64_t)rng_type;
    send_to_device(VIRTIO_IOC_CURAND_CREATEGENERATOR, &arg);
    *generator  = (curandGenerator_t)arg.flag;
    return (curandStatus_t)arg.cmd;
}

extern "C" curandStatus_t CURANDAPI 
curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed)
{
    // VirtIOArg arg;
    func();
    if(!generator)
        return CURAND_STATUS_SUCCESS;
    debug("generator %lx , seed = %llx\n", (uint64_t)generator, seed);
    // memset(&arg, 0, sizeof(VirtIOArg));
    // arg.cmd     = VIRTIO_CURAND_SETPSEUDORANDOMSEED;
    // arg.src     = (uint64_t)generator;
    // arg.param   = (uint64_t)seed;
    // send_to_device(VIRTIO_IOC_CURAND_SETPSEUDORANDOMSEED, &arg);
    // return (curandStatus_t)arg.cmd;
    p_generator.generator  = generator;
    p_generator.seed       = seed;
    return CURAND_STATUS_SUCCESS;
}

extern "C" curandStatus_t CURANDAPI 
curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset)
{
    VirtIOArg arg;
    func();
    if(!generator)
        return CURAND_STATUS_SUCCESS;
    debug("generator %lx , offset = %llx\n", (uint64_t)generator, offset);
    p_generator.offset = offset;
    if(p_last_generator.generator == generator && 
        p_last_generator.seed == p_generator.seed &&
        p_last_generator.offset == offset 
        ) {
        debug("stay same!\n");
        return CURAND_STATUS_SUCCESS;
    }
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_CURAND_SETGENERATOROFFSET;
    arg.src     = (uint64_t)generator;
    arg.param   = (uint64_t)offset;
    arg.param2   = (uint64_t)p_generator.seed;
    memcpy(&p_last_generator, &p_generator, sizeof(generator_buf_t));
    send_to_device(VIRTIO_IOC_CURAND_SETGENERATOROFFSET, &arg);
    return (curandStatus_t)arg.cmd;
}
//**************************SGX**********************************************//
//***************************************************************************//

void PRINT_BYTE_ARRAY(
    FILE *file, void *mem, uint32_t len)
{
    if(!mem || !len)
    {
        fprintf(file, "\n( null )\n");
        return;
    }
    uint8_t *array = (uint8_t *)mem;
    fprintf(file, "%u bytes:\n{\n", len);
    uint32_t i = 0;
    for(i = 0; i < len - 1; i++)
    {
        fprintf(file, "0x%x, ", array[i]);
        if(i % 8 == 7) fprintf(file, "\n");
    }
    fprintf(file, "0x%x ", array[i]);
    fprintf(file, "\n}\n");
}

extern "C" void ocall_print(const char *str)
{
    printf("OCALL: %s\n", str);
}

int ra_send0_receive(uint32_t extended_epid_group_id, uint8_t *p_resp, 
                        uint32_t resp_size,  uint32_t *resp_body_size)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd         = VIRTIO_SGX_MSG0;
    arg.flag        = 0;
    arg.srcSize     = (uint32_t)extended_epid_group_id;
    arg.dst         = (uint64_t)p_resp;
    arg.dstSize     = (uint32_t)resp_size;
    arg.tid         = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_SGX_MSG0, &arg);
    *resp_body_size = arg.paramSize;
    return arg.cmd;
}

int ra_send1_receive(sgx_ra_msg1_t *p_req, uint8_t *p_resp, 
                        uint32_t resp_size, uint32_t *resp_body_size)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd         = VIRTIO_SGX_MSG1;
    arg.flag        = 1;
    arg.src         = (uint64_t)p_req;
    arg.srcSize     = (uint32_t)sizeof(sgx_ra_msg1_t);
    arg.dst         = (uint64_t)p_resp;
    arg.dstSize     = (uint32_t)resp_size;
    arg.tid         = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_SGX_MSG1, &arg);
    *resp_body_size = arg.paramSize;
    return arg.cmd;
}

int ra_send3_receive(sgx_ra_msg3_t *p_req, uint32_t req_size, 
                        uint8_t *p_resp, uint32_t resp_size, 
                        uint32_t *resp_body_size)
{
    VirtIOArg arg;
    func();
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd         = VIRTIO_SGX_MSG3;
    arg.flag        = 3;
    arg.src         = (uint64_t)p_req;
    arg.srcSize     = (uint32_t)req_size;
    arg.dst         = (uint64_t)p_resp;
    arg.dstSize     = (uint32_t)resp_size;
    arg.tid         = (uint32_t)syscall(SYS_gettid);
    send_to_device(VIRTIO_IOC_SGX_MSG3, &arg);
    *resp_body_size = arg.paramSize;
    return arg.cmd;
}

int init_sgx_ecdh(SGX_RA_ENV *sgx_ctx)
{
    int ret = 0;
    uint32_t resp_size = 0x200;
    uint8_t *p_msg_resp_body = NULL;
    uint32_t resp_body_size;
    sgx_ra_msg1_t *p_msg1;
    sgx_ra_msg3_t *p_msg3 = NULL;
    sgx_enclave_id_t enclave_id = 0;
    int enclave_lost_retry_time = 1;
    int busy_retry_time = 4;
    sgx_ra_context_t context = INT_MAX;
    sgx_status_t status = SGX_SUCCESS;
    sgx_att_key_id_t selected_key_id = {0};
    // FILE* OUTPUT = stdout;   

        debug("\nWe will try EPID algorithm.\n");
        // Preparation for remote attestation by configuring extended epid group id.
        {
            uint32_t extended_epid_group_id = 0;
            ret = sgx_get_extended_epid_group_id(&extended_epid_group_id);
            if (SGX_SUCCESS != ret)
            {
                ret = -1;
                error("Error, call sgx_get_extended_epid_group_id fail.\n");
                return ret;
            }
            debug("Call sgx_get_extended_epid_group_id success.\n");
            // The ISV application sends msg0 to the SP.
            // The ISV decides whether to support this extended epid group id.
            debug("\nSending msg0 to remote attestation service provider.\n");
            p_msg_resp_body = (uint8_t *)malloc(resp_size);
            ret = ra_send0_receive(extended_epid_group_id, p_msg_resp_body, 
                                    resp_size, &resp_body_size);
            if (ret != 0)
            {
                error("\nError, ra_network_send_receive for msg0 failed\n");
                goto CLEANUP;
            }
            debug("\nSent MSG0 to remote attestation service.\n");
            
            debug("\nQEMU MSG0 resp size 0x%x.\n", resp_body_size);
            ret = sgx_select_att_key_id(p_msg_resp_body, resp_body_size, &selected_key_id);
            if(SGX_SUCCESS != ret)
            {
                ret = -1;
                debug("\nInfo, call sgx_select_att_key_id fail, "
                        "current platform configuration doesn't support "
                        "this attestation key ID.");
                goto CLEANUP;
            }
            debug("\nCall sgx_select_att_key_id success.");
        }
        // Remote attestation will be initiated the ISV server challenges the ISV
        // app or if the ISV app detects it doesn't have the credentials
        // (shared secret) from a previous attestation required for secure
        // communication with the server.
        {
            // ISV application creates the ISV enclave.
            do
            {
            #ifndef SGX_SWITCHLESS
                debug("\nCreate regular enclave.");
                ret = sgx_create_enclave(enclave_path,
                                         SGX_DEBUG_FLAG,
                                         NULL,
                                         NULL,
                                         &enclave_id, NULL);
            #else
                const void* enclave_ex_p[32] = { 0 };
                sgx_uswitchless_config_t us_config = SGX_USWITCHLESS_CONFIG_INITIALIZER;
                us_config.num_uworkers = 2;
                us_config.num_tworkers = 2;
                enclave_ex_p[SGX_CREATE_ENCLAVE_EX_SWITCHLESS_BIT_IDX] = &us_config;
                debug("\nCreate switchless enclave.");
                ret = sgx_create_enclave_ex(enclave_path,
                                         SGX_DEBUG_FLAG,
                                         NULL,
                                         NULL,
                                         &enclave_id, NULL,
                                         SGX_CREATE_ENCLAVE_EX_SWITCHLESS, enclave_ex_p);
            #endif
                if(SGX_SUCCESS != ret)
                {
                    ret = -1;
                    error("\nError, call sgx_create_enclave fail.\n");
                    goto CLEANUP;
                }
                debug("\nCall sgx_create_enclave success. %lx\n", enclave_id);

                ret = enclave_init_ra(enclave_id,
                                      &status,
                                      false,
                                      &context);
            //Ideally, this check would be around the full attestation flow.
            } while (SGX_ERROR_ENCLAVE_LOST == ret && enclave_lost_retry_time--);

            if(SGX_SUCCESS != ret || status)
            {
                ret = -1;
                error("\nError, call enclave_init_ra fail.\n");
                goto CLEANUP;
            }
            debug("\nCall enclave_init_ra success.");

            // isv application call uke sgx_ra_get_msg1
            p_msg1 = (sgx_ra_msg1_t *)malloc(sizeof(sgx_ra_msg1_t));
            if(NULL == p_msg1)
            {
                ret = -1;
                goto CLEANUP;
            }
            do
            {
                ret = sgx_ra_get_msg1_ex(&selected_key_id, context, enclave_id, sgx_ra_get_ga, p_msg1);
                if (SGX_ERROR_BUSY == ret) {
                    debug("SGX BUSY %ld, TRY again\n", syscall(SYS_gettid));
                }
            } while (SGX_ERROR_BUSY == ret && busy_retry_time--);
            if(SGX_SUCCESS != ret)
            {
                ret = -1;
                error("\nError, call sgx_ra_get_msg1_ex fail.\n");
                goto CLEANUP;
            }
            else
            {
                debug("\nCall sgx_ra_get_msg1_ex success.\n");
                debug("\nMSG1 body generated -\n");
                // PRINT_BYTE_ARRAY(OUTPUT, p_msg1, sizeof(sgx_ra_msg1_t));
            }


            // The ISV application sends msg1 to the SP to get msg2,
            // msg2 needs to be freed when no longer needed.
            // The ISV decides whether to use linkable or unlinkable signatures.
            debug("\nSending msg1 to remote attestation service provider."
                            "Expecting msg2 back.\n");

            memset(p_msg_resp_body, 0, resp_size);
            ret = ra_send1_receive(p_msg1, p_msg_resp_body, 
                                resp_size, &resp_body_size);

            if(ret != 0 || !p_msg_resp_body)
            {
                error("\nError, ra_network_send_receive for msg1 failed\n");
                goto CLEANUP;
            }
            else
            {
                // Successfully sent msg1 and received a msg2 back.
                // Time now to check msg2.
                debug("\nSent MSG1 to remote attestation service "
                                "provider. Received the following MSG2:\n");
                // PRINT_BYTE_ARRAY(OUTPUT, p_msg_resp_body, resp_body_size);
            }
            debug("\nMSG2 resp size %x.\n", resp_body_size);

            sgx_ra_msg2_t* p_msg2_body = (sgx_ra_msg2_t*)p_msg_resp_body;

            uint32_t msg3_size = 0;
            busy_retry_time = 2;
            // The ISV app now calls uKE sgx_ra_proc_msg2,
            // The ISV app is responsible for freeing the returned p_msg3!!
            do
            {
                ret = sgx_ra_proc_msg2_ex(&selected_key_id,
                                   context,
                                   enclave_id,
                                   sgx_ra_proc_msg2_trusted,
                                   sgx_ra_get_msg3_trusted,
                                   p_msg2_body,
                                   resp_body_size,
                                   &p_msg3,
                                   &msg3_size);
                if (SGX_ERROR_BUSY == ret) {
                    debug("SGX BUSY %ld, TRY again\n", syscall(SYS_gettid));
                }
            } while (SGX_ERROR_BUSY == ret && busy_retry_time--);
            if(!p_msg3)
            {
                error("\nError, call sgx_ra_proc_msg2_ex fail. "
                                "p_msg3 = 0x%p.", p_msg3);
                ret = -1;
                goto CLEANUP;
            }
            if(SGX_SUCCESS != (sgx_status_t)ret)
            {
                error("\nError, call sgx_ra_proc_msg2_ex fail. "
                                "ret = 0x%08x .\n", ret);
                ret = -1;
                goto CLEANUP;
            }
            else
            {
                debug("\nCall sgx_ra_proc_msg2_ex success.\n");
                debug("\nMSG3 - \n");
            }

            // PRINT_BYTE_ARRAY(OUTPUT, p_msg3, msg3_size);
            debug("\n msg3_size = 0x%x.\n", msg3_size);
            debug("\n sizeof(sgx_ra_msg3_t) = 0x%lx.\n", sizeof(sgx_ra_msg3_t));
            
            // The ISV application sends msg3 to the SP to get the attestation
            // result message, attestation result message needs to be freed when
            // no longer needed. The ISV service provider decides whether to use
            // linkable or unlinkable signatures. The format of the attestation
            // result is up to the service provider. This format is used for
            // demonstration.  Note that the attestation result message makes use
            // of both the MK for the MAC and the SK for the secret. These keys are
            // established from the SIGMA secure channel binding.
            
            memset(p_msg_resp_body, 0, resp_size);
            ret = ra_send3_receive(p_msg3, msg3_size, p_msg_resp_body, 
                                resp_size, &resp_body_size);
            if(ret || !p_msg_resp_body)
            {
                ret = -1;
                error("\nError, sending msg3 failed.\n");
                goto CLEANUP;
            }

            debug("\nMSG4 resp size %x.\n", resp_body_size);
            // sample_ra_att_result_msg_t * p_att_result_msg_body =
            //     (sample_ra_att_result_msg_t *)((uint8_t*)p_att_result_msg_full
            //                                    + sizeof(ra_samp_response_header_t));
            sample_ra_att_result_msg_t * p_att_result_msg_body =
                        (sample_ra_att_result_msg_t *)p_msg_resp_body;

            debug("\nSent MSG3 successfully. Received an attestation "
                                "result message back\n.");

            debug("\nATTESTATION RESULT RECEIVED - ");
            // PRINT_BYTE_ARRAY(OUTPUT, p_att_result_msg_body, resp_body_size);

            // Check the MAC using MK on the attestation result message.
            // The format of the attestation result message is ISV specific.
            // This is a simple form for demonstration. In a real product,
            // the ISV may want to communicate more information.
            ret = verify_att_result_mac(enclave_id,
                    &status,
                    context,
                    (uint8_t*)&p_att_result_msg_body->platform_info_blob,
                    sizeof(ias_platform_info_blob_t),
                    (uint8_t*)&p_att_result_msg_body->mac,
                    sizeof(sgx_mac_t));
            if((SGX_SUCCESS != ret) ||
               (SGX_SUCCESS != status))
            {
                ret = -1;
                error("\nError: INTEGRITY FAILED - attestation result "
                                "message MK based cmac failed.\n");
                goto CLEANUP;
            }

            // The attestation result message should contain a field for the Platform
            // Info Blob (PIB).  The PIB is returned by attestation server in the attestation report.
            // It is not returned in all cases, but when it is, the ISV app
            // should pass it to the blob analysis API called sgx_report_attestation_status()
            // along with the trust decision from the ISV server.
            // The ISV application will take action based on the update_info.
            // returned in update_info by the API.
            // This call is stubbed out for the sample.
            //
            // sgx_update_info_bit_t update_info;
            // ret = sgx_report_attestation_status(
            //     &p_att_result_msg_body->platform_info_blob,
            //     attestation_passed ? 0 : 1, &update_info);

            // Get the shared secret sent by the server using SK (if attestation
            // passed)
            /*ret = enclave_key_out(enclave_id, &status, context, buf, sizeof(sgx_ec_key_128bit_t));
            if((SGX_SUCCESS != ret)  || (SGX_SUCCESS != status))
            {
                error("\nError, can not get secret "
                                "failed in [%s]. ret = "
                                "0x%0x. status = 0x%0x", __FUNCTION__, ret,
                                 status);
                goto CLEANUP;
            }
            debug("\n Get sk key\n");
            for(k=0; k<sizeof(sgx_ec_key_128bit_t); k++) {
                fprintf(OUTPUT, "\t%x", buf[k]);
            }
            debug_clean("\n");
            debug("payload size %x\n", p_att_result_msg_body->secret.payload_size);
            debug("payload\n");
            for (k=0; k<p_att_result_msg_body->secret.payload_size; k++) {
                debug_clean("%x ", p_att_result_msg_body->secret.payload[k]);
            }
            debug_clean("\n");
            debug("payload tag\n");
            for (k=0; k<SAMPLE_SP_TAG_SIZE; k++) {
                debug_clean("%x ", p_att_result_msg_body->secret.payload_tag[k]);
            }
            debug_clean("\n\n");*/
            ret = put_secret_data(enclave_id,
                                  &status,
                                  context,
                                  p_att_result_msg_body->secret.payload,
                                  p_att_result_msg_body->secret.payload_size,
                                  p_att_result_msg_body->secret.payload_tag);
            if((SGX_SUCCESS != ret)  || (SGX_SUCCESS != status))
            {
                error("\nError, attestation result message secret "
                                "using SK based AESGCM failed in [%s]. ret = "
                                "0x%0x. status = 0x%0x\n", __FUNCTION__, ret,
                                 status);
                goto CLEANUP;
            }
            debug("Secret successfully received from server.\n");
            debug("Remote attestation success!\n");
            
        }
        sgx_ctx->context    = context;
        sgx_ctx->enclave_id = enclave_id;
        return 0;

    CLEANUP:
        // Clean-up
        // Need to close the RA key state.
        if(INT_MAX != context)
        {
            int ret_save = ret;
            ret = enclave_ra_close(enclave_id, &status, context);
            if(SGX_SUCCESS != ret || status)
            {
                ret = -1;
                error("\nError, call enclave_ra_close fail.\n");
            }
            else
            {
                // enclave_ra_close was successful, let's restore the value that
                // led us to this point in the code.
                ret = ret_save;
            }
            debug("\nCall enclave_ra_close success.");
        }

        sgx_destroy_enclave(enclave_id);

        // p_msg3 is malloc'd by the untrusted KE library. App needs to free.
        free(p_msg_resp_body);
        SAFE_FREE(p_msg3);

    return 0;
}

static int fini_sgx_ecdh(SGX_RA_ENV sgx_ctx)
{
    int ret = 0;
    sgx_status_t status = SGX_SUCCESS;
    // Need to close the RA key state.
    if(INT_MAX != sgx_ctx.context)
    {
        int ret_save = ret;
        ret = enclave_ra_close(sgx_ctx.enclave_id, &status, sgx_ctx.context);
        if(SGX_SUCCESS != ret || status)
        {
            ret = -1;
            error("Error, call enclave_ra_close fail.\n");
        }
        else
        {
            // enclave_ra_close was successful, let's restore the value that
            // led us to this point in the code.
            ret = ret_save;
        }
        debug("Call enclave_ra_close success.\n");
    }

    sgx_destroy_enclave(sgx_ctx.enclave_id);
    return ret;
}


extern "C" cudaError_t cudaMemcpySafe(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    VirtIOArg arg;
    uint8_t payload_tag[SAMPLE_SP_TAG_SIZE];
    uint8_t *data;

    func();
    if (kind <0 || kind >4) {
        return cudaErrorInvalidMemcpyDirection;
    }
    if(count<=0 || !dst || !src )
        return cudaSuccess;
    if (kind == cudaMemcpyHostToHost) {
        memcpy(dst, src, count);
        return cudaSuccess;
    } 
    memset(&arg, 0, sizeof(VirtIOArg));
    arg.cmd     = VIRTIO_SGX_MEMCPY;
    arg.flag    = kind;
    arg.src     = (uint64_t)src;
    arg.srcSize = (uint32_t)count;
    arg.dst     = (uint64_t)dst;
    arg.dstSize = (uint32_t)count;
    arg.tid     = (uint32_t)syscall(SYS_gettid);
    debug("gettid %d\n", arg.tid);
    if (kind == cudaMemcpyHostToDevice) {
        data = (uint8_t *)my_malloc(count);
        get_encrypted_data((uint8_t *)src, count, data, payload_tag);
        memcpy(arg.mac, payload_tag, SAMPLE_SP_TAG_SIZE);
        arg.src = (uint64_t)data;
    }
    if (kind == cudaMemcpyDeviceToHost) {
        data        = (uint8_t *)my_malloc(count);
        arg.dst     = (uint64_t)data;
    }
    send_to_device(VIRTIO_IOC_SGX_MEMCPY, &arg);
    if (kind == cudaMemcpyDeviceToHost) {
        get_decrypted_data((uint8_t *)arg.dst, count, (uint8_t *)dst, arg.mac);
    }
    my_free(data);
    data = NULL;
    return (cudaError_t)arg.cmd;    
}