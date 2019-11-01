# virtio-cuda-module
This is the para-virtualized front driver of cuda-supported qemu and test case.   

The user runtime wrappered library in VM (guest OS) provides CUDA runtime access, interfaces of memory allocation, CUDA commands, and passes cmds to the driver.

The front-end driver is responsible for the memory management, transferring data, analyzing the ioctl cmds from the customed library, and passing the cmds by the control channel.


The Intel SGX is emulated in [qemu](https://github.com/intel/qemu-sgx).  
We add sgx-support in our virtio-based vCUDA which is developed in branch *cuda-sgx*. The corresponding QEMU can be found [here](https://github.com/juniorprincewang/qemu-sgx-cuda).  

## Installation

### Prerequisites

The our experiment environment is as follows:  

#### Host

* Ubuntu 16.04.5 LTS (kernel v4.15.0-29-generic  x86_64)
* cuda-9.1
* PATH

```sh
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64' >> ~/.bashrc
source ~/.bashrc

sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig
```
* Install required packages
  
``` sh
sudo apt-get install -y  pkg-config bridge-utils uml-utilities zlib1g-dev libglib2.0-dev autoconf
automake libtool libsdl1.2-dev libsasl2-dev libcurl4-openssl-dev libsasl2-dev libaio-dev libvde-dev libspice-server-dev
```

#### Guest

* Ubuntu 16.04 x86_64 image (guest OS)
* cuda-9.1 toolkit

### How to install

#### Host

* [Our QEMU](https://github.com/juniorprincewang/qemu) was modified from QEMU 2.12.0, for further information please refer to [QEMU installation steps](https://en.wikibooks.org/wiki/QEMU/Installing_QEMU)


#### Guest

1. clone this repo.  
[to do]


## A CUDA sample in guest OS 

In the guest OS, *nvcc* compiles the source with host/device code and 
standard CUDA runtime APIs. To compare with a native OS, in the 
guest VM, compiling the CUDA program must add the nvcc flag 
"**-cudart=shared**", which can be dynamically linked to the userspace 
library as a shared library.   
Therefore, the wrappered library provided functions that intercepted 
dynamic memory allocation of CPU code and CUDA runtime APIs.  
After installing qCUdriver and qCUlibrary in the guest OS, modify the 
internal flags in the Makefile as below:  
```shell
# internal flags
NVCCFLAGS   := -m${TARGET_SIZE} --cudart=shared      
```

Finally, run make and perform the executable file without change any 
source code by `LD_PRELOAD` or change the `LD_LIBRARY_PATH`.   

```sh
LD_PRELOAD=\path\to\libvcuda.so ./vectorAdd
```

+ benchmarking vectorAdd

A command-line benchmarking tool [hyperfine](https://github.com/sharkdp/hyperfine) is recommended.  
To run a benchmark, you can simply call `hyperfine <command>....` , for example.  

```
hyperfine 'LD_PRELOAD=\path\to\libvcuda.so ./vectorAdd'
```

By default, Hyperfine will perform at least 10 benchmarking runs. To change this, you can use the *-m/--min-runs* option or *-M/--max-runs*.


# supported API 

## CUDA Runtime API

In our current version, we implement necessary CUDA runtime APIs. These CUDA 
runtime API are shown as below:  

<table class="tg">
  <tr>
    <th class="tg-yw4l">Classification</th>
    <th class="tg-yw4l">supported CUDA runtime API</th>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="8">Memory Management</td>
    <td class="tg-3we0">cudaMalloc</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaMemset</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaMemcpy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaMemcpyAsync</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaFree</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaMemGetInfo</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaMemcpyToSymbol</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaMemcpyFromSymbol</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="7">Device Management</td>
    <td class="tg-3we0">cudaGetDevice</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaGetDeviceCount</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaSetDevice</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaSetDeviceFlags</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaGetDeviceProperties</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaDeviceSynchronize</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaDeviceReset</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="5">Stream Management</td>
    <td class="tg-3we0">cudaStreamCreate</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaStreamCreateWithFlags</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaStreamDestroy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaStreamSynchronize</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaStreamWaitEvent</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="7">Event Management</td>
    <td class="tg-3we0">cudaEventCreate</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventCreateWithFlags</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventRecord</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventSynchronize</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventElapsedTime</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventDestroy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaEventQuery</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="2">Error Handling</td>
    <td class="tg-3we0">cudaGetLastError</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaGetErrorString</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="7">Zero-copy</td>
    <td class="tg-3we0">cudaHostRegister</td>
  </tr>
  <tr>
    <td class="tg-3we0">~~cudaHostGetDevicePointer~~</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaHostUnregister</td>
  </tr>
    <tr>
    <td class="tg-3we0">cudaHostAlloc</td>
  </tr>
    <tr>
    <td class="tg-3we0">cudaMallocHost</td>
  </tr>
  </tr>
    <tr>
    <td class="tg-3we0">cudaFreeHost</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaSetDeviceFlags</td>
  </tr>
  <tr>
    <td class="tg-yw4l">Thread Management</td>
    <td class="tg-3we0">cudaThreadSynchronize</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="7">Module &amp; Execution Control</td>
    <td class="tg-3we0">__cudaRegisterFatBinary</td>
  </tr>
  <tr>
    <td class="tg-3we0">__cudaUnregisterFatBinary</td>
  </tr>
  <tr>
    <td class="tg-3we0">__cudaRegisterFunction</td>
  </tr>
  <tr>
    <td class="tg-3we0">__cudaRegisterVar</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaConfigureCall</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaSetupArgument</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaLaunch</td>
  </tr>
</table>

## CUBLAS API & CURAND API

To support [Caffe](https://github.com/BVLC/caffe.git), we implement CUBLAS & CURAND API in *libcudart.so*.  

<table class="tg">
  <tr>
    <th class="tg-yw4l">Classification</th>
    <th class="tg-yw4l">supported API</th>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="22">CUBLAS API</td>
    <td class="tg-3we0">cublasCreate</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasDestroy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasSetVector</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasGetVector</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasSetStream</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasGetStream</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasSasum</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasDasum</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasScopy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasDcopy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasSdot</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasDdot</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasSaxpy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasDaxpy</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasSscal</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasDscal</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasSgemv</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasDgemv</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasSgemm</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasDgemm</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasSetMatrix</td>
  </tr>
  <tr>
    <td class="tg-3we0">cublasGetMatrix</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="10">CURAND API</td>
    <td class="tg-3we0">curandCreateGenerator</td>
  </tr>
  <tr>
    <td class="tg-3we0">curandCreateGeneratorHost</td>
  </tr>
  <tr>
    <td class="tg-3we0">curandGenerate</td>
  </tr>
  <tr>
    <td class="tg-3we0">curandGenerateUniform</td>
  </tr>
  <tr>
    <td class="tg-3we0">curandGenerateUniformDouble</td>
  </tr>
  <tr>
    <td class="tg-3we0">curandGenerateNormal</td>
  </tr>
  <tr>
    <td class="tg-3we0">curandGenerateNormalDouble</td>
  </tr>
  <tr>
    <td class="tg-3we0">curandDestroyGenerator</td>
  </tr>
  <tr>
    <td class="tg-3we0">curandSetGeneratorOffset</td>
  </tr>
  <tr>
    <td class="tg-3we0">curandSetPseudoRandomGeneratorSeed</td>
  </tr>
</table>


# Supported Software

+ part of NVIDIA_CUDA-9.1_Samples  
+ [Rodinia benchmark ](https://github.com/yuhc/gpu-rodinia.git)
+ [Caffe: a fast open framework for deep learning.](https://github.com/BVLC/caffe.git)




Last but not least, thanks [qcuda](https://github.com/coldfunction/qCUDA) for inspiring.  
Also, what we use for message channels is [chan: Pure C implementation of Go channels. ](https://github.com/tylertreat/chan)
