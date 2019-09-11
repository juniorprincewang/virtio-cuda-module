# virtio-cuda-module
This is the para-virtualized front driver of cuda-supported qemu and test case. 

The user runtime wrappered library in VM (guest OS) provides CUDA runtime access, interfaces of memory allocation, CUDA commands, and passes cmds to the driver.

The front-end driver is responsible for the memory management, data movement, analyzing the ioctl cmds from the customed library, and passing the cmds by the control channel.


## Installation

### Prerequisites

#### Host

* Ubuntu 16.04.5 LTS (kernel v4.15.0-29-generic  )
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

* Ubuntu 16.04 image (guest OS)
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

By default, Hyperfine will perform at least 10 benchmarking runs. To change this, you can use the *-m/--min-runs* option.


# supported CUDA runtime API 

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
    <td class="tg-yw4l" rowspan="6">Device Management</td>
    <td class="tg-3we0">cudaGetDevice</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaGetDeviceCount</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaSetDevice</td>
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
    <td class="tg-yw4l" rowspan="2">Stream Management</td>
    <td class="tg-3we0">cudaStreamCreate</td>
  </tr>
  <tr>
    <td class="tg-3we0">cudaStreamDestroy</td>
  </tr>
  <tr>
    <td class="tg-yw4l" rowspan="6">Event Management</td>
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
    <td class="tg-yw4l">Error Handling</td>
    <td class="tg-3we0">cudaGetLastError</td>
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


Last but not least, thanks [qcuda](https://github.com/coldfunction/qCUDA) for inspiring.
