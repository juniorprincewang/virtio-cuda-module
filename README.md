# virtio-cuda-module
This is the para-virtualized front driver of cuda-supported qemu and test case. 

The user runtime wrappered library in VM (guest OS) provides CUDA runtime access, interfaces of memory allocation, CUDA commands, and passes cmds to the driver.

The front-end driver is responsible for the memory management, data movement, analyzing the ioctl cmds from the customed library, and passing the cmds by the control channel.


## Installation

### Prerequisites

#### Host

* Ubuntu 16.04.5 LTS (kernel v4.15.0-29-generic  )
* cuda-8.0
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
* cuda-8.0 toolkit

### How to install

#### Host

* [Our QEMU](https://github.com/juniorprincewang/qemu) was modified from QEMU 2.12.0, for further information please refer to [QEMU installation steps](https://en.wikibooks.org/wiki/QEMU/Installing_QEMU)


#### Guest

1. clone this repo.  
[to do]


## A CUDA sample in guest OS 

In the guest OS, *nvcc* compiles the source with host/device code and standard CUDA runtime APIs. To compare with a native OS, in the guest VM, compiling the CUDA program must add the nvcc flag "**-cudart=shared**", which can be dynamically linked to the userspace library as a shared library.   
Therefore, the wrappered library provided functions that intercepted dynamic memory allocation of CPU code and CUDA runtime APIs.  
After installing qCUdriver and qCUlibrary in the guest OS, modify the internal flags in the Makefile as below:  
```shell
# internal flags
NVCCFLAGS   := -m${TARGET_SIZE} -cudart=shared      
```
Finally, run make and perform the executable file without change any source code.


# add the new CUDA runtime API

To do

Thanks [qcuda](https://github.com/coldfunction/qCUDA) for inspiring.
