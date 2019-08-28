################################################################################
#
# Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
#  findegl.mk is used to find the necessary EGL Libraries for specific distributions
#               this is supported on Linux
#
################################################################################

# Determine OS platform and unix distribution
ifeq ("$(TARGET_OS)","linux")
   # first search lsb_release
   DISTRO  = $(shell lsb_release -i -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
   ifeq ("$(DISTRO)","")
     # second search and parse /etc/issue
     DISTRO = $(shell more /etc/issue | awk '{print $$1}' | sed '1!d' | sed -e "/^$$/d" 2>/dev/null | tr "[:upper:]" "[:lower:]")
     # ensure data from /etc/issue is valid
     ifeq (,$(filter $(DISTRO),ubuntu fedora red rhel centos suse))
       DISTRO = 
     endif
     ifeq ("$(DISTRO)","")
       # third, we can search in /etc/os-release or /etc/{distro}-release
       DISTRO = $(shell awk '/ID/' /etc/*-release | sed 's/ID=//' | grep -v "VERSION" | grep -v "ID" | grep -v "DISTRIB")
     endif
   endif
endif

ifeq ("$(TARGET_OS)","linux")
    # $(info) >> findegl.mk -> LINUX path <<<)
    # Each set of Linux Distros have different paths for where to find their OpenGL libraries reside
    UBUNTU = $(shell echo $(DISTRO) | grep -i ubuntu      >/dev/null 2>&1; echo $$?)
    FEDORA = $(shell echo $(DISTRO) | grep -i fedora      >/dev/null 2>&1; echo $$?)
    RHEL   = $(shell echo $(DISTRO) | grep -i 'red\|rhel' >/dev/null 2>&1; echo $$?)
    CENTOS = $(shell echo $(DISTRO) | grep -i centos      >/dev/null 2>&1; echo $$?)
    SUSE   = $(shell echo $(DISTRO) | grep -i suse        >/dev/null 2>&1; echo $$?)
    ifeq ("$(UBUNTU)","0")
      ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        GLPATH := /usr/arm-linux-gnueabihf/lib
        GLLINK := -L/usr/arm-linux-gnueabihf/lib
        ifneq ($(TARGET_FS),) 
          GLPATH += $(TARGET_FS)/usr/lib/arm-linux-gnueabihf
          GLLINK += -L$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif 
      else
        UBUNTU_PKG_NAME = $(shell which dpkg >/dev/null 2>&1 && dpkg -l 'nvidia-*' | grep '^ii' | awk '{print $$2}' | head -1)
        ifneq ("$(UBUNTU_PKG_NAME)","")
          GLPATH    ?= /usr/lib/$(UBUNTU_PKG_NAME)
          GLLINK    ?= -L/usr/lib/$(UBUNTU_PKG_NAME)
        endif

        DFLT_PATH ?= /usr/lib
      endif
    endif
    ifeq ("$(SUSE)","0")
      GLPATH    ?= /usr/X11R6/lib64
      GLLINK    ?= -L/usr/X11R6/lib64
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(FEDORA)","0")
      GLPATH    ?= /usr/lib64/nvidia
      GLLINK    ?= -L/usr/lib64/nvidia
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(RHEL)","0")
      GLPATH    ?= /usr/lib64/nvidia
      GLLINK    ?= -L/usr/lib64/nvidia
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(CENTOS)","0")
      GLPATH    ?= /usr/lib64/nvidia
      GLLINK    ?= -L/usr/lib64/nvidia
      DFLT_PATH ?= /usr/lib64
    endif

  EGLLIB  := $(shell find -L $(GLPATH) $(DFLT_PATH) -name libEGL.so    -print 2>/dev/null)

  ifeq ("$(EGLLIB)","")
      $(info >>> WARNING - libEGL.so not found, please install libEGL.so <<<)
      SAMPLE_ENABLED := 0
  endif

  HEADER_SEARCH_PATH ?= $(TARGET_FS)/usr/include
  ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
      HEADER_SEARCH_PATH += /usr/arm-linux-gnueabihf/include
  endif

  EGLHEADER  := $(shell find -L $(HEADER_SEARCH_PATH) -name egl.h -print 2>/dev/null)
  EGLEXTHEADER  := $(shell find -L $(HEADER_SEARCH_PATH) -name eglext.h -print 2>/dev/null)

  ifeq ("$(EGLHEADER)","")
      $(info >>> WARNING - egl.h not found, please install egl.h <<<)
      SAMPLE_ENABLED := 0
  endif
  ifeq ("$(EGLEXTHEADER)","")
      $(info >>> WARNING - eglext.h not found, please install eglext.h <<<)
      SAMPLE_ENABLED := 0
  endif
else
endif

# Attempt to compile a minimal EGL application and run to check if EGL_SUPPORT_REUSE_NV is supported in the EGL headers available.
ifneq ($(SAMPLE_ENABLED), 0)
      $(shell printf "#include <EGL/egl.h>\n#include <EGL/eglext.h>\nint main() {\n#ifdef EGL_SUPPORT_REUSE_NV \n #pragma message(\"Compatible EGL header found\") \n  return 0;\n#endif \n return 1;\n}"  > test.c; )
      EGL_DEFINES := $(shell $(HOST_COMPILER) $(CCFLAGS) $(EXTRA_CCFLAGS) $(LDFLAGS) $(EXTRA_LDFLAGS) -lEGL test.c 2>&1 | grep -ic "Compatible EGL header found";)
      EXE_EXISTS := $(shell find a.out 2>/dev/null)
      SHOULD_WAIVE := 0
      ifeq ($(EGL_DEFINES),0)
        SHOULD_WAIVE := 1
      endif
      ifeq (,$(EXE_EXISTS))
        SHOULD_WAIVE := 1
      endif
      ifeq ($(SHOULD_WAIVE),1)
          $(info -----------------------------------------------------------------------------------------------)
          $(info WARNING - NVIDIA EGL EXTENSIONS are not available in the present EGL headers)
          $(info -----------------------------------------------------------------------------------------------)
          $(info   This CUDA Sample cannot be built if the EGL NVIDIA EXTENSIONS like EGL_SUPPORT_REUSE_NV are not supported in EGL headers.)
          $(info   This will be a dry-run of the Makefile.)
          $(info   Please install the latest khronos EGL headers and libs to build this sample)
          $(info -----------------------------------------------------------------------------------------------)
          SAMPLE_ENABLED := 0
      endif
      $(shell rm a.out test.c 2>/dev/null)
endif

