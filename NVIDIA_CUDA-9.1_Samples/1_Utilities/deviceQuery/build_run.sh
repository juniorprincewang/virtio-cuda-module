nvcc --cudart=shared -I../../common/inc deviceQuery.cu -o device -L /usr/local/vcuda/lib64/ -lcudart
LD_PRELOAD=/usr/local/vcuda/lib64/libcudart.so ./device
