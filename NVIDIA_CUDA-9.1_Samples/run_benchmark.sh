#!/bin/bash

dir_array=(	
	./0_Simple/vectorAdd/
	./0_Simple/simpleStreams/
	./0_Simple/matrixMul/
	./3_Imaging/convolutionSeparable/
	./4_Finance/binomialOptions/
	./4_Finance/BlackScholes/
	./4_Finance/MonteCarloMultiGPU/
	./6_Advanced/fastWalshTransform/
	./6_Advanced/scan/
	./6_Advanced/alignedTypes/
	./6_Advanced/sortingNetworks/
)
	
PATH_ABS=${PWD}

BIN_RELEASE_PATH=${PATH_ABS}/bin/x86_64/linux/release

echo "Benchmarking in "${BIN_RELEASE_PATH}

dirlist=$(find ${BIN_RELEASE_PATH} -mindepth 1 -maxdepth 1 -type f)
for f in ${dirlist}
do 
	echo "Benchmarking "${f}
	hyperfine -M 10 --warmup 1 ${f}
	if [ "${f##*/}" == "simpleStreams" ]
	then
		echo "simpleStreams"
		hyperfine -M 10 --warmup 1 "${f} --use_cuda_malloc_host"
	fi
done

