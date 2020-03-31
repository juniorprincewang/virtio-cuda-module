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
	
parent_dir_name=${PWD##*/}
PATH_ABS=${PWD}

BIN_RELEASE_PATH=${PATH_ABS}/bin/x86_64/linux/release

echo "Removing bins in "${BIN_RELEASE_PATH}
rm -rf  ${BIN_RELEASE}/*

echo "Building binaries"
for f in ${dir_array[@]}
do 
	echo '[==]cd '${f}
	cd ${PATH_ABS}'/'${f};make clean; make;
done

