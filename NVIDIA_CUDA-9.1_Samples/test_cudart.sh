#!/bin/bash

WARMUP_TIMES=3
EVAL_TIMES=10

bm=( 
    ./0_Simple/vectorAdd
    ./0_Simple/simpleStreams
    ./0_Simple/matrixMul
    ./3_Imaging/convolutionSeparable
    ./4_Finance/binomialOptions
    ./4_Finance/BlackScholes
    ./4_Finance/MonteCarloMultiGPU
    ./6_Advanced/fastWalshTransform
    ./6_Advanced/scan
    ./6_Advanced/alignedTypes
    ./6_Advanced/sortingNetworks
)

PATH_ABS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTDIR=${PATH_ABS}/benchmark_results
CUDART_DIR=${PATH_ABS}

mkdir $OUTDIR &>/dev/null

exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    "$@" |& tee -a $OUTDIR/$b.txt ; }

# Save execution time in an array
declare -a time_array
b_idx=0


for bs in ${bm[*]}; do
    b=${bs##*/}
    i=0
    e2etime=0

    echo -n > $OUTDIR/$b.txt # clean output file

    cd $CUDART_DIR/$bs
    echo "$(date) # compiling $b"
    make clean &>/dev/null ; make &>/dev/null

    # warm up
    echo "$(date) # warming $b"
    for idx in `seq 1 ${WARMUP_TIMES}`; do
        ./${b}
        sleep 0.1
    done

    # test
    echo "$(date) # running $b"
    for idx in `seq 1 ${EVAL_TIMES}`; do
        tstart=$(date +%s%N)

        exe ./${b}

        tend=$((($(date +%s%N) - $tstart)/1000000))
        e2etime=$(( $tend + $e2etime ))
        i=$(( $i + 1 ))
        exe echo "$(date) # end2end elapsed $tend ms"

        exe echo
        sleep 0.1
    done

    et=$( echo "scale=3; $e2etime / $i " | bc )
    exe echo "${b}: Average ${et} ms per run"

    time_array[$b_idx]=${et}
    b_idx=$((b_idx+1))

    exe echo
    echo
    make clean &>/dev/null 
done

b_idx=0
for b in ${bm[*]}; do
    b=${bs##*/}
    echo "${b}: Average ${time_array[$b_idx]} ms per run"
    b_idx=$((b_idx+1))
done
