#!/bin/bash

WARMUP_TIMES=3
EVAL_TIMES=10


PATH_ABS=${PWD}

BIN_RELEASE_PATH=${PATH_ABS}/bin/x86_64/linux/release

bm=${BIN_RELEASE_PATH}/*
echo "Benchmarking in "${BIN_RELEASE_PATH}



OUTDIR=${PATH_ABS}/cudart_results

mkdir $OUTDIR &>/dev/null

exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    "$@" |& tee -a $OUTDIR/$b.txt ; }

# Save execution time in an array
declare -a time_array
b_idx=0

cd ${BIN_RELEASE_PATH}

for bs in $bm; do
    b=${bs##*/}
    i=0
    e2etime=0

    echo -n > $OUTDIR/$b.txt # clean output file

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
done

b_idx=0
for b in $bm; do
    echo "${b}: Average ${time_array[$b_idx]} ms per run"
    b_idx=$((b_idx+1))
done
