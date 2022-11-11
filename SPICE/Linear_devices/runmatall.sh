#!/bin/bash

phelp() {
    cat <<EOF
Usage:
  [ OPTIONS ]  ${0##/}  NUM_THREADS
Run Matlab-only simulations using up to NUM_THREADS processes

OPTIONS/arguments and default values
  dir=../ccc       Directory for input/output .csv files
  rw=1             Wire Resistance
  nbits=1          1: TNN, 2: 5NN
  NUM_THREADS      Max number of processes to run in parallel

Note: Input is all "L*.csv" files in "dir"
EOF
}

[ -z "$1" -o "$1" = "-h" ] && { phelp; exit 2; }

[ -d "${0%/*}" ] && cd "${0%/*}"

let NT=$1; shift
((NT > 0)) || { phelp; exit 2; }

echo "OPTIONS: Xdir=${Xdir:=Xdir} Wdir=${Wdir:=Wdir} Ydir=${Ydir:=Ydir} rw=${rw:=rw} nbits=${nb:=nb} cbsize=${cbsize:=cbsize} NT=$NT"

proc=($(seq 1 $NT))        # 0 means unavailable
update_proc () { ((++pp >= NT)) && let pp=0; }
wait_for_proc () {
    while true; do
        for ((pp=0; pp<NT; pp++)); do
            [ "${proc[$pp]}" = 0 ] && continue
            kill -0 "${proc[$pp]}" 2>/dev/null || return;
        done
        sleep 0.5
    done
}

if [ $rw = 0p1 ]; then rw=0.1; fi

for ((i=1; i<=100; i++)) do
    file=L${i}.csv
    a=../../$Xdir/${file}
    b=../../$Wdir/${file}
    c=../../$Ydir/${file}
    d=$cbsize
    e=$rw
    f=$nb
    [ -f $c ] && continue
    wait_for_proc
    echo "launching process for $i"
    time matlab -singleCompThread -nodisplay -nojvm -r "Sim1 $a $b $c $d $e $f $ref; quite" &

    proc[$pp]=$!
    update_proc
done
wait
