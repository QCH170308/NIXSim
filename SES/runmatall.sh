#!/bin/bash

phelp() {
    cat <<EOF
Usage:
  [ OPTIONS ]  ${0##/}  NUM_THREADS
Run Matlab-only simulations using up to NUM_THREADS processes

OPTIONS/arguments and default values
  dir=../ccc       Directory for input/output .csv files
  NUM_THREADS      Max number of processes to run in parallel

Note: Input is all "L*.csv" files in "dir"
EOF
}

[ -z "$1" -o "$1" = "-h" ] && { phelp; exit 2; }

[ -d "${0%/*}" ] && cd "${0%/*}"

let NT=$1; shift
((NT > 0)) || { phelp; exit 2; }

echo "OPTIONS: dir=${dir:=../ccc} rw=${rw:=1} ref=${ref:=0} nb=${nb:=1} sp=${sp:=sp} NT=$NT \
sig=${sig:=0} alp=${alp:=0} sa0=${sa0:=0} sa1=${sa1:=0} noise=${noise:=0}"

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
if [ $rw = 0p01 ]; then rw=0.01; fi

mkdir -p ../$Ydir/
mkdir -p ../$Ydir/lg

let cnt=0
for a in ../$dir/L*.csv; do
    b=${a##*/}
    c=../$Ydir/$b
    [ -f $c ] && continue
    arrname="${b%.*}"
    l=../$Ydir/lg/$arrname.log
    wait_for_proc
    echo "launching process for $a .. $cnt"
    ./run_Sim.sh /home/quanch/MATLAB/R2022a/ $a $c $rw $ref $nb $sig $alp $sa0 $sa1 $noise $gmin $gmax
    proc[$pp]=$!
    update_proc

    let cnt++
done
wait
