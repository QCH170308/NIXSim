#!/bin/bash

gmin=1e-6
gmax=1e-3

for cbsize in 32; do
for nb in 1; do
for rw in 0.1; do
for sig in 0.1; do
for sa0 in 0.05; do
for sa1 in 0.05; do
for noise in 0.1; do
for ref in 1; do

dir=./rand_wght/rand_wght-${cbsize}-${nb}b

#--------- Balanced Case -------------

sp=spRw${rw}-${sig}-${sa0}-${sa1}-${noise}Double
Ydir=$dir/$sp

mkdir -p $Ydir
mkdir -p $Ydir/lg
  
  dir=$dir rw=$rw ref=0 nb=$nb Ydir=$Ydir sig=$sig alp=1 sa0=$sa0 sa1=$sa1 noise=$noise gmin=$gmin gmax=$gmax ./SES/runmatall.sh 30
  
#-------------------------------------

#--------- Unbalanced Case -----------

#sp=spRw${rw}-${sig}-${sa0}-${sa1}-${noise}Ref$ref
#mkdir -p $dir/$sp

  #dir=$dir rw=$rw ref=$ref nb=$nb sp=$sp sig=$sig alp=1 sa0=$sa0 sa1=$sa1 noise=$noise gmin=$gmin gmax=$gmax ./SES/runmatall.sh 30
  
#-------------------------------------

done
done
done
done
done
done
done
done