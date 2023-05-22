#!/bin/bash

gmin=1e-6
gmax=1e-3

for cbsize in 32; do
for nb in 1; do
for rw in 0.1; do
for ref in 1; do

dir=./rand_wght/rand_wght-${cbsize}-${nb}b

#--------- Balanced Case -------------

sp=spRw${rw}Double
Ydir=$dir/$sp

mkdir -p $Ydir
mkdir -p $Ydir/lg
  
  dir=$dir rw=$rw ref=0 nb=$nb Ydir=$Ydir sig=0 alp=1 sa0=0 sa1=0 noise=0 gmin=$gmin gmax=$gmax ./SES/runmatall.sh 30
  
#-------------------------------------

#--------- Unbalanced Case -----------

#sp=spRw${rw}Ref$ref
#Ydir=$dir/$sp

#mkdir -p $Ydir
#mkdir -p $Ydir/lg

  #dir=$dir rw=$rw ref=$ref nb=$nb Ydir=$Ydir sig=0 alp=1 sa0=0 sa1=0 noise=0 gmin=$gmin gmax=$gmax ./SES/runmatall.sh 30
  
#-------------------------------------

done
done
done
done
