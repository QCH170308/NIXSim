#! /bin/bash

for cbsize in 32; do
for nb in 1; do
for rw in 0.1; do

scn_model=./SCN/saves/${cbsize}-${nb}b-spRw${rw}Double.model #SCN model

echo $cbsize $nb $rw 

python simulation_test.py --input_quant $nb --weight_quant $nb --cbsize $cbsize --rw $rw --scn $scn_model  #SCN

done
done
done