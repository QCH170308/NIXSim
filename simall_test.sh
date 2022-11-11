#! /bin/bash

for cbsize in 32; do
for nb in 1; do
for rw in 0.1; do
for sa0 in 0.05; do
for sa1 in 0.05; do
for sig in 0.1; do
for noise in 0.1; do
    
    echo $cbsize $nb $rw $sig $sa0 $sa1 $noise

    #--------- Balanced Case -------------

    scn_model=./SCN/saves_var_saf/${cbsize}-${nb}b-spRw${rw}Double.model
    python nonidealities_test.py --input_quant $nb --weight_quant $nb --cbsize $cbsize --rw $rw --scn $scn_model --sigma $sig --alpha 1 --sa0 $sa0 --sa1 $sa1 --read_noise $noise --dcrxb true

    #-------------------------------------

    #--------- Unbalanced Case -----------

    #scn_model=./SCN/saves_var_saf/${cbsize}-${nb}b-spRw${rw}Ref${ref}.model
    #python nonidealities_test.py --input_quant $nb --weight_quant $nb --cbsize $cbsize --rw $rw --scn $scn_model --sigma $sig --alpha 1 --sa0 $sa0 --sa1 $sa1 --read_noise $noise --dcrxb false

    #-------------------------------------




done
done
done
done
done
done
done