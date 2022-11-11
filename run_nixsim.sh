#!/bin/bash

Dataset='cifar10'  # mnist or cifar10
MODEL='vgg_cifar10' # mlp_mnist or vgg_cifar10

for cbsize in 32; do
for nb in 1; do
for rw in 0.1; do
for sa0 in 0.05; do
for sa1 in 0.05; do
for sig in 0.1; do
for noise in 0.1; do
    
    echo $cbsize $nb $rw $sig $sa0 $sa1 $noise

    #--------- Balanced Case -------------

    mkdir -p ./SCN_double_log/Sig${sig}_${sa0}_${sa1}_${noise}
    ### accuracy ###
    python main_binary.py --dataset ${Dataset} --model ${MODEL}_binary \
       -e results/${MODEL}_nb${nb}/model_best.pth.tar --cbsize $cbsize --nb $nb \
       --dcrxb true --sigma $sig --alpha 1 --sa0 $sa0 --sa1 $sa1 --read_noise $noise \
       --scn ./SCN/saves/${cbsize}-${nb}b-spRw${rw}Double.model \
       2>&1 | tee SCN_double_log/Sig${sig}_${sa0}_${sa1}_${noise}/vgg_cifar10_nb${nb}_${cbsize}_Rw${rw}.log

    #-------------------------------------

    #--------- Unbalanced Case -----------

    #mkdir -p ./SCN_Ref_log/Sig${sig}_${sa0}_${sa1}_${noise}
    ### accuracy ###
    #python main_binary.py --dataset ${Dataset} --model ${MODEL}_binary \
       #-e results/${MODEL}_nb${nb}/model_best.pth.tar --cbsize $cbsize --nb $nb \
       #--dcrxb false --sigma $sig --alpha 1 --sa0 $sa0 --sa1 $sa1 --read_noise $noise \
       #--scn ./SCN/saves/${cbsize}-${nb}b-spRw${rw}Ref1.model \
       #2>&1 | tee SCN_Ref_log/Sig${sig}_${sa0}_${sa1}_${noise}/vgg_cifar10_nb${nb}_${cbsize}_Rw${rw}.log


done
done
done
done
done
done
done