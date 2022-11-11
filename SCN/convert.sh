#!/bin/bash

for cbsize in 32; do
for nb in 1; do
for rw in 0.1; do
for ref in 1; do

  #--------- Balanced Case -------------
  
  common=${cbsize}-${nb}b-spRw${rw}Double
  
  #--------- Unbalanced Case -----------
  
  #common=$cbsize-${nb}b-spRw${rw}Ref${ref}
  
  #-------------------------------------
  
  mkdir -p csv_saves/$common
  mkdir -p npy_saves/$common
  mkdir -p saves/
  th save_t7_to_csv.lua ./lua_saves/${common}_epoch100.model csv_saves/$common
  ./save_csv_to_npy.py csv_saves/$common npy_saves/$common
  ./save_scn_from_npy.py npy_saves/$common saves/${common}.model $cbsize
done
done
done
done