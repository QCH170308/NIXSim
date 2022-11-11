#! /bin/bash
bool=(false true)

#check_log () { 
#    if [ -f "$log" ]; then
#        echo "Log file already exists: $log" >&2
#        [ -n "$exit_if_log_exists" ] && exit 1 || return 1  # return 0 to overwrite
#    fi
#}

gg=0
update_gpu () { let gg++; [ $gg -gt 3 ] && gg=0; }
gpu=(1 1 1 1)                   # 0 means unavailable
wait_for_gpu () {
    while true; do
        for gg in {0..3}; do
            [ ${gpu[$gg]} = 0 ] && continue
            kill -0 "${gpu[$gg]}" 2>/dev/null || return;
        done
        sleep 0.5
    done
}

me=100   #maxEpochs
di=10     #displayInterval
si=$me    #saveInterval
bs=32     #batchSize
opt=adam  #optimizer

lr=1e-3
hc=32
nl=7

for cbsize in 32; do
for nb in 1; do
for rw in 0.1; do
for ref in 1; do

  #--------- Balanced Case -------------
  
  common=$cbsize-${nb}b-spRw${rw}Double
  
  #-------------------------------------

  #--------- Unbalanced Case -----------
  
  #common=$cbsize-${nb}b-spRw${rw}Ref${ref}
  
  #-------------------------------------
  
  tr=../rand_wght/IR_dataset/${common}_train.t7  # training dataset
  te=../rand_wght/IR_dataset/${common}_test.t7  # test dataset
  sn=lua_saves/${common}   #saveName
  lg=lua_saves/${common}.log
  mkdir -p lua_saves/

#  if [ -e $lg ]; then continue; fi
  wait_for_gpu
  echo l$nl c$hc nb$nb cbsize$cbsize rw$rw
  CUDA_VISIBLE_DEVICES=$gg th scn.lua -numLayers $nl -hiddenChannels $hc \
    -maxEpochs $me -displayInterval $di -saveInterval $si -saveName $sn \
    -learningRate $lr -batchSize $bs -optimizer $opt -cbsize $cbsize \
    -trainData $tr -testData $te 2>&1 | tee $lg &
  gpu[$gg]=$!
done
done
done
done