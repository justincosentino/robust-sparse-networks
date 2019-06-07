#!/usr/bin/env bash
L1_REGS=( "0.0" "0.01" "0.001" "0.0001")
ATTACKS=( "fgsm" "pgd" )
EXPERIMENTS=( "no_pruning" )

MODEL="dense-300-100"
DATASET="digits"
TRIALS="10"
EPOCHS="100"
BATCH_SIZE="128"
LEARNING_RATE="0.001"
DEVICES="1,2,3,4"


for EXP in "${EXPERIMENTS[@]}"
do
    for L1 in "${L1_REGS[@]}"
    do
        for ATK in "${ATTACKS[@]}"
        do
            # without adv training
            python -m robust-sparse-networks.run_experiments --model=$MODEL --dataset=$DATASET --trials=$TRIALS --epochs=$EPOCHS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --l1_reg=$L1 --attack=$ATK --devices=$DEVICES --experiment=$EXP

            # with adv training
            python -m robust-sparse-networks.run_experiments --model=$MODEL --dataset=$DATASET --trials=$TRIALS --epochs=$EPOCHS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --l1_reg=$L1 --attack=$ATK --devices=$DEVICES --experiment=$EXP --adv_train
        done
    done
done
