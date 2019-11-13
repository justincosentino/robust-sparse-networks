#!/usr/bin/env bash
L1_REGS=( "0.0" )
ATTACKS=( "pgd" )
EXPERIMENTS=( "reinit_rand" )

MODEL="dense-300-100"
DATASET="fashion"
TRIALS="5"
TRAIN_ITERS="50000"
PRUNE_ITERS="20"
BATCH_SIZE="60"
LEARNING_RATE="0.0012"
DEVICES="0"


for EXP in "${EXPERIMENTS[@]}"
do
    for L1 in "${L1_REGS[@]}"
    do
        for ATK in "${ATTACKS[@]}"
        do
            # without adv training
            python -m robust-sparse-networks.run_experiments --model=$MODEL --dataset=$DATASET --trials=$TRIALS --train_iters=$TRAIN_ITERS --prune_iters=$PRUNE_ITERS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --l1_reg=$L1 --attack=$ATK --devices=$DEVICES --experiment=$EXP

            # with adv training
            python -m robust-sparse-networks.run_experiments --model=$MODEL --dataset=$DATASET --trials=$TRIALS --train_iters=$TRAIN_ITERS --prune_iters=$PRUNE_ITERS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --l1_reg=$L1 --attack=$ATK --devices=$DEVICES --experiment=$EXP --adv_train
        done
    done
done
