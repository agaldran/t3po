#!/usr/bin/env bash

for SEED in 0 1 2 3 4 5 6 7 8 9; do
    python train_baseline.py --dataset kather2016 --split_idx=0 --seed=${SEED} --save_path experiments/kather2016/baseline
    python train_baseline.py --dataset kather2016 --split_idx=0 --seed=${SEED} --save_path experiments/kather2016/mcdropout --dropout_p 0.2
    python train_t3po.py     --dataset kather2016 --split_idx=0 --seed=${SEED} --save_path experiments/kather2016/T3PO --transform T3PO_color_wide

    python train_baseline.py --dataset kather2016 --split_idx=1 --seed=${SEED} --save_path experiments/kather2016/baseline
    python train_baseline.py --dataset kather2016 --split_idx=1 --seed=${SEED} --save_path experiments/kather2016/mcdropout --dropout_p 0.2
    python train_t3po.py     --dataset kather2016 --split_idx=1 --seed=${SEED} --save_path experiments/kather2016/T3PO --transform T3PO_color_wide

    python train_baseline.py --dataset kather2016 --split_idx=2 --seed=${SEED} --save_path experiments/kather2016/baseline
    python train_baseline.py --dataset kather2016 --split_idx=2 --seed=${SEED} --save_path experiments/kather2016/mcdropout --dropout_p 0.2
    python train_t3po.py     --dataset kather2016 --split_idx=2 --seed=${SEED} --save_path experiments/kather2016/T3PO --transform T3PO_color_wide
done
for SEED in 0 1 2 3 4 5 6 7 8 9; do
    python train_baseline.py --dataset kather100k --split_idx=0 --seed=${SEED} --save_path experiments/kather100k/baseline --max_epoch 20
    python train_baseline.py --dataset kather100k --split_idx=0 --seed=${SEED} --save_path experiments/kather100k/mcdropout --dropout_p 0.2 --max_epoch 20
    python train_t3po.py     --dataset kather100k --split_idx=0 --seed=${SEED} --save_path experiments/kather100k/T3PO --transform T3PO_color_wide --max_epoch 20

    python train_baseline.py --dataset kather100k --split_idx=1 --seed=${SEED} --save_path experiments/kather100k/baseline --max_epoch 20
    python train_baseline.py --dataset kather100k --split_idx=1 --seed=${SEED} --save_path experiments/kather100k/mcdropout --dropout_p 0.2 --max_epoch 20
    python train_t3po.py     --dataset kather100k --split_idx=1 --seed=${SEED} --save_path experiments/kather100k/T3PO --transform T3PO_color_wide --max_epoch 20

    python train_baseline.py --dataset kather100k --split_idx=2 --seed=${SEED} --save_path experiments/kather100k/baseline --max_epoch 20
    python train_baseline.py --dataset kather100k --split_idx=2 --seed=${SEED} --save_path experiments/kather100k/mcdropout --dropout_p 0.2 --max_epoch 20
    python train_t3po.py     --dataset kather100k --split_idx=2 --seed=${SEED} --save_path experiments/kather100k/T3PO --transform T3PO_color_wide --max_epoch 20
done
