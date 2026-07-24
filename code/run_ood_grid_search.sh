#!/bin/bash
# OOD hyperparameter grid search: lr_u, dropout, beta, anneal_cap.
# Phase 1: lr search (dropout=0.5, beta=0.2 fixed)
# Phase 2: dropout search (best lr, beta=0.2 fixed)
# Phase 3: beta search (best lr, best dropout)
# 27 runs per dataset, 50 epochs each.
PY=/c/hl/Anaconda/conda/envs/HyperRec/python.exe
export CUDA_VISIBLE_DEVICES=0
LRS="0.0001 0.001 0.002"
DROPOUTS="0.1 0.3 0.5 0.7"
BETAS="0.0 0.1 0.2 0.5"
for DS in Taobao; do
    echo "============================================"
    echo "=== OOD $DS Phase 1: lr search (dropout=0.5, beta=0.2) ==="
    echo "============================================"
    for LR in $LRS; do
        cd /e/PangChao-DualRec/DualRec/code/OOD
        $PY -u main002.py --epochs 50 --batch_size 500 --dataset $DS --lr $LR --dropout 0.5 --anneal_cap 0.2 --regs 0.01 --lambda2 1.0 --lambda3 1.0 2>&1 | grep -E "Recall@20|final|Phase|exit|done"
        echo "--- $DS lr=$LR done ---"
    done
    echo "============================================"
    echo "=== OOD $DS Phase 2: dropout search (best lr, beta=0.2) ==="
    echo "============================================"
    BEST_LR="0.001"
    for DROPOUT in $DROPOUTS; do
        cd /e/PangChao-DualRec/DualRec/code/OOD
        $PY -u main002.py --epochs 50 --batch_size 500 --dataset $DS --lr $BEST_LR --dropout $DROPOUT --anneal_cap 0.2 --regs 0.01 --lambda2 1.0 --lambda3 1.0 2>&1 | grep -E "Recall@20|final|Phase|exit|done"
        echo "--- $DS dropout=$DROPOUT done ---"
    done
    echo "============================================"
    echo "=== OOD $DS Phase 3: beta search (best lr, best dropout) ==="
    echo "============================================"
    BEST_DROPOUT="0.1"
    for BETA in $BETAS; do
        cd /e/PangChao-DualRec/DualRec/code/OOD
        $PY -u main002.py --epochs 50 --batch_size 500 --dataset $DS --lr $BEST_LR --dropout $BEST_DROPOUT --anneal_cap $BETA --regs 0.01 --lambda2 1.0 --lambda3 1.0 2>&1 | grep -E "Recall@20|final|Phase|exit|done"
        echo "--- $DS beta=$BETA done ---"
    done
done
echo "============================================"
echo "=== ALL DONE ==="
echo "============================================"
