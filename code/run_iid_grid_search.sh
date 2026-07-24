#!/bin/bash
# IID hyperparameter grid search: lr_s, epsilon_s, lambda, contrast_weight.
# Phase 1: lr search (epsilon_s=0.3, lambda=0.015, contrast_weight=0.1 fixed)
# Phase 2: epsilon_s search (best lr, lambda=0.015, contrast_weight=0.1 fixed)
# Phase 3: lambda search (best lr, best epsilon_s, contrast_weight=0.1 fixed)
# 27 runs per dataset, 50 epochs each.
PY=/c/hl/Anaconda/conda/envs/HyperRec/python.exe
export CUDA_VISIBLE_DEVICES=0
LRS="0.0001 0.0005 0.001 0.002"
EPSILONS="0.1 0.3 0.5 0.7"
LAMBDAS="0.001 0.005 0.01 0.015 0.02 0.03 0.05"
for DS in Taobao; do
    echo "============================================"
    echo "=== IID $DS Phase 1: lr search (epsilon_s=0.3, lambda=0.015, contrast_weight=0.1) ==="
    echo "============================================"
    for LR in $LRS; do
        cd /e/PangChao-DualRec/DualRec/code/IID
        $PY -u main004.py --data_dir ../../data/$DS/ --data_name iid_users --path_name $DS --epochs 50 --lr $LR --hidden_dropout_prob 0.3 --pvn_weight 0.015 --contrast_weight 0.1 --weight_decay 0.01 --max_seq_length 50 --no_cuda 2>&1 | grep -E "Recall@20|final|Phase|exit|done"
        echo "--- $DS lr=$LR done ---"
    done
    echo "============================================"
    echo "=== IID $DS Phase 2: epsilon_s search (best lr, lambda=0.015, contrast_weight=0.1) ==="
    echo "============================================"
    BEST_LR="0.001"
    for EPSILON in $EPSILONS; do
        cd /e/PangChao-DualRec/DualRec/code/IID
        $PY -u main004.py --data_dir ../../data/$DS/ --data_name iid_users --path_name $DS --epochs 50 --lr $BEST_LR --hidden_dropout_prob $EPSILON --pvn_weight 0.015 --contrast_weight 0.1 --weight_decay 0.01 --max_seq_length 50 --no_cuda 2>&1 | grep -E "Recall@20|final|Phase|exit|done"
        echo "--- $DS epsilon_s=$EPSILON done ---"
    done
    echo "============================================"
    echo "=== IID $DS Phase 3: lambda search (best lr, best epsilon_s, contrast_weight=0.1) ==="
    echo "============================================"
    BEST_EPSILON="0.3"
    for LAMBDA in $LAMBDAS; do
        cd /e/PangChao-DualRec/DualRec/code/IID
        $PY -u main004.py --data_dir ../../data/$DS/ --data_name iid_users --path_name $DS --epochs 50 --lr $BEST_LR --hidden_dropout_prob $BEST_EPSILON --pvn_weight $LAMBDA --contrast_weight 0.1 --weight_decay 0.01 --max_seq_length 50 --no_cuda 2>&1 | grep -E "Recall@20|final|Phase|exit|done"
        echo "--- $DS lambda=$LAMBDA done ---"
    done
done
echo "============================================"
echo "=== ALL DONE ==="
echo "============================================"
