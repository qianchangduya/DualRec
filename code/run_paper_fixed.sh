#!/bin/bash
# Run OOD + IID for all 4 datasets with paper's fixed hyperparameters (γ=0.1, α=0.01, n=50, 200 epochs, patience=20).
# OOD: --regs 0.01 --lambda2 1.0 --epochs 200
# IID: --contrast_weight 0.1 --weight_decay 0.01 --max_seq_length 50 --epochs 200
PY=/c/hl/Anaconda/conda/envs/HyperRec/python.exe
export CUDA_VISIBLE_DEVICES=0

for DS in Meituan ml1m Beauty Taobao; do
    echo "============================================"
    echo "=== OOD $DS (regs=0.01, lambda2=1.0, 200 epochs) ==="
    echo "============================================"
    cd /e/PangChao-DualRec/DualRec/code/OOD
    $PY -u main002.py --epochs 200 --batch_size 500 --dataset $DS --regs 0.01 --lambda2 1.0 2>&1 | grep -vE "NumPy|pybind|module of|easiest|expect|_ARRAY_API|UserWarning|^\s*$|downgrade|warnings.warn"
    echo "OOD $DS EXIT ${PIPESTATUS[0]}"

    echo "============================================"
    echo "=== IID $DS (contrast_weight=0.1, weight_decay=0.01, max_seq_length=50, 200 epochs) ==="
    echo "============================================"
    cd /e/PangChao-DualRec/DualRec/code/IID
    $PY -u main004.py --data_dir ../../data/$DS/ --data_name iid_users --path_name $DS --epochs 200 --contrast_weight 0.1 --weight_decay 0.01 --max_seq_length 50 2>&1 | grep -vE "NumPy|pybind|module of|easiest|expect|_ARRAY_API|UserWarning|^\s*$|downgrade|warnings.warn"
    echo "IID $DS EXIT ${PIPESTATUS[0]}"
done

echo "============================================"
echo "=== ALL DONE ==="
echo "============================================"
