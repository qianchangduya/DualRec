#!/bin/bash
# Run OOD + IID for Meituan, ml1m, Beauty sequentially.
PY=/c/hl/Anaconda/conda/envs/HyperRec/python.exe
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "=== OOD Meituan ==="
echo "============================================"
cd /e/PangChao-DualRec/DualRec/code/OOD
$PY -u main002.py --epochs 100 --batch_size 500 --dataset Meituan 2>&1 | grep -vE "NumPy|pybind|module of|easiest|expect|_ARRAY_API|UserWarning|^\s*$|downgrade|warnings.warn"
echo "OOD Meituan EXIT ${PIPESTATUS[0]}"

echo "============================================"
echo "=== IID Meituan ==="
echo "============================================"
cd /e/PangChao-DualRec/DualRec/code/IID
$PY -u main004.py --data_dir ../../data/Meituan/ --data_name iid_users --path_name Meituan --epochs 100 2>&1 | grep -vE "NumPy|pybind|module of|easiest|expect|_ARRAY_API|UserWarning|^\s*$|downgrade|warnings.warn"
echo "IID Meituan EXIT ${PIPESTATUS[0]}"

echo "============================================"
echo "=== OOD ml1m ==="
echo "============================================"
cd /e/PangChao-DualRec/DualRec/code/OOD
$PY -u main002.py --epochs 100 --batch_size 500 --dataset ml1m 2>&1 | grep -vE "NumPy|pybind|module of|easiest|expect|_ARRAY_API|UserWarning|^\s*$|downgrade|warnings.warn"
echo "OOD ml1m EXIT ${PIPESTATUS[0]}"

echo "============================================"
echo "=== IID ml1m ==="
echo "============================================"
cd /e/PangChao-DualRec/DualRec/code/IID
$PY -u main004.py --data_dir ../../data/ml1m/ --data_name iid_users --path_name ml1m --epochs 100 2>&1 | grep -vE "NumPy|pybind|module of|easiest|expect|_ARRAY_API|UserWarning|^\s*$|downgrade|warnings.warn"
echo "IID ml1m EXIT ${PIPESTATUS[0]}"

echo "============================================"
echo "=== OOD Beauty ==="
echo "============================================"
cd /e/PangChao-DualRec/DualRec/code/OOD
$PY -u main002.py --epochs 100 --batch_size 500 --dataset Beauty 2>&1 | grep -vE "NumPy|pybind|module of|easiest|expect|_ARRAY_API|UserWarning|^\s*$|downgrade|warnings.warn"
echo "OOD Beauty EXIT ${PIPESTATUS[0]}"

echo "============================================"
echo "=== IID Beauty ==="
echo "============================================"
cd /e/PangChao-DualRec/DualRec/code/IID
$PY -u main004.py --data_dir ../../data/Beauty/ --data_name iid_users --path_name Beauty --epochs 100 2>&1 | grep -vE "NumPy|pybind|module of|easiest|expect|_ARRAY_API|UserWarning|^\s*$|downgrade|warnings.warn"
echo "IID Beauty EXIT ${PIPESTATUS[0]}"

echo "============================================"
echo "=== ALL DONE ==="
echo "============================================"
