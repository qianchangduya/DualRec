# -*- coding: utf-8 -*-
"""Evaluate OOD predictions: Recall/NDCG/MRR @ K for OOD users only."""
import os
import sys
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)
import evaluate_util

DATA = 'Taobao'
DATA_DIR = os.path.abspath(os.path.join(_BASE_DIR, '../../data', DATA)) + '/'
OUTPUT_DIR = os.path.join(_BASE_DIR, 'output') + os.sep

TOP_N = [10, 20, 50, 100]

ood_users = np.load(DATA_DIR + 'ood_users.npy', allow_pickle=True)
print(f'{DATA} OOD users: {int(ood_users.sum())} / {len(ood_users)}')

predict = np.load(OUTPUT_DIR + f'{DATA}_best_test_predict.npy', allow_pickle=True)
target = np.load(OUTPUT_DIR + f'{DATA}_best_test_target.npy', allow_pickle=True)

# main002.py's evaluate() already filters to OOD users before saving, so predict/target are OOD-only.
predict_ood = predict
target_ood = target

predict_list = predict_ood.tolist()
target_list = target_ood.tolist()

print(f'{DATA} OOD test: {len(predict_list)} users')
precision, recall, NDCG, MRR = evaluate_util.computeTopNAccuracy(target_list, predict_list, TOP_N)
print(f'{DATA} OOD [Test]: Precision: {"-".join(str(x) for x in precision)}  Recall: {"-".join(str(x) for x in recall)}  NDCG: {"-".join(str(x) for x in NDCG)}  MRR: {"-".join(str(x) for x in MRR)}')
for i, k in enumerate(TOP_N):
    print(f'  @{k:<3}: Recall={recall[i]:.4f}  NDCG={NDCG[i]:.4f}  MRR={MRR[i]:.4f}  Precision={precision[i]:.4f}')
