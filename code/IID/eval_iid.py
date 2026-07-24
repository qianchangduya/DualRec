# -*- coding: utf-8 -*-
"""Evaluate IID predictions: Recall/NDCG/MRR @ K."""
import os
import sys
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)
from utils import recall_at_k, ndcg_k, cal_mrr

DATA = 'Taobao'
DATA_DIR = os.path.abspath(os.path.join(_BASE_DIR, '../../data', DATA)) + '/'
OUTPUT_DIR = os.path.join(_BASE_DIR, 'output') + os.sep

TOP_N = [1, 5, 10, 15, 20, 40]

# IID predictions: best validation predictions and the validation answers
# (predictions.npy = best-val predictions, predictions_test.npy = best-test predictions)
for split, pred_file, ans_file in [('valid', 'predictions', 'predictions_answers'),
                                   ('test', 'predictions_test', 'predictions_test_answers')]:
    pred_path = OUTPUT_DIR + f'{DATA}_{pred_file}.npy'
    ans_path = OUTPUT_DIR + f'{DATA}_{ans_file}.npy'
    if not os.path.exists(pred_path):
        print(f'{split}: {pred_path} not found, skipping')
        continue
    pred = np.load(pred_path, allow_pickle=True)
    ans = np.load(ans_path, allow_pickle=True)
    print(f'\n=== {DATA} IID [{split}] ===  users={len(pred)}, pred[0] len={len(pred[0])}')
    # pred is a 2D array of item indices (top-100), ans is 2D array of answer item ids
    pred_list = pred.tolist() if hasattr(pred, 'tolist') else pred
    ans_list = ans.tolist() if hasattr(ans, 'tolist') else ans
    recall, ndcg, mrr = [], [], 0
    for k in TOP_N:
        r, _ = recall_at_k(ans_list, pred_list, k)
        recall.append(r)
        n, _ = ndcg_k(ans_list, pred_list, k)
        ndcg.append(n)
    mrr, _ = cal_mrr(ans_list, pred_list)
    print(f'{split} Recall@{TOP_N}: {["%.4f" % x for x in recall]}')
    print(f'{split} NDCG@{TOP_N}:   {["%.4f" % x for x in ndcg]}')
    print(f'{split} MRR: {mrr:.4f}')
    for i, k in enumerate(TOP_N):
        print(f'  @{k:<3}: Recall={recall[i]:.4f}  NDCG={ndcg[i]:.4f}')
