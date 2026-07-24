# -*- coding: utf-8 -*-
"""Merge evaluation for DualRec: combine IID (stable users) and OOD (unstable users) predictions
and compute combined Recall/NDCG/MRR @ K, matching Table 4 of the paper.

Per-user routing:
  - stable users  -> IID path  (sequential Wasserstein self-attention)
  - unstable users -> OOD path  (causal counterfactual)

This script concatenates the two paths' test predictions + answers and reports:
  (1) IID-only metrics (stable users)
  (2) OOD-only metrics (unstable users)
  (3) Combined DualRec metrics (all users)  <- comparable to Table 4

The metrics (recall_at_k / ndcg_k / cal_mrr) are rank-order invariant: they only check whether a ground-truth
item appears in the top-k, so concatenating stable + unstable predictions yields the correct combined metric.

Saves: <dataset>_merged_predictions_test.npy, <dataset>_merged_predictions_test_answers.npy,
        <dataset>_merged_metrics.txt
"""
import os
import sys
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE_DIR, 'IID'))
from utils import recall_at_k, ndcg_k, cal_mrr

TOP_N = [1, 5, 10, 15, 20, 50]

# ---------- dataset configuration ----------
# For each dataset: (iid_dir, ood_dir, dataset_name)
#   - IID predictions:  <iid_dir>/output/<Dataset>_predictions_test.npy / <Dataset>_predictions_test_answers.npy
#     (top-100 item indices, ascending Wasserstein distance; answers = last item per stable user)
#   - OOD predictions:  <ood_dir>/output/<Dataset>_best_test_predict.npy / <Dataset>_best_test_target.npy
#     (top-100 item indices, descending P* score; answers = last item per unstable user)

DATASETS = {
    'Taobao':  ('IID', 'OOD'),
    'Meituan': ('IID', 'OOD'),
    'ml1m':    ('IID', 'OOD'),
    'Beauty':  ('IID', 'OOD'),
}


def compute_metrics(pred_list, ans_list, label):
    recall, ndcg = [], []
    for k in TOP_N:
        r, _ = recall_at_k(ans_list, pred_list, k)
        n, _ = ndcg_k(ans_list, pred_list, k)
        recall.append(r)
        ndcg.append(n)
    mrr, _ = cal_mrr(ans_list, pred_list)
    print('\n=== ' + label + ' ===')
    print('users = ' + str(len(pred_list)))
    print('Recall@' + str(TOP_N) + ': ' + str(['%.4f' % x for x in recall]))
    print('NDCG@' + str(TOP_N) + ':   ' + str(['%.4f' % x for x in ndcg]))
    print('MRR: ' + ('%.4f' % mrr))
    for i, k in enumerate(TOP_N):
        print('  @' + str(k) + ': Recall=' + ('%.4f' % recall[i]) + '  NDCG=' + ('%.4f' % ndcg[i]))
    return recall, ndcg, mrr


def main():
    for dataset, (iid_sub, ood_sub) in DATASETS.items():
        iid_out = os.path.join(_BASE_DIR, iid_sub, 'output')
        ood_out = os.path.join(_BASE_DIR, ood_sub, 'output')

        # IID predictions (stable users)
        iid_pred_p = os.path.join(iid_out, dataset + '_predictions_test.npy')
        iid_ans_p = os.path.join(iid_out, dataset + '_predictions_test_answers.npy')
        # OOD predictions (unstable users)
        ood_pred_p = os.path.join(ood_out, dataset + '_best_test_predict.npy')
        ood_ans_p = os.path.join(ood_out, dataset + '_best_test_target.npy')

        print('=' * 60)
        print('Dataset: ' + dataset)
        print('=' * 60)

        have_iid = os.path.exists(iid_pred_p) and os.path.exists(iid_ans_p)
        have_ood = os.path.exists(ood_pred_p) and os.path.exists(ood_ans_p)

        if not have_iid and not have_ood:
            print('  [SKIP] No predictions found for ' + dataset)
            continue

        merged_pred = []
        merged_ans = []

        if have_iid:
            iid_pred = np.load(iid_pred_p, allow_pickle=True)
            iid_ans = np.load(iid_ans_p, allow_pickle=True)
            iid_pred_list = iid_pred.tolist() if hasattr(iid_pred, 'tolist') else list(iid_pred)
            iid_ans_list = [list(a) for a in iid_ans]
            print('  IID (stable):   ' + str(len(iid_pred_list)) + ' users, pred shape ' + str(iid_pred.shape) + ', ans shape ' + str(iid_ans.shape))
            recall_i, ndcg_i, mrr_i = compute_metrics(iid_pred_list, iid_ans_list, dataset + ' IID (stable users)')
            merged_pred.extend(iid_pred_list)
            merged_ans.extend(iid_ans_list)
        else:
            print('  [WARN] IID predictions not found: ' + iid_pred_p)
            recall_i = ndcg_i = mrr_i = None

        if have_ood:
            ood_pred = np.load(ood_pred_p, allow_pickle=True)
            ood_ans = np.load(ood_ans_p, allow_pickle=True)
            ood_pred_list = ood_pred.tolist() if hasattr(ood_pred, 'tolist') else list(ood_pred)
            ood_ans_list = [list(a) for a in ood_ans]
            print('  OOD (unstable): ' + str(len(ood_pred_list)) + ' users, pred shape ' + str(ood_pred.shape) + ', ans shape ' + str(ood_ans.shape))
            recall_o, ndcg_o, mrr_o = compute_metrics(ood_pred_list, ood_ans_list, dataset + ' OOD (unstable users)')
            merged_pred.extend(ood_pred_list)
            merged_ans.extend(ood_ans_list)
        else:
            print('  [WARN] OOD predictions not found: ' + ood_pred_p)
            recall_o = ndcg_o = mrr_o = None

        if have_iid and have_ood:
            recall_c, ndcg_c, mrr_c = compute_metrics(merged_pred, merged_ans, dataset + ' DualRec COMBINED (stable + unstable)')

            # save merged predictions + answers + metrics
            merged_pred_arr = np.array(merged_pred, dtype=object)
            merged_ans_arr = np.array(merged_ans, dtype=object)
            np.save(os.path.join(iid_out, dataset + '_merged_predictions_test.npy'), merged_pred_arr)
            np.save(os.path.join(iid_out, dataset + '_merged_predictions_test_answers.npy'), merged_ans_arr)

            metrics_lines = []
            metrics_lines.append('=== ' + dataset + ' DualRec Combined (stable + unstable, ' + str(len(merged_pred)) + ' users) ===')
            for i, k in enumerate(TOP_N):
                metrics_lines.append('@' + str(k) + ': Recall=' + ('%.4f' % recall_c[i]) + '  NDCG=' + ('%.4f' % ndcg_c[i]))
            metrics_lines.append('MRR: ' + ('%.4f' % mrr_c))
            metrics_lines.append('')
            if recall_i is not None:
                metrics_lines.append('IID (stable, ' + str(len(iid_pred_list)) + ' users):')
                for i, k in enumerate(TOP_N):
                    metrics_lines.append('  @' + str(k) + ': Recall=' + ('%.4f' % recall_i[i]) + '  NDCG=' + ('%.4f' % ndcg_i[i]))
                metrics_lines.append('  MRR: ' + ('%.4f' % mrr_i))
                metrics_lines.append('')
            if recall_o is not None:
                metrics_lines.append('OOD (unstable, ' + str(len(ood_pred_list)) + ' users):')
                for i, k in enumerate(TOP_N):
                    metrics_lines.append('  @' + str(k) + ': Recall=' + ('%.4f' % recall_o[i]) + '  NDCG=' + ('%.4f' % ndcg_o[i]))
                metrics_lines.append('  MRR: ' + ('%.4f' % mrr_o))
            metrics_txt = '\n'.join(metrics_lines)
            with open(os.path.join(iid_out, dataset + '_merged_metrics.txt'), 'w') as f:
                f.write(metrics_txt)
            print('\n  [SAVED] merged predictions + answers + metrics to ' + os.path.join(iid_out, dataset + '_merged_*'))


if __name__ == '__main__':
    main()
