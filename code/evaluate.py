import numpy as np
import math
from collections import defaultdict
import os


# iid_preds_vail = np.load('IID/output/Meituan_predictions.npy')
# iid_answers_vail = np.load('IID/output/Meituan_predictions_answers.npy')
# ood_preds_vail = np.load('OOD/output/Meituan_best_valid_predict.npy')
# ood_answers_vail = np.load('OOD/output/Meituan_best_valid_target.npy')
#
#
# iid_preds_test = np.load('IID/output/Meituan_predictions_test.npy')
# iid_answers_test = np.load('IID/output/Meituan_predictions_test_answers.npy')
# ood_preds_test = np.load('OOD/output/Meituan_best_test_predict.npy')
# ood_answers_test = np.load('OOD/output/Meituan_best_test_target.npy')


iid_preds_vail = np.load('IID/output/Taobao_predictions.npy')
iid_answers_vail = np.load('IID/output/Taobao_predictions_answers.npy')
ood_preds_vail = np.load('OOD/output/Taobao_best_valid_predict.npy')
ood_answers_vail = np.load('OOD/output/Taobao_best_valid_target.npy')


iid_preds_test = np.load('IID/output/Taobao_predictions_test.npy')
iid_answers_test = np.load('IID/output/Taobao_predictions_test_answers.npy')
ood_preds_test = np.load('OOD/output/Taobao_best_test_predict.npy')
ood_answers_test = np.load('OOD/output/Taobao_best_test_target.npy')


# iid_preds_vail = np.load('IID/output/ml1m_predictions.npy')
# iid_answers_vail = np.load('IID/output/ml1m_predictions_answers.npy')
# ood_preds_vail = np.load('OOD/output/ml1m_best_valid_predict.npy')
# ood_answers_vail = np.load('OOD/output/ml1m_best_valid_target.npy')
#
#
# iid_preds_test = np.load('IID/output/ml1m_predictions_test.npy')
# iid_answers_test = np.load('IID/output/ml1m_predictions_test_answers.npy')
# ood_preds_test = np.load('OOD/output/ml1m_best_test_predict.npy')
# ood_answers_test = np.load('OOD/output/ml1m_best_test_target.npy')

# iid_preds_vail = np.load('IID/output/Beauty_predictions.npy')
# iid_answers_vail = np.load('IID/output/Beauty_predictions_answers.npy')
# ood_preds_vail = np.load('OOD/output/Beauty_best_valid_predict.npy')
# ood_answers_vail = np.load('OOD/output/Beauty_best_valid_target.npy')
#
#
# iid_preds_test = np.load('IID/output/Beauty_predictions_test.npy')
# iid_answers_test = np.load('IID/output/Beauty_predictions_test_answers.npy')
# ood_preds_test = np.load('OOD/output/Beauty_best_test_predict.npy')
# ood_answers_test = np.load('OOD/output/Beauty_best_test_target.npy')

print("IID 验证预测 shape:", iid_preds_vail.shape)
print("OOD 验证预测 shape:", ood_preds_vail.shape)

vail_preds = np.concatenate([iid_preds_vail, ood_preds_vail])
vail_answers = np.concatenate([iid_answers_vail, ood_answers_vail])


test_preds = np.concatenate([iid_preds_test, ood_preds_test])
test_answers = np.concatenate([iid_answers_test, ood_answers_test])

num_items = 100
print(f"验证集(vail)形状: 用户数={vail_preds.shape[0]}, 物品数={vail_preds.shape[1]}")
print(f"测试集(test)形状: 用户数={test_preds.shape[0]}, 物品数={test_preds.shape[1]}")



def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    valid_count = 0

    for rank in pred_list:

        if rank < 0 or rank >= num_items:
            continue

        valid_count += 1

        # MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
            MRR += 1.0 / (rank + 1.0)
        else:
            pass

    if valid_count == 0:
        return 0.0, 0.0, 0.0

    return HIT / valid_count, NDCG / valid_count, MRR / valid_count


def safe_to_set(value):

    if isinstance(value, np.ndarray):

        return set(value.ravel().astype(int).tolist())
    elif isinstance(value, (list, tuple)):

        return set(int(x) for x in value)
    elif hasattr(value, '__iter__'):

        return set(int(x) for x in value)
    else:

        return {int(value)}

def calculate_ranks(preds, answers):
    ranks = []
    for user_id in range(preds.shape[0]):
        user_preds = preds[user_id]

        true_items = safe_to_set(answers[user_id])

        if not true_items:
            ranks.append(num_items)
            continue

        sorted_indices = np.argsort(user_preds)[::-1]
        rank = num_items


        for pos, item in enumerate(sorted_indices):
            if item in true_items:
                rank = pos
                break

        ranks.append(rank)

    return np.array(ranks)


def calculate_topk(preds, k=10):
    topk_lists = []
    for user_id in range(preds.shape[0]):
        user_preds = preds[user_id]
        sorted_indices = np.argsort(user_preds)[::-1]
        topk = sorted_indices[:k].tolist()
        topk_lists.append(topk)
    return topk_lists


def prepare_answers(answers):
    prepared = []
    for ans in answers:

        ans_set = safe_to_set(ans)
        prepared.append(list(ans_set))
    return prepared


def calculate_user_overlap(predicted, k=10):
    n_users = len(predicted)
    total_overlap = 0.0
    count = 0

    for i in range(n_users):

        set_i = set(int(x) for x in predicted[i][:k])
        for j in range(i + 1, n_users):
            set_j = set(int(x) for x in predicted[j][:k])
            overlap = len(set_i & set_j) / k
            total_overlap += overlap
            count += 1

    return total_overlap / count if count > 0 else 0.0

def get_average_rank(preds, answers):
    total_rank = 0.0
    total_hits = 0

    for user_id in range(preds.shape[0]):
        user_preds = preds[user_id]
        true_items = safe_to_set(answers[user_id])

        if not true_items:
            continue

        sorted_indices = np.argsort(user_preds)[::-1]
        ranks = [np.where(sorted_indices == item)[0][0] for item in true_items if item in sorted_indices]
        if ranks:
            total_rank += np.mean(ranks) + 1  # 排名从1开始算
            total_hits += 1

    if total_hits == 0:
        return 0.0

    return total_rank / total_hits


def evaluate_dataset(dataset_name, preds, answers, topk_list=[5, 10, 20, 50, 70, 90]):

    ranks = calculate_ranks(preds, answers)
    topk_preds = calculate_topk(preds, max(topk_list))
    prepared_answers = prepare_answers(answers)

    print("=" * 80)
    print(f"Comprehensive Evaluation Results for {dataset_name} Set")
    print(f"Users: {preds.shape[0]}, Items: {preds.shape[1]}")
    print(f"Invalid ranks: {np.sum((ranks < 0) | (ranks >= num_items))}")
    print("=" * 80)


    print("\nRanking-based Metrics:")
    print("-" * 80)
    for k in topk_list:
        hit, ndcg, mrr = get_metric(ranks, k)
        print(f"[Rank-based] HR@{k}: {hit:.6f}, NDCG@{k}: {ndcg:.6f}, MRR@{k}: {mrr:.6f}")

    avg_rank = get_average_rank(preds, answers)
    print(f"Average Rank (all hits): {avg_rank:.6f}")

    return {
        'ranks': ranks,
        'topk_preds': topk_preds,
        'prepared_answers': prepared_answers,
        'average_rank': avg_rank,
    }


if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("Evaluating VALIDATION Set (vail)")
    print("=" * 80)
    vail_results = evaluate_dataset("VALIDATION (vail)", vail_preds, vail_answers)


    print("\n" + "=" * 80)
    print("Evaluating TEST Set (test)")
    print("=" * 80)
    test_results = evaluate_dataset("TEST (test)", test_preds, test_answers)
