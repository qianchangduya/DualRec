import argparse

import time

import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np



# from tensorboardX import SummaryWriter

# 导入数据处理和评估相关库

from scipy import sparse

import models

import random

import data_utils002

import evaluate_util

import os



# 脚本所在目录，用于解析相对路径（兼容 VSCode/PyCharm 等不同工作目录）

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))



parser = argparse.ArgumentParser(description='PyTorch COR')

parser.add_argument('--model_name', type=str, default='COR', help='model name')

# parser.add_argument('--dataset', type=str, default='Meituan', help='dataset name')

parser.add_argument('--dataset', type=str, default='Taobao', help='dataset name')

# parser.add_argument('--dataset', type=str, default='ml1m', help='dataset name')

# parser.add_argument('--dataset', type=str, default='Beauty', help='dataset name')



parser.add_argument('--data_path', type=str, default='../../data/', help='directory of all datasets')

parser.add_argument('--log_name', type=str, default='', help='log/model special name')

# parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')

parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')

parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')

parser.add_argument('--batch_size', type=int, default=500, help='batch size')

parser.add_argument("--mlp_dims", default='[100, 20]', help="the dims of the mlp encoder")

parser.add_argument("--mlp_p1_1_dims", default='[100, 200]', help="the dims of the mlp p1-1")

parser.add_argument("--mlp_p1_2_dims", default='[1]', help="the dims of the mlp p1-2")

parser.add_argument("--mlp_p2_dims", default='[]', help="the dims of the mlp p2")

parser.add_argument("--mlp_p3_dims", default='[10]', help="the dims of the mlp p3")

parser.add_argument("--Z1_hidden_size", type=int, default=8, help="hidden size of Z1")

parser.add_argument('--E2_hidden_size', type=int, default=20, help='hidden size of E2')

parser.add_argument('--Z2_hidden_size', type=int, default=20, help='hidden size of Z2')

parser.add_argument('--total_anneal_steps', type=int, default=200000,

                    help='the total number of gradient updates for annealing')

parser.add_argument('--anneal_cap', type=float, default=0.2, help='largest annealing parameter')

parser.add_argument('--sample_freq', type=int, default=1, help='sample frequency for Z1/Z2')

parser.add_argument('--CI', type=int, default=1,

                    help='whether use counterfactual inference in ood settings')

parser.add_argument('--bn', type=int, default=1, help='batch norm')

parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

# parser.add_argument('--dropout', type=float, default=0.7, help='dropout')

parser.add_argument('--regs', type=float, default=0, help='regs')

parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')

# L_Unstable composite loss weights (Eq. 25): L_Unstable = L_rec + λ1·L_KL + λ2·L_reg + λ3·L_CONS

parser.add_argument('--lambda1', type=float, default=1.0, help='λ1: weight for L_KL (annealed)')

parser.add_argument('--lambda2', type=float, default=0.0, help='λ2: weight for L_reg')

parser.add_argument('--lambda3', type=float, default=1.0, help='λ3: weight for L_CONS (counterfactual consistency)')

parser.add_argument("--topN", default='[10, 20, 50, 100]', help="the recommended item num")

parser.add_argument('--cuda', action='store_true', help='use CUDA')

parser.add_argument('--gpu', type=str, default='0', help='GPU id')

parser.add_argument("--ood_test", default=True, help="whether test ood data during iid training")

parser.add_argument('--save_path', type=str, default='./models/', help='path to save the final model')

parser.add_argument('--act_function', type=str, default='tanh', help='activation function')

parser.add_argument('--ood_finetune', action='store_true', help='fine-tuning on ood data')

parser.add_argument('--ckpt', type=str, default=None, help='pre-trained model directory')

parser.add_argument('--X', type=int, default=10, help='use X percent of ood data for fine-tuning')

args = parser.parse_args()

# 将相对路径基于脚本所在目录解析为绝对路径

if not os.path.isabs(args.data_path):

    args.data_path = os.path.join(_BASE_DIR, args.data_path)

if not os.path.isabs(args.save_path):

    args.save_path = os.path.join(_BASE_DIR, args.save_path)

print(args)





random_seed = 1

torch.manual_seed(random_seed)  # cpu

torch.cuda.manual_seed_all(random_seed)  # gpu

np.random.seed(random_seed)  # numpy

random.seed(random_seed)  # random and transforms





torch.backends.cudnn.deterministic = True  # cudnn

torch.backends.cudnn.benchmark = False



def worker_init_fn(worker_id):

    np.random.seed(random_seed + worker_id)



os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# device = torch.device("cuda:0" if args.cuda else "cpu")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')





# ood_users mask (boolean) loaded per-dataset for OOD-user-only evaluation

ood_users = np.load(os.path.join(_BASE_DIR, '../../data/', args.dataset, 'ood_users.npy'), allow_pickle=True)







# train_dataset = 'Meituan'

train_dataset = 'Taobao'

# train_dataset = 'ml1m'

# train_dataset = 'Beauty'



data_path = args.data_path + args.dataset + '/'



train_path = data_path + 'ood_train.npy'

valid_path = data_path + 'ood_val.npy'

test_path = data_path + 'ood_test.npy'

user_feat_path = data_path + 'user_feature.npy'





train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items = \

    data_utils002.data_load(train_path, valid_path, test_path, args.dataset)



user_feature = np.load(user_feat_path, allow_pickle=True)





N = train_data.shape[0]

idxlist = list(range(N))





mask_val = train_data

mask_test = train_data + valid_y_data





if args.ood_finetune:

    model = torch.load(args.ckpt)

    ckpt_structure = args.ckpt.split('_')

else:

    E1_size = user_feature.shape[1]

    Z1_size = args.Z1_hidden_size

    mlp_q_dims = [n_items + user_feature.shape[1]] + eval(args.mlp_dims) + [args.E2_hidden_size]





    mlp_p1_dims = [E1_size + args.E2_hidden_size] + eval(args.mlp_p1_1_dims) + [Z1_size]



    mlp_p1_1_dims = [1] + eval(args.mlp_p1_1_dims)

    mlp_p1_2_dims = [mlp_p1_1_dims[-1]] + eval(args.mlp_p1_2_dims)



    mlp_p2_dims = [args.E2_hidden_size] + eval(args.mlp_p2_dims) + [args.Z2_hidden_size]

    mlp_p3_dims = [Z1_size + args.Z2_hidden_size] + eval(args.mlp_p3_dims) + [n_items]  # need to delete





    adj = np.concatenate((np.array([[0.0] * E1_size + [1.0] * args.E2_hidden_size,

                                    [0.0] * E1_size + [1.0] * args.E2_hidden_size]),

                          np.ones([6, E1_size + args.E2_hidden_size])), axis=0)

    adj = torch.FloatTensor(adj).to(device)





    if args.model_name == 'COR':

        model = models.COR(mlp_q_dims, mlp_p1_dims, mlp_p2_dims, mlp_p3_dims, \

                           adj, E1_size, n_items, args.dropout, args.bn, args.sample_freq, args.regs,

                           args.act_function).to(device)

    elif args.model_name == 'COR_G':

        # COR_G is a graph variant; item_feature is not loaded in main002.py (pass None).

        model = models.COR_G(mlp_q_dims, mlp_p1_1_dims, mlp_p1_2_dims, mlp_p2_dims, mlp_p3_dims, \

                             None, adj, E1_size, n_items, args.dropout, args.bn, args.sample_freq, args.regs,

                             args.act_function).to(device)



optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

criterion = models.loss_function







def naive_sparse2tensor(data):

    # 将稀疏矩阵转换为张量

    return torch.FloatTensor(data.toarray())





def adjust_lr(e):



    if args.dataset == 'Meituan':

        if e > 90:

            for param_group in optimizer.param_groups:

                param_group['lr'] = 0.1 * args.lr

    elif args.dataset == 'Taobao':

        if e > 60:

            for param_group in optimizer.param_groups:

                param_group['lr'] = 0.1 * args.lr

    else:

        pass





def train():



    model.train()

    global update_count

    np.random.shuffle(idxlist)



    epoch_loss = 0.0

    num_batches = 0

    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):

        end_idx = min(start_idx + args.batch_size, N)

        data = train_data[idxlist[start_idx:end_idx]]

        user_f = torch.FloatTensor(user_feature[idxlist[start_idx:end_idx]]).to(device)

        data = naive_sparse2tensor(data).to(device)



        if args.total_anneal_steps > 0:

            anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)

        else:

            anneal = args.anneal_cap



        optimizer.zero_grad()



        # Eq. 15-25: forward returns factual P, counterfactual P_star, factual E2 (mu, logvar),

        # Z2 (Z_stable), Z2_star (Z_stable*), reg_loss.

        P, P_star, mu, logvar, Z2, Z2_star, reg_loss = model(data, user_f)



        # Eq. 25: L_Unstable = L_rec(P*) + λ1·anneal·L_KL + λ2·L_reg + λ3·L_CONS

        loss = criterion(P_star, data, mu, logvar, Z2, Z2_star, reg_loss, anneal,

                         args.lambda1, args.lambda2, args.lambda3)

        loss.backward()

        optimizer.step()

        update_count += 1

        epoch_loss += loss.item()

        num_batches += 1



    print(f"[Train] epoch {epoch} | L_Unstable avg = {epoch_loss / max(1, num_batches):.4f}")





def evaluate(data_tr, data_te, his_mask, user_feat, topN, CI=0):



    assert data_tr.shape[0] == data_te.shape[0] == user_feat.shape[0]





    model.eval()

    total_loss = 0.0

    global update_count

    e_idxlist = list(range(data_tr.shape[0]))

    e_N = data_tr.shape[0]



    predict_items = []

    target_items = []

    for i in range(e_N):

        target_items.append(data_te[i, :].nonzero()[1].tolist())



    with torch.no_grad():

        for start_idx in range(0, e_N, args.batch_size):

            end_idx = min(start_idx + args.batch_size, N)

            data = data_tr[e_idxlist[start_idx:end_idx]]

            user_f = torch.FloatTensor(user_feat[e_idxlist[start_idx:end_idx]]).to(device)

            data_tensor = naive_sparse2tensor(data).to(device)

            his_data = his_mask[e_idxlist[start_idx:end_idx]]



            if args.total_anneal_steps > 0:

                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)

            else:

                anneal = args.anneal_cap



            P, P_star, mu, logvar, Z2, Z2_star, reg_loss = model(data_tensor, user_f, CI)



            # Eq. 25: L_Unstable on the counterfactual prediction P*

            loss = criterion(P_star, data_tensor, mu, logvar, Z2, Z2_star, reg_loss, anneal,

                             args.lambda1, args.lambda2, args.lambda3)

            total_loss += loss.item()



            # Rank by the counterfactual prediction P* (CI=1) or the factual P (CI=0)

            recon_batch = P_star if CI else P

            recon_batch[his_data.nonzero()] = -np.inf



            _, indices = torch.topk(recon_batch, topN[-1])

            indices = indices.cpu().numpy().tolist()

            predict_items.extend(indices)





    total_loss /= len(range(0, e_N, args.batch_size))



    predict_items_np = np.array(predict_items, dtype=object)

    target_items_np = np.array(target_items, dtype=object)



    # align ood_users mask to the number of evaluated users (robust to n_users floor mismatch)

    n_eval = predict_items_np.shape[0]

    ood_mask = ood_users[:n_eval] if len(ood_users) >= n_eval else np.pad(ood_users, (0, n_eval - len(ood_users)), constant_values=False)

    predict_items_ood = predict_items_np[ood_mask]

    target_items_ood = target_items_np[ood_mask]





    predict_items = predict_items_ood.tolist()

    target_items = target_items_ood.tolist()





    test_results = evaluate_util.computeTopNAccuracy(target_items, predict_items, topN)

    return total_loss, predict_items, target_items, test_results







best_recall = -np.inf

best_test_recall = -np.inf

best_epoch = 0

best_ood_epoch = 0

best_valid_results = None

best_test_results = None

best_ood_test_results = None

update_count = 0





K = 0 if args.dataset == 'synthetic' else 2

evaluate_interval = 1 if args.dataset == 'yelp' or args.ood_finetune else 5



best_valid_loss = float('inf')

patience = 20

patience_counter = 0

best_model_state = None

early_stopped = False

best_valid_predict = None

best_valid_target = None

best_test_predict = None

best_test_target = None



save_dir = os.path.join(_BASE_DIR, 'output')

os.makedirs(save_dir, exist_ok=True)





try:

    for epoch in range(1, args.epochs + 1):

        epoch_start_time = time.time()

        adjust_lr(epoch)

        train()



        if epoch % evaluate_interval == 0:





            valid_loss, valid_predict, valid_target, valid_results = evaluate(valid_x_data, valid_y_data, mask_val, user_feature, eval(args.topN), args.CI)

            test_loss, test_predict, test_target, test_results = evaluate(test_x_data, test_y_data, mask_test, user_feature, eval(args.topN), args.CI)



            # test_results = [precision@10, precision@20, precision@50, precision@100, recall@10, recall@20, recall@50, recall@100, NDCG@10, NDCG@20, NDCG@50, NDCG@100, MRR]

            # index 5 = recall@20

            valid_recall20 = valid_results[5]

            test_recall20 = test_results[5]



            # select best model by validation Recall@20 (primary metric from paper)

            if valid_recall20 > best_recall:

                best_recall = valid_recall20

                best_epoch = epoch

                best_test_recall = test_recall20

                best_valid_loss = valid_loss

                patience_counter = 0

                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}



                np.save(os.path.join(save_dir, '{}_best_valid_predict.npy'.format(train_dataset)), valid_predict)

                np.save(os.path.join(save_dir, '{}_best_valid_target.npy'.format(train_dataset)), valid_target)

                np.save(os.path.join(save_dir, '{}_best_test_predict.npy'.format(train_dataset)), test_predict)

                np.save(os.path.join(save_dir, '{}_best_test_target.npy'.format(train_dataset)), test_target)



                best_valid_predict = valid_predict

                best_valid_target = valid_target

                best_test_predict = test_predict

                best_test_target = test_target

                print(f"[Epoch {epoch}] Recall@20 improved: {valid_recall20:.4f} (valid), {test_recall20:.4f} (test), valid_loss={valid_loss:.4f})), saving best model")

            else:

                patience_counter += 1

                print(f"[Epoch {epoch}] Recall@20: {valid_recall20:.4f} (valid), {test_recall20:.4f} (test), valid_loss={valid_loss:.4f}), patience remaining: {patience - patience_counter}")



            if patience_counter >= patience:

                print(f"[Epoch {epoch}] 触发早停！最佳验证损失为：{best_valid_loss:.4f}")

                early_stopped = True

                break



except KeyboardInterrupt:

    print('-' * 18)

    early_stopped = True

    print('Exiting from training early')



print('===' * 18)





if best_model_state is not None:

    model.load_state_dict(best_model_state)

    print("已加载训练过程中最佳模型参数")



# Final machine-readable summary: dataset, best Recall@20 (valid), best test Recall@20, best valid loss, best epoch

# Final machine-readable summary: dataset, best Recall@20 (valid), best test Recall@20, best valid loss, best epoch
print(f"FINAL_SUMMARY dataset={train_dataset} best_recall20_valid={best_recall20_valid:.6f} best_recall20_test={best_test_recall20:.6f} best_valid_loss={best_valid_loss:.6f} best_epoch={best_epoch} early_stopped={early_stopped}")







