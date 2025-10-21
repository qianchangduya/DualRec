# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import pickle
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from trainers002 import FinetuneTrainer, DistSAModelTrainer
from models import S3RecModel
from seqmodels import SASRecModel, DistSAModel, DistMeanSAModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed

import numpy as np
import torch


class LossBasedEarlyStopping:
    """基于训练损失的早停策略"""

    def __init__(self, save_path, patience=5, verbose=False, delta=0.0001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_path = save_path
        self.delta = delta
        self.best_preds = None

    def __call__(self, val_loss, model, preds=None):
        score = -val_loss  # 因为我们希望损失越小越好

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, preds)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, preds)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, preds=None):
        """保存模型和预测结果"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

        if preds is not None:
            self.best_preds = preds  # 保存预测结果



def main():
    parser = argparse.ArgumentParser()


    parser.add_argument('--data_dir', default='../../data/Meituan/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='iid_users', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    # parser.add_argument("--model_name", default='Finetune_full', type=str)
    parser.add_argument("--model_name", default='DistSAModel', type=str)

    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    # parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    # parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    # parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.0, help="attention dropout p")
    # parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    # parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--max_seq_length', default=100, type=int)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    # parser.add_argument('--pvn_weight', default=0.1, type=float)
    parser.add_argument('--pvn_weight', default=0.005, type=float)
    parser.add_argument('--kernel_param', default=1.0, type=float)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    # 添加预测结果保存路径参数
    parser.add_argument("--prediction_path", type=str, default=None,
                        help="Path to save the prediction results (default: output_dir/predictions.npy)")

    args = parser.parse_args()
    if args.prediction_path is None:
        os.makedirs(args.output_dir, exist_ok=True)
        args.prediction_path = os.path.join(args.output_dir, f"{args.data_name}_predictions.npy")

    set_seed(args.seed)
    check_path(args.output_dir)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    #item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)

    #item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1
    #args.attribute_size = attribute_size + 1

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.ckp}-{args.kernel_param}-{args.pvn_weight}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    #args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    #eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=200)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    #test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=200)


    if args.model_name == 'DistSAModel':
        model = DistSAModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = DistSAModelTrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    elif args.model_name == 'DistMeanSAModel':
        model = DistMeanSAModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = DistSAModelTrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    else:
        model = SASRecModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)


    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info, _ = trainer.test(0, full_sort=True)

    else:
        # 初始化早停策略
        early_stopping = LossBasedEarlyStopping(
            save_path=args.checkpoint_path,
            patience=10,  # 容忍损失不下降的轮数
            verbose=True,
            delta=0.0001  # 损失变化阈值
        )


        for epoch in range(args.epochs):
            # 训练并获取损失
            train_loss = trainer.train(epoch)
            print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.6f}")

            # 验证（获取预测列表和答案列表）
            pred_list, answer_list = trainer.valid(epoch, full_sort=True)

            # 早停判断，传递答案列表
            early_stop = early_stopping(train_loss, trainer.model, pred_list)

            # 记录每个epoch的损失
            with open(args.log_file, 'a') as f:
                f.write(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}\n")

            if early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print('---------------Change to test_rating_matrix!-------------------')
        # 训练结束后，获取最佳预测和答案
        best_pred_list = early_stopping.best_preds

        if best_pred_list is not None and answer_list is not None:
            # 保存路径（如果需要单独保存）
            pred_path = args.prediction_path
            answer_path = os.path.splitext(pred_path)[0] + '_answers.npy'

            # 保存预测和答案
            np.save(pred_path, best_pred_list)
            np.save(answer_path, answer_list)


            print(f"Best predictions saved to {pred_path}")
            print(f"Answers saved to {answer_path}")


    print(args_str)

    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
main()
