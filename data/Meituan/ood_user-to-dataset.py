import numpy as np


def split_ood_to_txt(ood_path, train_path, val_path, test_path):
    with open(ood_path, 'r') as f:

        raw_data = [list(map(int, line.strip().split())) for line in f.readlines()]


    train_lines = []
    val_lines = []
    test_lines = []

    for sample in raw_data:
        n = len(sample)
        if n < 3:
            raise ValueError(f"样本长度必须至少为3：{sample}")


        train_sample = sample[:-2]
        val_sample = [sample[0], sample[-2]]
        test_sample = [sample[0], sample[-1]]


        train_lines.append(' '.join(map(str, train_sample)))
        val_lines.append(' '.join(map(str, val_sample)))
        test_lines.append(' '.join(map(str, test_sample)))

    with open(train_path, 'w') as f:
        f.write('\n'.join(train_lines))
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_lines))
    with open(test_path, 'w') as f:
        f.write('\n'.join(test_lines))


def convert_train_txt_to_pairs(train_txt_path, train_npy_path):

    user_item_pairs = []

    with open(train_txt_path, 'r') as f:
        for line in f:

            parts = list(map(int, line.strip().split()))
            if len(parts) < 2:
                raise ValueError(f"行数据不完整，至少需要用户ID和1个物品ID：{line}")

            user_id = parts[0]
            item_ids = parts[1:]


            for item_id in item_ids:
                user_item_pairs.append([user_id, item_id])


    user_item_np = np.array(user_item_pairs, dtype=int)
    np.save(train_npy_path, user_item_np)
    print(f"成功生成 {train_npy_path}，形状：{user_item_np.shape}")

def txt_to_npy(txt_path, npy_path):
    with open(txt_path, 'r') as f:

        data = [list(map(int, line.strip().split())) for line in f.readlines()]

    np.save(npy_path, np.array(data, dtype=object))


if __name__ == "__main__":

    ood_path = 'ood_users.txt'
    train_txt_path = 'ood_train.txt'
    val_txt_path = 'ood_val.txt'
    test_txt_path = 'ood_test.txt'
    train_npy_path = 'ood_train.npy'
    val_npy_path = 'ood_val.npy'
    test_npy_path = 'ood_test.npy'


    split_ood_to_txt(ood_path, train_txt_path, val_txt_path, test_txt_path)
    print("成功生成 train.txt、val.txt、test.txt")


    convert_train_txt_to_pairs(train_txt_path, train_npy_path)
    txt_to_npy(val_txt_path, val_npy_path)
    txt_to_npy(test_txt_path, test_npy_path)
    print("成功生成 train.npy、val.npy、test.npy")
