import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

data_path = '../../data/Beauty/'
train_path = data_path + 'workday/training_list.npy'
valid_path = data_path + 'workday/validation_dict.npy'
user_feat_path = data_path + 'workday/user_feature.npy'

week_train_path = data_path + 'weekend/training_list.npy'
week_test_path = data_path + 'weekend/testing_dict.npy'
week_user_feat_path = data_path + 'weekend/user_feature.npy'

# 加载数据
print("加载数据...")
train_list = np.load(train_path, allow_pickle=True)
week_train_list = np.load(week_train_path, allow_pickle=True)
valid_dict = np.load(valid_path, allow_pickle=True).item()
test_dict = np.load(week_test_path, allow_pickle=True).item()

# 构建训练字典
print("构建训练字典...")
uid_max = 0
iid_max = 0
train_dict = {}
for entry in tqdm(train_list, desc="处理训练数据"):
    user, item = entry
    if user not in train_dict:
        train_dict[user] = []
    train_dict[user].append(item)
    if user > uid_max:
        uid_max = user
    if item > iid_max:
        iid_max = item

# 构建验证列表和测试列表
print("构建验证和测试列表...")
valid_list = []
test_list = []

for u in tqdm(valid_dict, desc="处理验证数据"):
    if u > uid_max:
        uid_max = u
    for i in valid_dict[u]:
        valid_list.append([u, i])
        if i > iid_max:
            iid_max = i

for u in tqdm(test_dict, desc="处理测试数据"):
    if u > uid_max:
        uid_max = u
    for i in test_dict[u]:
        test_list.append([u, i])
        if i > iid_max:
            iid_max = i

# 转换为NumPy数组
train_list = np.array(train_list)
week_train_list = np.array(week_train_list)
valid_list = np.array(valid_list)
test_list = np.array(test_list)

n_users = max(uid_max + 1, 1217)
n_items = max(iid_max + 1, 3582)
print(f'n_users: {n_users}')
print(f'n_items: {n_items}')

# 构建稀疏矩阵
print("构建稀疏矩阵...")
train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                            (train_list[:, 0], train_list[:, 1])), dtype='float32',
                           shape=(n_users, n_items))
week_train_data = sp.csr_matrix((np.ones_like(week_train_list[:, 0]),
                            (week_train_list[:, 0], week_train_list[:, 1])), dtype='float32',
                           shape=(n_users, n_items))


# 加载用户特征
print("加载用户特征...")
work_user_feature = np.load(user_feat_path, allow_pickle=True).astype(np.float32)
week_user_feature = np.load(week_user_feat_path, allow_pickle=True).astype(np.float32)

# ======== 1. 用户特征变化计算 ========


def calculate_user_feature_change(work_user_feature, week_user_feature):
    """计算用户特征变化（内存优化版本）"""
    # 计算均值/std时避免内存峰值
    work_mean = np.zeros(work_user_feature.shape[1], dtype=np.float32)
    work_std = np.zeros(work_user_feature.shape[1], dtype=np.float32)
    week_mean = np.zeros(week_user_feature.shape[1], dtype=np.float32)
    week_std = np.zeros(week_user_feature.shape[1], dtype=np.float32)

    for i in tqdm(range(work_user_feature.shape[1]), desc="计算特征统计量"):
        work_mean[i] = np.mean(work_user_feature[:, i])
        work_std[i] = np.std(work_user_feature[:, i])
        week_mean[i] = np.mean(week_user_feature[:, i])
        week_std[i] = np.std(week_user_feature[:, i])

    # 标准化用户特征（避免完整矩阵操作）
    work_norm = np.empty_like(work_user_feature)
    week_norm = np.empty_like(week_user_feature)

    for i in tqdm(range(work_user_feature.shape[1]), desc="标准化特征"):
        work_norm[:, i] = (work_user_feature[:, i] - work_mean[i]) / (work_std[i] + 1e-8)
        week_norm[:, i] = (week_user_feature[:, i] - week_mean[i]) / (week_std[i] + 1e-8)

    # 计算欧几里得距离（分块计算避免内存峰值）
    euclidean_distance = np.zeros(work_user_feature.shape[0], dtype=np.float32)
    chunk_size = 1000

    for i in tqdm(range(0, work_user_feature.shape[0], chunk_size), desc="计算欧几里得距离"):
        end = min(i + chunk_size, work_user_feature.shape[0])
        diff = work_norm[i:end] - week_norm[i:end]
        euclidean_distance[i:end] = np.linalg.norm(diff, axis=1)

    # 特征差异的绝对值均值（分块计算）
    feature_difference = np.zeros(work_user_feature.shape[0], dtype=np.float32)

    for i in tqdm(range(0, work_user_feature.shape[0], chunk_size), desc="计算特征差异"):
        end = min(i + chunk_size, work_user_feature.shape[0])
        feature_difference[i:end] = np.mean(np.abs(work_user_feature[i:end] - week_user_feature[i:end]), axis=1)

    return euclidean_distance, feature_difference


print("计算用户特征变化...")
user_feature_euclidean, user_feature_difference = calculate_user_feature_change(work_user_feature, week_user_feature)


# ======== 2. 用户-物品交互变化计算 (修正版) ========
def calculate_interaction_change(work_matrix, week_matrix):
    """计算交互变化（修正版）"""
    n_users = work_matrix.shape[0]
    jaccard_similarity = np.zeros(n_users, dtype=np.float32)
    interaction_difference = np.zeros(n_users, dtype=np.float32)

    # 确保矩阵是二进制格式（0/1）
    work_matrix = work_matrix.astype(bool).astype(np.float32)
    week_matrix = week_matrix.astype(bool).astype(np.float32)

    # 计算每个用户的交集和并集大小
    print("计算交集...")
    intersection_matrix = work_matrix.multiply(week_matrix)
    intersection_counts = np.array(intersection_matrix.sum(axis=1)).flatten()

    print("计算并集...")
    union_counts = np.array((work_matrix + week_matrix).astype(bool).sum(axis=1)).flatten()

    print("计算Jaccard相似度...")
    valid_mask = union_counts > 0
    jaccard_similarity[valid_mask] = intersection_counts[valid_mask] / union_counts[valid_mask]

    print("计算交互差异...")
    interaction_difference = np.array(np.abs(work_matrix - week_matrix).sum(axis=1)).flatten() / n_items

    return jaccard_similarity, interaction_difference


print("计算用户交互变化...")
interaction_jaccard_sim, interaction_difference = calculate_interaction_change(train_data, week_train_data)


# ======== 3. 用户分类 (改进阈值策略) ========
def classify_users(euclidean_change, feature_diff, jaccard_sim, interaction_diff):
    """分类用户（改进阈值策略）"""
    # 动态选择基于分位数 - 使用更合理的分位数
    feature_threshold = (np.percentile(euclidean_change, 60), np.percentile(feature_diff, 80))
    interaction_threshold = (np.percentile(jaccard_sim, 30), np.percentile(interaction_diff, 50))

    print(f"特征阈值: 欧几里得={feature_threshold[0]:.4f}, 差异={feature_threshold[1]:.4f}")
    print(f"交互阈值: Jaccard={interaction_threshold[0]:.4f}, 差异={interaction_threshold[1]:.4f}")

    # 使用向量化操作进行分类
    iid_users = (
            (euclidean_change <= feature_threshold[0]) &
            (feature_diff <= feature_threshold[1]) &
            (jaccard_sim >= interaction_threshold[0]) &
            (interaction_diff <= interaction_threshold[1])
    )

    ood_users = ~iid_users

    return iid_users, ood_users


print("分类用户...")
iid_users, ood_users = classify_users(
    user_feature_euclidean, user_feature_difference,
    interaction_jaccard_sim, interaction_difference
)

# 保存分类结果
print("保存结果...")
np.save(data_path + 'iid_users.npy', np.array(iid_users))
np.save(data_path + 'ood_users.npy', np.array(ood_users))

# ======== 4. 结果输出 ========
print(f"Number of IID Users: {np.sum(iid_users)}")
print(f"Number of OOD Users: {np.sum(ood_users)}")
print(f"IID 用户比例: {np.mean(iid_users) * 100:.2f}%")
print(f"OOD 用户比例: {np.mean(ood_users) * 100:.2f}%")

# 打印统计信息
print("\n特征变化统计:")
print(
    f"欧几里得距离: min={np.min(user_feature_euclidean):.4f}, max={np.max(user_feature_euclidean):.4f}, mean={np.mean(user_feature_euclidean):.4f}, std={np.std(user_feature_euclidean):.4f}")
print(
    f"特征差异: min={np.min(user_feature_difference):.4f}, max={np.max(user_feature_difference):.4f}, mean={np.mean(user_feature_difference):.4f}, std={np.std(user_feature_difference):.4f}")

print("\n交互变化统计:")
print(
    f"Jaccard相似度: min={np.min(interaction_jaccard_sim):.4f}, max={np.max(interaction_jaccard_sim):.4f}, mean={np.mean(interaction_jaccard_sim):.4f}, std={np.std(interaction_jaccard_sim):.4f}")
print(
    f"交互差异: min={np.min(interaction_difference):.4f}, max={np.max(interaction_difference):.4f}, mean={np.mean(interaction_difference):.4f}, std={np.std(interaction_difference):.4f}")
