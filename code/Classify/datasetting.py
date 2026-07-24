#Meituan数据集处理
import numpy as np
import pandas as pd
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.abspath(os.path.join(_BASE_DIR, '../../data/Meituan/')) + '/'

# 加载 .txt 文件，假设数据是制表符分隔的
orders_train = pd.read_csv(_DATA_DIR + 'orders_train.txt', sep='\t')

# 将 dt 列转换为字符串类型，确保没有数字格式问题
orders_train['dt'] = orders_train['dt'].astype(str)

# 定义需要筛选的 dt 值
dt_values = ['20210306', '20210307', '20210313', '20210314', '20210320', '20210321']

# 筛选出 dt 列符合指定日期的数据（第一类）
class_1 = orders_train[orders_train['dt'].isin(dt_values)]

# 筛选出不符合指定日期的数据（第二类）
class_2 = orders_train[~orders_train['dt'].isin(dt_values)]

# 根据 user_id 分组并筛选出出现次数 >= 3 次的数据
class_weekend = class_1[class_1.groupby('user_id')['user_id'].transform('count') >= 3]
class_workday = class_2[class_2.groupby('user_id')['user_id'].transform('count') >= 3]

# 打印一下筛选结果，检查是否正确
print(f"Class 1 filtered: {class_weekend.shape[0]} rows")
print(f"Class 2 filtered: {class_workday.shape[0]} rows")


# 定义一个函数，获取每个 user_id 对应的倒数前三个 wm_poi_id
def get_last_three_poi_ids(group):
    return group['wm_poi_id'].iloc[-3:].values

# 获取每个 user_id 的倒数前三个 wm_poi_id
last_three_poi_ids_class_1 = class_weekend.groupby('user_id').apply(get_last_three_poi_ids)
last_three_poi_ids_class_2 = class_workday.groupby('user_id').apply(get_last_three_poi_ids)

# 将结果转换为 DataFrame，方便后续处理
last_three_df_class_1 = pd.DataFrame(last_three_poi_ids_class_1.tolist(), index=last_three_poi_ids_class_1.index, columns=['last', 'second_last', 'third_last'])
last_three_df_class_2 = pd.DataFrame(last_three_poi_ids_class_2.tolist(), index=last_three_poi_ids_class_2.index, columns=['last', 'second_last', 'third_last'])

# 找出需要删除的 user_id
to_delete_class_1 = last_three_df_class_1[(last_three_df_class_1['last'] == last_three_df_class_1['second_last']) | (last_three_df_class_1['second_last'] == last_three_df_class_1['third_last'])].index
to_delete_class_2 = last_three_df_class_2[(last_three_df_class_2['last'] == last_three_df_class_2['second_last']) | (last_three_df_class_2['second_last'] == last_three_df_class_2['third_last'])].index

# 删除这些 user_id 对应的行
class_1_filtered = class_weekend[~class_weekend['user_id'].isin(to_delete_class_1)]
class_2_filtered = class_workday[~class_workday['user_id'].isin(to_delete_class_2)]

# 获取两个文件中 user_id 的交集
common_user_ids = pd.merge(class_weekend[['user_id']], class_workday[['user_id']], on='user_id', how='inner')

# 过滤出两个数据集中 user_id 在交集中的行
class_1_common = class_weekend[class_weekend['user_id'].isin(common_user_ids['user_id'])]
class_2_common = class_workday[class_workday['user_id'].isin(common_user_ids['user_id'])]

# 打印筛选后的行数，检查是否正确
print(f"Class 1 common: {class_1_common.shape[0]} rows")
print(f"Class 2 common: {class_2_common.shape[0]} rows")


# 选择需要的列：user_id, wm_poi_id
class_1_common = class_1_common[['user_id', 'wm_poi_id']]
class_2_common = class_2_common[['user_id', 'wm_poi_id']]

# 为 user_id 和 wm_poi_id 重新编号
class_1_common['user_id_new'], user_id_mapping_1 = pd.factorize(class_1_common['user_id'])
class_1_common['wm_poi_id_new'], wm_poi_id_mapping_1 = pd.factorize(class_1_common['wm_poi_id'])

class_2_common['user_id_new'], user_id_mapping_2 = pd.factorize(class_2_common['user_id'])
class_2_common['wm_poi_id_new'], wm_poi_id_mapping_2 = pd.factorize(class_2_common['wm_poi_id'])

# 确保两个文件的编号对应一致（在所有数据中合并并重新编号）
combined_user_ids = pd.concat([class_1_common['user_id'], class_2_common['user_id']]).unique()
combined_wm_poi_ids = pd.concat([class_1_common['wm_poi_id'], class_2_common['wm_poi_id']]).unique()

# 对所有数据进行统一编号
user_id_new_mapping = {user_id: idx for idx, user_id in enumerate(combined_user_ids)}
wm_poi_id_new_mapping = {wm_poi_id: idx for idx, wm_poi_id in enumerate(combined_wm_poi_ids)}

# 应用统一编号
class_1_common['user_id_new'] = class_1_common['user_id'].map(user_id_new_mapping)
class_1_common['wm_poi_id_new'] = class_1_common['wm_poi_id'].map(wm_poi_id_new_mapping)

class_2_common['user_id_new'] = class_2_common['user_id'].map(user_id_new_mapping)
class_2_common['wm_poi_id_new'] = class_2_common['wm_poi_id'].map(wm_poi_id_new_mapping)

# 只保留新的编号列
class_1_common = class_1_common[['user_id_new', 'wm_poi_id_new']]
class_2_common = class_2_common[['user_id_new', 'wm_poi_id_new']]

# 保存处理后的文件
class_1_common.to_csv(_DATA_DIR + 'weekday/weekend.txt', sep='\t', index=False)
class_2_common.to_csv(_DATA_DIR + 'workday/workday.txt', sep='\t', index=False)

# 保存映射关系到文件
user_id_mapping_df = pd.DataFrame(list(user_id_new_mapping.items()), columns=['original_user_id', 'new_user_id'])
wm_poi_id_mapping_df = pd.DataFrame(list(wm_poi_id_new_mapping.items()), columns=['original_wm_poi_id', 'new_wm_poi_id'])

user_id_mapping_df.to_csv('user_id_mapping_Meituan.txt', sep='\t', index=False)
# wm_poi_id_mapping_df.to_csv('wm_poi_id_mapping_Meituan.txt', sep='\t', index= False)

def split_dataset(df):
    # 初始化训练集、验证集和测试集
    train_data = []
    val_data = []
    test_data = []

    # 对每个 user_id_new 进行操作
    for user_id, group in df.groupby('user_id_new'):
        # 获取每个 user_id_new 对应的 wm_poi_id_new 列表
        wm_poi_ids = group['wm_poi_id_new'].values

        # 获取编号
        wm_poi_ids_with_index = list(enumerate(wm_poi_ids))

        # 划分到不同的数据集中
        if len(wm_poi_ids_with_index) >= 2:
            # 测试集：最后一个编号
            test_data.append((user_id, wm_poi_ids_with_index[-1][1]))

            # 验证集：倒数第二个编号
            val_data.append((user_id, wm_poi_ids_with_index[-2][1]))

            # 训练集：其余的编号
            for wm_poi_id in wm_poi_ids_with_index[:-2]:
                train_data.append((user_id, wm_poi_id[1]))
        else:
            # 如果只有1个 wm_poi_id_new，全部作为训练集
            for wm_poi_id  in wm_poi_ids_with_index:
                train_data.append((user_id, wm_poi_id))

    # 将训练集、验证集、测试集合并为一个数据框
    train_df = pd.DataFrame(train_data, columns=['user_id_new', 'wm_poi_id_new'])
    val_df = pd.DataFrame(val_data, columns=['user_id_new', 'wm_poi_id_new'])
    test_df = pd.DataFrame(test_data, columns=['user_id_new', 'wm_poi_id_new'])

    return train_df, val_df, test_df

# 对 class_1_common 和 class_2_common 划分数据集
train_class_1, val_class_1, test_class_1 = split_dataset(class_1_common)
train_class_2, val_class_2, test_class_2 = split_dataset(class_2_common)

# 保存划分后的数据集
train_class_1.to_csv(_DATA_DIR + 'weekday/weekend_train.txt', sep='\t', index=False)
val_class_1.to_csv(_DATA_DIR + 'weekday/weekend_val.txt', sep='\t', index=False)
test_class_1.to_csv(_DATA_DIR + 'weekday/weekend_test.txt', sep='\t', index=False)

train_class_2.to_csv(_DATA_DIR + 'workday/workday_train.txt', sep='\t', index=False)
val_class_2.to_csv(_DATA_DIR + 'workday/workday_val.txt', sep='\t', index=False)
test_class_2.to_csv(_DATA_DIR + 'workday/workday_test.txt', sep='\t', index=False)

print("数据集划分完成并保存！")

print("处理完成并保存文件！")

print("保存完成！")


# 读取文件
weekend_df = pd.read_csv(_DATA_DIR + 'weekday/weekend.txt', sep='\t')
workday_df = pd.read_csv(_DATA_DIR + 'workday/workday.txt', sep='\t')

# 合并数据
combined_df = pd.concat([weekend_df, workday_df], ignore_index=True)

# 加载布尔数组
iid_users = np.load(_DATA_DIR + 'iid_users.npy')
ood_users = np.load(_DATA_DIR + 'ood_users.npy')

# 获取True值的user_id_new索引
iid_user_ids = set(np.where(iid_users)[0])
ood_user_ids = set(np.where(ood_users)[0])

# 筛选 IID 和 OOD 用户
iid_df = combined_df[combined_df['user_id_new'].isin(iid_user_ids)]
ood_df = combined_df[combined_df['user_id_new'].isin(ood_user_ids)]

# 排序
iid_sorted = iid_df.sort_values(by='user_id_new')
ood_sorted = ood_df.sort_values(by='user_id_new')

# 构建映射关系
iid_user_to_pois = iid_sorted.groupby('user_id_new')['wm_poi_id_new'].apply(list).to_dict()
ood_user_to_pois = ood_sorted.groupby('user_id_new')['wm_poi_id_new'].apply(list).to_dict()

# 合并映射并排序
total_user_to_pois = {**iid_user_to_pois, **ood_user_to_pois}
total_user_to_pois = dict(sorted(total_user_to_pois.items()))

# 生成合并后的数据
total_result = []
for user_id, poi_ids in total_user_to_pois.items():
    total_result.append(f"{user_id} " + " ".join(map(str, poi_ids)))

# 保存IID数据
with open(_DATA_DIR + 'iid_users.txt', 'w') as f:
    for user_id in sorted(iid_user_to_pois):
        f.write(f"{user_id} " + " ".join(map(str, iid_user_to_pois[user_id])) + '\n')

# 保存OOD数据
with open(_DATA_DIR + 'ood_users.txt', 'w') as f:
    for user_id in sorted(ood_user_to_pois):
        f.write(f"{user_id} " + " ".join(map(str,ood_user_to_pois[user_id])) + '\n')

# 保存合并排序后的数据
with open(_DATA_DIR + 'Meituan.txt', 'w') as f:
    for line in total_result:
        f.write(line + '\n')
