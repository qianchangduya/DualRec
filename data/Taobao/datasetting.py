import numpy as np
import os
from collections import defaultdict
import pandas as pd


data_path = '../../data/Taobao/'
iid_mask = np.load(data_path + 'iid_users.npy')
ood_mask = np.load(data_path + 'ood_users.npy')


user_classification = {}
for user_id in range(len(iid_mask)):
    if iid_mask[user_id]:
        user_classification[user_id] = 'iid'
    elif ood_mask[user_id]:
        user_classification[user_id] = 'ood'
    else:

        user_classification[user_id] = 'ood'

print(f"已加载用户分类数据: {len(user_classification)} 个用户")



def load_user_data(file_path):

    user_to_items = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user_id = int(parts[0])
            items = list(map(int, parts[1:]))
            user_to_items[user_id] = items
    return user_to_items


print("加载周末数据...")
weekend_data = load_user_data('../../data/Taobao/first_seven/first_seven.txt')
print("加载工作日数据...")
workday_data = load_user_data('../../data/Taobao/last_three/last_three.txt')


print("合并周末和工作日数据...")
combined_data = defaultdict(list)
all_users = set(weekend_data.keys()) | set(workday_data.keys())

for user_id in all_users:

    weekend_items = weekend_data.get(user_id, [])
    workday_items = workday_data.get(user_id, [])


    combined_data[user_id] = weekend_items + workday_items


print("分类用户数据...")
iid_user_to_pois = {}
ood_user_to_pois = {}

for user_id, items in combined_data.items():

    if user_id >= len(user_classification):

        ood_user_to_pois[user_id] = items
        continue

    if user_classification[user_id] == 'iid':
        iid_user_to_pois[user_id] = items
    else:
        ood_user_to_pois[user_id] = items

print(f"IID用户数量: {len(iid_user_to_pois)}")
print(f"OOD用户数量: {len(ood_user_to_pois)}")


print("创建合并排序后的所有用户数据...")
total_result = []
for user_id in sorted(combined_data.keys()):
    items = combined_data[user_id]
    line = f"{user_id} " + " ".join(map(str, items))
    total_result.append(line)


os.makedirs(data_path, exist_ok=True)


print("保存IID用户数据...")
with open('../../data/Taobao/iid_users.txt', 'w') as f:
    for user_id in sorted(iid_user_to_pois.keys()):
        items = iid_user_to_pois[user_id]
        f.write(f"{user_id} " + " ".join(map(str, items)) + '\n')


print("保存OOD用户数据...")
with open('../../data/Taobao/ood_users.txt', 'w') as f:
    for user_id in sorted(ood_user_to_pois.keys()):
        items = ood_user_to_pois[user_id]
        f.write(f"{user_id} " + " ".join(map(str, items)) + '\n')


print("保存合并排序后的所有用户数据...")
with open('../../data/Taobao/Taobao.txt', 'w') as f:
    for line in total_result:
        f.write(line + '\n')

print("数据处理完成！")


def count_interactions(file_path):

    total_interactions = 0

    with open(file_path, 'r') as f:
        for line in f:

            parts = line.strip().split()
            if not parts:
                continue

            num_interactions = len(parts) - 1
            total_interactions += num_interactions

    return total_interactions



first_seven_path = '../../data/Taobao/first_seven/first_seven.txt'
last_three_path = '../../data/Taobao/last_three/last_three.txt'


first_seven_count = count_interactions(first_seven_path)
last_three_count = count_interactions(last_three_path)
total_count = first_seven_count + last_three_count


print(f"first_seven.txt 交互记录数量: {first_seven_count}")
print(f"last_three.txt 交互记录数量: {last_three_count}")
print(f"总交互记录数量: {total_count}")


file_path = 'user_feature.txt'
df = pd.read_csv(file_path, sep='\t')


df = df.iloc[:, 1:]


output_path = 'user_feature.csv'

df.to_csv(output_path, sep='\t', index=False)