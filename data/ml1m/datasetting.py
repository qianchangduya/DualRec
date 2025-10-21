import pandas as pd
import numpy as np

df = pd.read_csv("ratings.dat",
                 sep="::",
                 engine="python",
                 names=["user_id", "item_id", "rating", "timestamp"])


df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
df["weekday"] = df["datetime"].dt.weekday
df["day_type"] = df["weekday"].apply(lambda x: "weekday" if x < 5 else "weekend")


class_weekend = df[df["day_type"] == "weekend"].copy()
class_workday = df[df["day_type"] == "weekday"].copy()


class_weekend = class_weekend[class_weekend.groupby('user_id')['user_id'].transform('count') >= 3]
class_workday = class_workday[class_workday.groupby('user_id')['user_id'].transform('count') >= 3]

print(f"Weekend filtered: {class_weekend.shape[0]} rows")
print(f"Workday filtered: {class_workday.shape[0]} rows")


def get_last_three_items(group):
    return group['item_id'].iloc[-3:].values

last_three_weekend = class_weekend.groupby('user_id').apply(get_last_three_items)
last_three_workday = class_workday.groupby('user_id').apply(get_last_three_items)

last_three_df_weekend = pd.DataFrame(last_three_weekend.tolist(), index=last_three_weekend.index,
                                     columns=['last', 'second_last', 'third_last'])
last_three_df_workday = pd.DataFrame(last_three_workday.tolist(), index=last_three_workday.index,
                                     columns=['last', 'second_last', 'third_last'])


to_delete_weekend = last_three_df_weekend[
    (last_three_df_weekend['last'] == last_three_df_weekend['second_last']) |
    (last_three_df_weekend['second_last'] == last_three_df_weekend['third_last'])
].index
to_delete_workday = last_three_df_workday[
    (last_three_df_workday['last'] == last_three_df_workday['second_last']) |
    (last_three_df_workday['second_last'] == last_three_df_workday['third_last'])
].index

class_weekend = class_weekend[~class_weekend['user_id'].isin(to_delete_weekend)]
class_workday = class_workday[~class_workday['user_id'].isin(to_delete_workday)]


common_user_ids = pd.merge(class_weekend[['user_id']], class_workday[['user_id']],
                           on='user_id', how='inner')

class_weekend = class_weekend[class_weekend['user_id'].isin(common_user_ids['user_id'])]
class_workday = class_workday[class_workday['user_id'].isin(common_user_ids['user_id'])]

print(f"Weekend common: {class_weekend.shape[0]} rows")
print(f"Workday common: {class_workday.shape[0]} rows")


class_weekend = class_weekend[['user_id', 'item_id', 'rating']]
class_workday = class_workday[['user_id', 'item_id', 'rating']]


combined_user_ids = pd.concat([class_weekend['user_id'], class_workday['user_id']]).unique()
combined_item_ids = pd.concat([class_weekend['item_id'], class_workday['item_id']]).unique()

user_id_new_mapping = {uid: idx for idx, uid in enumerate(combined_user_ids)}
item_id_new_mapping = {iid: idx for idx, iid in enumerate(combined_item_ids)}


class_weekend['user_id_new'] = class_weekend['user_id'].map(user_id_new_mapping)
class_weekend['item_id_new'] = class_weekend['item_id'].map(item_id_new_mapping)

class_workday['user_id_new'] = class_workday['user_id'].map(user_id_new_mapping)
class_workday['item_id_new'] = class_workday['item_id'].map(item_id_new_mapping)


def split_dataset(df):
    train_data, val_data, test_data = [], [], []
    for user_id, group in df.groupby('user_id_new'):
        items = group['item_id_new'].values
        if len(items) >= 2:
            test_data.append((user_id, items[-1]))
            val_data.append((user_id, items[-2]))
            for iid in items[:-2]:
                train_data.append((user_id, iid))
        else:
            for iid in items:
                train_data.append((user_id, iid))
    return (
        pd.DataFrame(train_data, columns=['user_id_new', 'item_id_new']),
        pd.DataFrame(val_data, columns=['user_id_new', 'item_id_new']),
        pd.DataFrame(test_data, columns=['user_id_new', 'item_id_new']),
    )

train_weekend, val_weekend, test_weekend = split_dataset(class_weekend)
train_workday, val_workday, test_workday = split_dataset(class_workday)


class_weekend[['user_id_new', 'item_id_new']].to_csv('weekend.txt', sep='\t', index=False)
class_workday[['user_id_new', 'item_id_new']].to_csv('workday.txt', sep='\t', index=False)

train_weekend.to_csv('weekend_train.txt', sep='\t', index=False)
val_weekend.to_csv('weekend_val.txt', sep='\t', index=False)
test_weekend.to_csv('weekend_test.txt', sep='\t', index=False)

train_workday.to_csv('workday_train.txt', sep='\t', index=False)
val_workday.to_csv('workday_val.txt', sep='\t', index=False)
test_workday.to_csv('workday_test.txt', sep='\t', index=False)

print("基本数据集划分完成！")


class_weekend['rating'] = df.loc[class_weekend.index, 'rating'].values
class_weekend['timestamp'] = df.loc[class_weekend.index, 'timestamp'].values

class_workday['rating'] = df.loc[class_workday.index, 'rating'].values
class_workday['timestamp'] = df.loc[class_workday.index, 'timestamp'].values


combined = pd.concat([class_weekend, class_workday], axis=0)


user_all_mean = combined.groupby('user_id_new')['rating'].mean().rename("all_mean")


user_work_mean = class_workday.groupby('user_id_new')['rating'].mean().rename("work_mean")


user_week_mean = class_weekend.groupby('user_id_new')['rating'].mean().rename("week_mean")


work_user_feature = pd.concat([user_all_mean, user_work_mean], axis=1).fillna(0)
work_user_feature = work_user_feature.sort_index()[['all_mean', 'work_mean']]
work_user_feature.to_csv("work_user_feature.csv", sep="\t", index=False)


week_user_feature = pd.concat([user_all_mean, user_week_mean], axis=1).fillna(0)
week_user_feature = week_user_feature.sort_index()[['all_mean', 'week_mean']]
week_user_feature.to_csv("week_user_feature.csv", sep="\t", index=False)

print("work_user_feature.csv 和 week_user_feature.csv 已保存！")


user_feature = pd.concat([user_all_mean, user_work_mean, user_week_mean], axis=1).fillna(0)
user_feature = user_feature.sort_index()[['all_mean', 'work_mean', 'week_mean']]


user_feature.to_csv("user_feature.csv", sep="\t", index=False)

print("user_feature.csv 已保存！")



all_data = pd.concat([class_weekend, class_workday], axis=0)


all_data = all_data.sort_values(by=['user_id_new', 'timestamp'])


user_to_items = all_data.groupby('user_id_new')['item_id_new'].apply(list)


with open("../../../IORec-g/SASRec.pytorch-main/python/ml1m.txt", "w") as f:
    for user_id, items in user_to_items.items():
        line = str(user_id) + " " + " ".join(map(str, items))
        f.write(line + "\n")

print("ml1m.txt 已生成！")



weekend_df = pd.read_csv('../../data/ml1m/weekend/weekend.txt', sep='\t')
workday_df = pd.read_csv('../../data/ml1m/workday/workday.txt', sep='\t')


combined_df = pd.concat([weekend_df, workday_df], ignore_index=True)


iid_users = np.load('../../data/ml1m/iid_users.npy')
ood_users = np.load('../../data/ml1m/ood_users.npy')


iid_user_ids = set(np.where(iid_users)[0])
ood_user_ids = set(np.where(ood_users)[0])

iid_df = combined_df[combined_df['user_id_new'].isin(iid_user_ids)]
ood_df = combined_df[combined_df['user_id_new'].isin(ood_user_ids)]


iid_sorted = iid_df.sort_values(by='user_id_new')
ood_sorted = ood_df.sort_values(by='user_id_new')


iid_user_to_pois = iid_sorted.groupby('user_id_new')['item_id_new'].apply(list).to_dict()
ood_user_to_pois = ood_sorted.groupby('user_id_new')['item_id_new'].apply(list).to_dict()


total_user_to_pois = {**iid_user_to_pois, **ood_user_to_pois}
total_user_to_pois = dict(sorted(total_user_to_pois.items()))


total_result = []
for user_id, poi_ids in total_user_to_pois.items():
    total_result.append(f"{user_id} " + " ".join(map(str, poi_ids)))


with open('../../data/ml1m/iid_users.txt', 'w') as f:
    for user_id in sorted(iid_user_to_pois):
        f.write(f"{user_id} " + " ".join(map(str, iid_user_to_pois[user_id])) + '\n')

with open('../../data/ml1m/ood_users.txt', 'w') as f:
    for user_id in sorted(ood_user_to_pois):
        f.write(f"{user_id} " + " ".join(map(str,ood_user_to_pois[user_id])) + '\n')
