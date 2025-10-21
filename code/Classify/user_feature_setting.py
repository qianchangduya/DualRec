import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 读取 user.txt 文件
user_df = pd.read_csv('../../data/Meituan/users.txt', sep='\t', header=0)

# 读取 mapping.txt 文件
mapping_df = pd.read_csv('user_id_mapping_Meituan.txt', sep='\t', header=0)

# 筛选出 mapping.txt 中出现过的 original_user_id
user_ids_to_filter = mapping_df['original_user_id'].tolist()

# 筛选 user.txt 中的 user_id
filtered_user_df = user_df[user_df['user_id'].isin(user_ids_to_filter)]

# 创建 iid 类：只保留 user_id, avg_pay_amt, avg_pay_amt_weekdays
work_df = filtered_user_df[['user_id', 'avg_pay_amt', 'avg_pay_amt_weekdays']]

# 创建 ood 类：只保留 user_id, avg_pay_amt, avg_pay_amt_weekends
week_df = filtered_user_df[['user_id', 'avg_pay_amt', 'avg_pay_amt_weekends']]

df = filtered_user_df[['user_id', 'avg_pay_amt', 'avg_pay_amt_weekdays', 'avg_pay_amt_weekends']]

# 将 mapping.txt 中的映射关系转换为字典
user_id_mapping = dict(zip(mapping_df['original_user_id'], mapping_df['new_user_id']))

# 替换 iid 和 ood 中的 user_id 为 new_user_id
work_df.loc['user_id'] = work_df['user_id'].map(user_id_mapping)
week_df.loc['user_id'] = week_df['user_id'].map(user_id_mapping)
df.loc['user_id'] = df['user_id'].map(user_id_mapping)

# 对数据按 user_id 列排序，保持原有行不变
work_df_sorted = work_df.sort_values(by='user_id', ascending=True)
week_df_sorted = week_df.sort_values(by='user_id', ascending=True)
df_sorted = df.sort_values(by='user_id', ascending=True)

# 保存排序后的数据到新文件
# work_df_sorted.to_csv('work_users.txt', sep='\t', index=False)
# week_df_sorted.to_csv('week_users.txt', sep='\t', index=False)

# print("数据已分为 'work_users.txt' 和 'week_users.txt' 文件。")


# 定义区间映射函数
def map_interval(value):
    mapping = {
        '<29': 14.5,
        '[29,36)': 32.5,
        '[36,49)': 42.5,
        '[49,65)': 57,
        '>=65': 70  # 将 >=65 映射为 70
    }
    return mapping.get(value, np.nan)  # 对于未知值返回 NaN

# 读取 iid_user_sorted.txt 和 ood_user_sorted.txt 文件
work_DF = work_df_sorted
week_DF = week_df_sorted

# 转换区间数据为数值
work_DF['avg_pay_amt'] = work_DF['avg_pay_amt'].apply(map_interval)
work_DF['avg_pay_amt_weekdays'] = work_DF['avg_pay_amt_weekdays'].apply(map_interval)

week_DF['avg_pay_amt'] = week_DF['avg_pay_amt'].apply(map_interval)
week_DF['avg_pay_amt_weekends'] = week_DF['avg_pay_amt_weekends'].apply(map_interval)

df_sorted['avg_pay_amt'] = df_sorted['avg_pay_amt'].apply(map_interval)
df_sorted['avg_pay_amt_weekdays'] = df_sorted['avg_pay_amt_weekdays'].apply(map_interval)
df_sorted['avg_pay_amt_weekends'] = df_sorted['avg_pay_amt_weekends'].apply(map_interval)

# 初始化 KNNImputer
imputer = KNNImputer(n_neighbors=5)

# 对含有缺失值的特征进行插补
work_features = work_DF[['avg_pay_amt', 'avg_pay_amt_weekdays']].values
week_features = week_DF[['avg_pay_amt', 'avg_pay_amt_weekends']].values
features = df_sorted[['avg_pay_amt', 'avg_pay_amt_weekdays', 'avg_pay_amt_weekends']].values

work_features_imputed = imputer.fit_transform(work_features)
week_features_imputed = imputer.transform(week_features)
features_imputed = imputer.fit_transform(features)

# 将插补后的特征保存为新的 DataFrame
work_user_feature = pd.DataFrame(work_features_imputed, columns=['avg_pay_amt', 'avg_pay_amt_weekdays'])
week_user_feature = pd.DataFrame(week_features_imputed, columns=['avg_pay_amt', 'avg_pay_amt_weekends'])
user_feature = pd.DataFrame(features_imputed, columns=['avg_pay_amt', 'avg_pay_amt_weekdays', 'avg_pay_amt_weekends'])

# 保存处理后的特征文件
work_user_feature.to_csv('../../data/Meituan/workday/work_user_feature.csv', sep='\t', index=False)
week_user_feature.to_csv('../../data/Meituan/weekday/week_user_feature.csv', sep='\t', index=False)
user_feature.to_csv('../../data/Meituan/user_feature.csv', sep='\t', index=False)

print("特征文件已保存为 'work_user_feature.csv' 和 'week_user_feature.csv', user_feature。")