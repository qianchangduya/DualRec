#将 CSV 文件转换为 Numpy 格式
# 将 TXT 文件转换为稀疏矩阵并保存为 Numpy 格式
import numpy as np
import pandas as pd
import scipy.sparse as sp

# 数据路径
# data_path = '../../data/Meituan/'
# train_path = data_path + 'workday/workday_train.txt'
# week_train_path = data_path + 'weekday/weekend_train.txt'
# valid_path = data_path + 'workday/workday_val.txt'
# week_valid_path = data_path + 'weekday/weekend_val.txt'
# test_path = data_path + 'workday/workday_test.txt'
# week_test_path = data_path + 'weekday/weekend_test.txt'
# work_user_feat_path = data_path + 'workday/work_user_feature.csv'
# week_user_feat_path = data_path + 'weekday/week_user_feature.csv'
# user_feat_path = data_path + 'user_feature.csv'

# data_path = '../../data/Taobao/'
# train_path = data_path + 'first_seven/first_seven_train.txt'
# week_train_path = data_path + 'last_three/last_three_train.txt'
# valid_path = data_path + 'first_seven/first_seven_val.txt'
# week_valid_path = data_path + 'last_three/last_three_val.txt'
# test_path = data_path + 'first_seven/first_seven_test.txt'
# week_test_path = data_path + 'last_three/last_three_test.txt'
# work_user_feat_path = data_path + 'first_seven/seven_user_feature.csv'
# week_user_feat_path = data_path + 'last_three/last_user_feature.csv'
# user_feat_path = data_path + 'user_feature.csv'


# data_path = '../../data/ml1m/'
# train_path = data_path + 'workday/workday_train.txt'
# week_train_path = data_path + 'weekend/weekend_train.txt'
# valid_path = data_path + 'workday/workday_val.txt'
# week_valid_path = data_path + 'weekend/weekend_val.txt'
# test_path = data_path + 'workday/workday_test.txt'
# week_test_path = data_path + 'weekend/weekend_test.txt'
# work_user_feat_path = data_path + 'workday/work_user_feature.csv'
# week_user_feat_path = data_path + 'weekend/week_user_feature.csv'
# user_feat_path = data_path + 'user_feature.csv'

data_path = '../../data/Beauty/'
train_path = data_path + 'workday/workday_train.txt'
week_train_path = data_path + 'weekend/weekend_train.txt'
valid_path = data_path + 'workday/workday_val.txt'
week_valid_path = data_path + 'weekend/weekend_val.txt'
test_path = data_path + 'workday/workday_test.txt'
week_test_path = data_path + 'weekend/weekend_test.txt'
work_user_feat_path = data_path + 'workday/work_user_feature.csv'
week_user_feat_path = data_path + 'weekend/week_user_feature.csv'
user_feat_path = data_path + 'user_feature.csv'


# 1. 将 CSV 文件转换为 Numpy 格式
def csv_to_npy(csv_path, npy_path):
    data = pd.read_csv(csv_path, sep='\t', header=0).values  # 读取 CSV 文件并转换为 NumPy 数组
    np.save(npy_path, data)  # 保存为 .npy 文件
    print(f"已将 {csv_path} 转换为 {npy_path}")

# work_feat_path = '../../data/Meituan/workday/user_feature.npy'
# week_feat_path = '../../data/Meituan/weekday/user_feature.npy'
# feat_path = '../../data/Meituan/user_feature.npy'

# work_feat_path = '../../data/Taobao/first_seven/user_feature.npy'
# week_feat_path = '../../data/Taobao/last_three/user_feature.npy'
# feat_path = '../../data/Taobao/user_feature.npy'

# work_feat_path = '../../data/ml1m/workday/user_feature.npy'
# week_feat_path = '../../data/ml1m/weekend/user_feature.npy'
# feat_path = '../../data/ml1m/user_feature.npy'

work_feat_path = '../../data/Beauty/workday/user_feature.npy'
week_feat_path = '../../data/Beauty/weekend/user_feature.npy'
feat_path = '../../data/Beauty/user_feature.npy'

csv_to_npy(work_user_feat_path, work_feat_path)
csv_to_npy(week_user_feat_path, week_feat_path)
csv_to_npy(user_feat_path, feat_path)

# 2. 将 TXT 文件转换为稀疏矩阵并保存为 Numpy 格式
def txt_to_npy(txt_path, npy_path):
    data = pd.read_csv(txt_path, sep='\t', header=0)  # 读取 TXT 文件
    data_array = data.values  # 获取文件内容并转换为 NumPy 数组

    # 保存为 .npy 文件
    np.save(npy_path, data_array)  # 保存为密集矩阵形式的 .npy 文件
    print(f"已将 {txt_path} 转换为 {npy_path}")

# work_train_path = '../../data/Meituan/workday/training_list.npy'
# week_train_path = '../../data/Meituan/weekday/training_list.npy'
# work_train_path = '../../data/Taobao/first_seven/training_list.npy'
# week_train_path = '../../data/Taobao/last_three/training_list.npy'
# work_train_path = '../../data/ml1m/workday/training_list.npy'
# week_train_path = '../../data/ml1m/weekend/training_list.npy'
work_train_path = '../../data/Beauty/workday/training_list.npy'
week_train_path = '../../data/Beauty/weekend/training_list.npy'
txt_to_npy(train_path,work_train_path)
txt_to_npy(train_path,week_train_path)


valid = pd.read_csv(valid_path, sep='\t', header=0)  # 读取 TXT 文件
# 将数据转换为字典形式
# validation_dict = {key: [value] for key, value in zip(valid['user_id_new'], valid['wm_poi_id_new'])}
# validation_dict = {key: [value] for key, value in zip(valid['new_user_id'], valid['adgroup_id'])}
validation_dict = {key: [value] for key, value in zip(valid['user_id_new'], valid['item_id_new'])}
# 保存字典为 .npy 文件
# np.save('../../data/Meituan/workday/validation_dict.npy', validation_dict)
# np.save('../../data/Taobao/first_seven/validation_dict.npy', validation_dict)
# np.save('../../data/ml1m/workday/validation_dict.npy', validation_dict)
np.save('../../data/Beauty/workday/validation_dict.npy', validation_dict)

week_valid = pd.read_csv(week_valid_path, sep='\t', header=0)  # 读取 TXT 文件
# 将数据转换为字典形式
# week_validation_dict = {key: [value] for key, value in zip(week_valid['user_id_new'], week_valid['wm_poi_id_new'])}
# week_validation_dict = {key: [value] for key, value in zip(week_valid['new_user_id'], week_valid['adgroup_id'])}
week_validation_dict = {key: [value] for key, value in zip(week_valid['user_id_new'], week_valid['item_id_new'])}

# 保存字典为 .npy 文件
# np.save('../../data/Meituan/weekday/validation_dict.npy', week_validation_dict)
# np.save('../../data/Taobao/last_three/validation_dict.npy', week_validation_dict)
# np.save('../../data/ml1m/weekend/validation_dict.npy', week_validation_dict)
np.save('../../data/Beauty/weekend/validation_dict.npy', week_validation_dict)

# txt_to_npy(test_path, './data/meituan002/ood_weekend_test.npy')
test = pd.read_csv(test_path, sep='\t', header=0)  # 读取 TXT 文件
# 将数据转换为字典形式
# testing_dict = {key: [value] for key, value in zip(test['user_id_new'], test['wm_poi_id_new'])}
# testing_dict = {key: [value] for key, value in zip(test['new_user_id'], test['adgroup_id'])}
testing_dict = {key: [value] for key, value in zip(test['user_id_new'], test['item_id_new'])}

# 保存字典为 .npy 文件
# np.save('../../data/Meituan/workday/testing_dict.npy', testing_dict)
# np.save('../../data/Taobao/first_seven/testing_dict.npy', testing_dict)
# np.save('../../data/ml1m/workday/testing_dict.npy', testing_dict)
np.save('../../data/Beauty/workday/testing_dict.npy', testing_dict)

week_test = pd.read_csv(week_test_path, sep='\t', header=0)  # 读取 TXT 文件
# 将数据转换为字典形式
# week_testing_dict = {key: [value] for key, value in zip(week_test['user_id_new'], week_test['wm_poi_id_new'])}
# week_testing_dict = {key: [value] for key, value in zip(week_test['new_user_id'], week_test['adgroup_id'])}
week_testing_dict = {key: [value] for key, value in zip(week_test['user_id_new'], week_test['item_id_new'])}

# 保存字典为 .npy 文件
# np.save('../../data/Meituan/weekday/testing_dict.npy', week_testing_dict)
# np.save('../../data/Taobao/last_three/testing_dict.npy', week_testing_dict)
# np.save('../../data/ml1m/weekend/testing_dict.npy', week_testing_dict)
np.save('../../data/Beauty/weekend/testing_dict.npy', week_testing_dict)

