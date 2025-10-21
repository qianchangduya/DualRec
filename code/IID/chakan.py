import pandas as pd
import numpy as np

# 加载预测结果
pred_list = np.load('output/iid_users_predictions_answers.npy')

# 创建 DataFrame
df = pd.DataFrame(pred_list)
df.columns = [f'rec_{i+1}' for i in range(pred_list.shape[1])]

print(f"预测结果数据框:\n{df.head()}")

# 查看基本信息
print(f"预测结果形状: {pred_list.shape}")  # (用户数, 推荐物品数)
print(f"前5个用户的推荐列表:\n{pred_list[:5]}")