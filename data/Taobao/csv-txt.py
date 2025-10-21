import pandas as pd


raw_sample = pd.read_csv('raw_sample.csv')

ad_feature = pd.read_csv('ad_feature.csv')

merged_data = pd.merge(raw_sample, ad_feature, on='adgroup_id', how='inner')


filtered_data = merged_data[merged_data['clk'] == 1]


result_data = filtered_data[['user', 'adgroup_id', 'time_stamp', 'clk', 'price']]


result_data.to_csv('Taobao_data.txt', sep='\t', index=False, header=True)


print("数据处理完成，结果已保存至Taobao.txt")