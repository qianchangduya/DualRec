import numpy as np
import torch.utils.data as data
import scipy.sparse as sp

def feature_load(user_feat_path, item_feat_path):

    user_feature = np.load(user_feat_path, allow_pickle=True)
    item_feature = np.load(item_feat_path, allow_pickle=True)
    return user_feature, item_feature



def data_load(train_path, valid_path, test_path, dataset=None):

    train_list = np.load(train_path, allow_pickle=True)
    valid_dict = np.load(valid_path, allow_pickle=True)
    test_dict = np.load(test_path, allow_pickle=True)




    uid_max = 0
    iid_max = 0
    train_dict = {}
    for entry in train_list:
        user, item = entry
        if user not in train_dict:
            train_dict[user] = []
        train_dict[user].append(item)
        if user > uid_max:
            uid_max = user
        if item > iid_max:
            iid_max = item


    valid_list = valid_dict.tolist()
    test_list = test_dict.tolist()


    if dataset == 'synthetic':
        n_users = max(uid_max + 1, 1000)
        n_items = max(iid_max + 1, 1000)
    elif dataset == 'meituan':
       n_users = max(uid_max + 1, 2145)
       n_items = max(iid_max + 1, 7189)
    elif dataset == 'Taobao':
        n_users = max(uid_max + 1, 20575)
        n_items = max(iid_max + 1, 112342)
    elif dataset == 'ml1m':
        n_users = max(uid_max + 1, 1217)
        n_items = max(iid_max + 1, 3500)
    elif dataset == 'Beauty':
        n_users = max(uid_max + 1, 3902)
        n_items = max(iid_max + 1, 12844)
    else:
        n_users = max(uid_max + 1, 25768)
        n_items = max(iid_max + 1, 3044)
    print(f'n_users: {n_users}')
    print(f'n_items: {n_items}')

    valid_list = np.array(valid_list)
    test_list = np.array(test_list)


    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))


    valid_x_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    test_x_list = train_list

    test_x_data = sp.csr_matrix((np.ones_like(test_x_list[:, 0]),
                 (test_x_list[:, 0], test_x_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))


    return train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items


