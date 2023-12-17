import os
import csv
import scipy.sparse as sp
from collections import defaultdict
import numpy as np
import torch.utils.data as datas
import pickle

row, col, data = [], [], []
users, items = defaultdict(int), defaultdict(int)

# train user num: 27113, train item num: 223723
def data_load(file):
    user_lens = 0
    item_lens = 0
    row2, col2, data2 = [], [], []
    with open(file, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for rows in reader:
            user, item, target = rows[0], rows[1], rows[5]
            if user not in users:
                users[user] = user_lens
                user_lens += 1
            if item not in items:
                items[item] = item_lens
                item_lens += 1
            row.append(users[user])
            col.append(items[item])
            data.append(int(target))
            if target == '1':
                row2.append(users[user])
                col2.append(items[item])
                data2.append(int(target))
    f.close()
    inter_matrix = sp.coo_matrix((np.array(data), (np.array(row), np.array(col))), shape=(user_lens, item_lens), dtype=int)
    inter_matrix2 = sp.coo_matrix((np.array(data2), (np.array(row2), np.array(col2))), shape=(user_lens, item_lens), dtype=int)
    sp.save_npz("inter_matrix.npz", inter_matrix2)
    return inter_matrix, user_lens, item_lens

class Data(datas.Dataset):
    def __init__(self, coomat, neg_nums=1):
        self.rows = coomat.row
        self.cols = coomat.col
        self.datas = coomat.data
        self.dokmat = coomat.todok()
        self.neg_nums = neg_nums
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        
    
    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.datas[idx]


inter_matrix, user_lens, item_lens = data_load("kkbox-music-data/train.csv")
sp.save_npz("total_data.npz", inter_matrix)

 
with open("users2.pickle", "wb") as f:
    pickle.dump(users, f)
f.close()

with open("item2.pickle", "wb") as f:
    pickle.dump(items, f)
f.close()