import scipy.sparse as sp
import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from data_load_gnn import Data
from model import LightGCN
from parsers import parse_args
import pickle
import csv
import torch.nn.functional as F
args = parse_args()

#hyperparameters
lr = args.lr
decay = args.decay
batch_size = args.batch
epochs = args.epoch
device = torch.device(args.device)
#load train dataset
inter_matrix = sp.load_npz("inter_matrix.npz")

train_data = sp.load_npz("total_data.npz")
user_num, item_num = train_data.shape[0], train_data.shape[1]
print("Data Loaded.")
print("user num: ", user_num, " item num: ", item_num)


model = LightGCN(user_num, item_num, inter_matrix, args)
model.to(device)
model.load_state_dict(torch.load('saved_model/saved_model_epoch_32.pt'))
user_embeddings, item_embeddings = model.forward()
user_embeddings = F.normalize(user_embeddings)
item_embeddings = F.normalize(item_embeddings)

with open("users2.pickle", "rb") as file:
    users = pickle.load(file)
file.close()
with open("item2.pickle", "rb") as file:
    items = pickle.load(file)
file.close()

datas = [['id', 'target']]
cur = 0
with open("kkbox-music-data/test.csv", encoding="utf-8") as f:
    for rows in csv.reader(f):
        id, user, item = rows[0], rows[1], rows[2]
        if id == 'id':
            continue
        if user not in users or item not in items:
            datas.append([cur, 0.5])
            cur += 1
        else:
            target = torch.dot(user_embeddings[users[user]], item_embeddings[items[item]])
            datas.append([cur, (1 + target.item()) / 2])
            cur += 1
f.close()

import csv
 
with open('output.csv', mode='w', encoding='utf-8', newline="") as file:
    writer = csv.writer(file)
    for row in datas:
        writer.writerow(row)
file.close()