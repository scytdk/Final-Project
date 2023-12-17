import scipy.sparse as sp
import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from data_load_gnn import Data
from model import LightGCN
from parsers import parse_args

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
train_data = Data(train_data)

train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
print("Data Loaded.")
print("user num: ", user_num, " item num: ", item_num)


model = LightGCN(user_num, item_num, inter_matrix, args)
model.to(device)
#model.load_state_dict(torch.load('saved_model/saved_model_epoch_9.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    
    epoch_loss = 0
    #train_loader.dataset.neg_sampling()
    for i, batch in enumerate(tqdm(train_loader)):
        users, items, negs = batch
        optimizer.zero_grad()
        loss = model.calculate_loss(users, items, negs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
    print("epoch: ", epoch, " loss: ", epoch_loss)
    
    if (epoch+1) % 3 == 0:
        torch.save(model.state_dict(),'saved_model/saved_model_epoch_'+str(epoch)+'.pt')
        torch.save(optimizer.state_dict(),'saved_model/saved_optim_epoch_'+str(epoch)+'.pt')