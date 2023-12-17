import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, user_nums, item_nums, inter_matrix, args):
        super(LightGCN, self).__init__()
        self.user_nums = user_nums
        self.item_nums = item_nums
        self.inter_matrix = inter_matrix
        self.args = args
        self.norm = 2
        self.device = torch.device(self.args.device)
        self.gamma = 1e-10
        self.BCEloss = nn.BCELoss()
        self.user_embeddings = nn.Embedding(self.user_nums, self.args.hidden_dim)
        self.item_embeddings = nn.Embedding(self.item_nums, self.args.hidden_dim)

        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

        self.norm_inter_matrix = self.get_norm_adj_mat().to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix(
            (self.user_nums + self.item_nums, self.user_nums + self.item_nums), dtype=np.float32
        )
        inter_M = self.inter_matrix
        inter_M_t = self.inter_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.user_nums), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.user_nums, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        user_embeddings = self.user_embeddings.weight
        item_embeddings = self.item_embeddings.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    

    def EmbLoss(self, *embeddings, require_pow = False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.args.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_inter_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.user_nums, self.item_nums]
        )
        return user_all_embeddings, item_all_embeddings
    
    '''
    def calculate_loss(self, user, item, neg_item):
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = F.normalize(user_embeddings[user])
        i_embeddings = F.normalize(item_embeddings[item])
        neg_embeddings = F.normalize(item_embeddings[neg_item])

        pos_scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        pos_scores = (pos_scores + 1) / 2
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        neg_scores = (neg_scores + 1) / 2
        bpr_loss = self.BPRLoss(pos_scores, neg_scores)

        reg_loss = self.EmbLoss(
            self.user_embeddings.weight[user],
            self.item_embeddings.weight[item],
            self.item_embeddings.weight[neg_item]
        )

        return bpr_loss + self.args.reg_weight*reg_loss
    '''
    def calculate_loss(self, user, item, data):
        user = user.to(self.device)
        item = item.to(self.device)
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = F.normalize(user_embeddings[user])
        i_embeddings = F.normalize(item_embeddings[item])

        pos_scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        pos_scores = (pos_scores + 1) / 2
        data = data.float().to(self.device)
        bce_loss = self.BCEloss(pos_scores, data)

        reg_loss = self.EmbLoss(
            self.user_embeddings.weight[user],
            self.item_embeddings.weight[item]
        )

        return bce_loss + self.args.reg_weight*reg_loss