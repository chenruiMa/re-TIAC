from os import stat
import numpy as np
import torch
import sys
import torch.nn as nn
from utils import *
import math
import torch.nn.functional as F

FLOAT_MIN = -sys.float_info.max


class HTP(torch.nn.Module):
    def __init__(self, user_num, item_num,cate_num, yearnum, monthnum, daynum, args,time_int,item_time_matirx,norm_adj,uc_adj,ui_adj_test,uc_adj_test):
        super(HTP, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.year_num = yearnum
        self.day_num = daynum
        self.month_num = monthnum
        self.time_int = time_int


        self.args = args

        self.dev = args.device
        self.beta = args.beta
        self.item_time_matrix = item_time_matirx.to(self.dev)
        #graph
        self.weight_size = [args.hidden_units, args.hidden_units, args.hidden_units, args.hidden_units]
        dropout_list = [0.1, 0.1, 0.1, 0.1]
        self.norm_adj = norm_adj.to(self.dev)
        self.uc_adj = uc_adj.to(self.dev)
        self.ui_adj_test = ui_adj_test.to(self.dev)
        self.uc_adj_test = uc_adj_test.to(self.dev)

        self.n_layers = self.args.gcn_layer
        #ui_graph
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.weight_size = [args.hidden_units] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))
        #uc_graph
        self.dropout_list_c = nn.ModuleList()
        self.GC_Linear_list_c = nn.ModuleList()
        self.Bi_Linear_list_c = nn.ModuleList()
        self.weight_size = [args.hidden_units] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list_c.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.Bi_Linear_list_c.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.dropout_list_c.append(nn.Dropout(dropout_list[i]))


        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num, args.hidden_units, padding_idx=0)
        self.category_emb = torch.nn.Embedding(self.cate_num, args.hidden_units, padding_idx=0)
        self.year_emb = torch.nn.Embedding(self.year_num, args.hidden_units, padding_idx=0)
        self.month_emb = torch.nn.Embedding(self.month_num, args.hidden_units, padding_idx=0)
        self.day_emb = torch.nn.Embedding(self.day_num, args.hidden_units, padding_idx=0)
        self.mu_all = torch.nn.Embedding(self.user_num, args.maxlen)
        self.sigma_all = torch.nn.Embedding(self.user_num, args.maxlen)

        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.year_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.month_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.day_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # position encoding
        self.abs_pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.GRU = torch.nn.GRU(input_size=args.hidden_units, hidden_size=args.hidden_units,
                                num_layers=1, batch_first=True)
        self.softmax = torch.nn.Softmax(dim=-1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def seq2feats(self, user_ids, log_seqs, year, month, day,state):
        train = True
        if log_seqs.shape[0] == 1:
            train = False

        if state == '1':
            ui_adj = self.norm_adj
            uc_adj = self.uc_adj
        else:
            ui_adj = self.ui_adj_test
            uc_adj = self.uc_adj_test
        # item embedding
        user_emb ,items_emb = self.UI(ui_adj)
        user_emb_c, categry_emb = self.UC(uc_adj)
        con_loss2 = self.SSL_ci(user_emb , user_emb_c)
        
        # items_emb = self.item_emb.weight
        seqs = items_emb[log_seqs].to(self.dev)
        # seqs = items_emb(torch.LongTensor(log_seqs).to(self.dev))
        
        seqs = seqs * self.item_emb.embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)

        # position encoding
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_embs = self.abs_pos_emb(positions)
        abs_pos_embs = self.abs_pos_emb_dropout(abs_pos_embs)

        # time embedding
        year_embs = self.year_emb(torch.LongTensor(year).to(self.dev))
        month_embs = self.month_emb(torch.LongTensor(month).to(self.dev))
        day_embs = self.day_emb(torch.LongTensor(day).to(self.dev))

        times_emb = torch.cat((self.year_emb.weight, self.month_emb.weight, self.day_emb.weight), dim=0)
        item_time_embs = torch.sparse.mm(self.item_time_matrix, times_emb)
        if train:
            item_time_emb = item_time_embs[log_seqs]
        else:
            item_time_emb = item_time_embs[log_seqs].unsqueeze(0)
            
        seqs = seqs + abs_pos_embs+item_time_emb
#         seqs = seqs + abs_pos_embs

        year_embs = self.year_emb_dropout(year_embs)
        month_embs = self.month_emb_dropout(month_embs)
        day_embs = self.day_emb_dropout(day_embs)
        
        time_embs = month_embs + day_embs

#         # history time
        history_time_embs = time_embs[:, :self.args.maxlen]       # B * maxlen * d
#         # target time
        perdiction_time_embs = time_embs[:, 1:self.args.maxlen+1]   # B * maxlen * d

        
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)  # B * len
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality  # maxlen
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # compute time interval matrix
        src_time_embs = history_time_embs.unsqueeze(1)  # B * 1 * N * dim
        dst_time_embs = history_time_embs.unsqueeze(2)  # B * N * 1 * dim
        # time_matrices = src_time_embs - dst_time_embs

        Fu, _ = self.GRU(seqs)
        Gu = user_emb[user_ids]
        Gu = Gu.unsqueeze(1)
        
#         con_loss = 0
        int_seq = Fu - seqs
#         Iu,_ = self.GRU(int_seq)

        self.delta_t = torch.Tensor(self.time_int[user_ids]).to(self.dev)
        mu_all = self.mu_all.weight[user_ids].to(self.dev)
        sigma_all = self.sigma_all.weight[user_ids].to(self.dev)
        
        E_recom = self.perdiction_time_process(perdiction_time_embs, history_time_embs, seqs, Fu, attention_mask,mu_all,sigma_all)
        E_recom = self.last_layernorm(E_recom)
        con_loss = self.SSL(Fu , Gu)
        # Fusion
        log_feats = E_recom+self.last_layernorm(Fu)
#         log_feats = E_recom
        #TODO：两个图之间的协同信息需进行进一步探索
        return log_feats, self.beta * con_loss + self.beta * con_loss2, items_emb

    def forward(self, user_ids, log_seqs, year, month, day, pos_seqs, neg_seqs):  # for training
        log_feats , con_loss, items_emb = self.seq2feats(user_ids, log_seqs, year, month, day,'1')
        
        pos_seqs = torch.LongTensor(pos_seqs).to(self.dev)

        pos_embs = self.item_emb(pos_seqs)  # B *  N * d
#         pos_embs = items_emb[pos_seqs]
        neg_seqs = torch.LongTensor(neg_seqs).to(self.dev)
        # print(neg_seqs.shape, neg_seqs.max(), self.item_num, neg_seqs.min(), self.item_emb.weight.shape)
        neg_embs = self.item_emb(neg_seqs)
#         neg_embs = items_emb[neg_seqs]
        pos_logits = (log_feats * pos_embs).sum(dim=-1)#128*25*50   *   128*25*50  =128*25
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, con_loss  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices, year, month, day):  # for inference

        log_feats,con_loss,items_emb = self.seq2feats(user_ids, log_seqs, year, month, day, '0')

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
#         item_embs = items_emb(torch.LongTensor(item_indices).to(self.dev))
        
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)
    def perdiction_time_process(self, per_time_embs, history_time_embs, item_embs, Fu, attention_mask, mu, sigma):
        # 计算时间的影响因子
        src_time_embs = per_time_embs.unsqueeze(1)  # B * 1 * N * dim
        dst_time_embs = history_time_embs.unsqueeze(2)  # B * N * 1 * dim
        time_embs = (src_time_embs - dst_time_embs).sum(-1)  # B * N * N * d

        paddings = torch.ones(time_embs.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(attention_mask, paddings, time_embs)  # enforcing causality
        intent_attention = torch.matmul(Fu, item_embs.permute(0, 2, 1))  # B* N * N
        # print(intent_attention.shape, paddings.shape, attention_mask.shape)
        attn_weights = torch.where(attention_mask, paddings, intent_attention)   # enforcing causality
        intent_attention = self.softmax(attn_weights)

        #核函数变化
        att_weight = intent_attention[:, -1, :]
        pi = 3.14
        # mus = mu.detach().numpy()
        # sigmas = sigma.detach().numpy()
        change = -pi * torch.exp(-(self.delta_t - mu) ** 2 / (2 * sigma ** 2)/1000) / (math.sqrt(2 * pi) * sigma)
        change = change.to(self.dev)
        intent_attentions = att_weight + change
        intent_attentions = intent_attentions.unsqueeze(-1).repeat(1, 1, self.args.maxlen)
        final_intent = self.softmax(intent_attentions)
        W_i = 1
        final_intent = W_i*final_intent
#         final_intent = intent_attention
# #         计算聚合得分
        embs = torch.matmul(final_intent, item_embs)
#         embs = torch.matmul(intent_attention, item_embs)
        
        return embs

    def SSL(self, Fu, Gu):#Fu , Gu
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            corrupted_embedding = corrupted_embedding[:,:,torch.randperm(corrupted_embedding.size()[2])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2), 1)
#             return torch.cosine_similarity(x1, x2, dim=0)
        Fu = Fu[:,-1,:]
        pos = score(Fu, Gu).to(self.dev)
        neg1 = score(Fu, row_column_shuffle(Gu)).to(self.dev)
        one = torch.FloatTensor(pos.shape).fill_(1).to(self.dev)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.mean(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))#将-换成+
        return con_loss

    def SSL_ci(self, u_i, u_c):#Fu , Gu
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2), 1)
#             return torch.cosine_similarity(x1, x2, dim=0)
        pos = score(u_i, u_c).to(self.dev)
        neg1 = score(u_i, row_shuffle(u_c)).to(self.dev)
        one = torch.FloatTensor(pos.shape).fill_(1).to(self.dev)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.mean(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))#将-换成+
        return con_loss

    #NGCF user and item
    def UI(self,adj):
        ego_embeddings = torch.cat((self.user_emb.weight, self.item_emb.weight), dim=0).to(self.dev)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
#             print(side_embeddings.device)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        # all_embeddings = torch.cat(all_embeddings, dim=1)
        all_embeddings = torch.stack(all_embeddings,dim = 1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num], dim=0)
        return u_g_embeddings, i_g_embeddings

    #NGCF user and category
    #TODO：卷积层数需要进行尝试
    def UC(self,adj):
        ego_embeddings = torch.cat((self.user_emb.weight, self.category_emb.weight), dim=0).to(self.dev)
        all_embeddings = [ego_embeddings]
        for i in range(2):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
#             print(side_embeddings.device)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list_c[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list_c[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list_c[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        # all_embeddings = torch.cat(all_embeddings, dim=1)
        all_embeddings = torch.stack(all_embeddings,dim = 1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.user_num, self.cate_num], dim=0)
        return u_g_embeddings, i_g_embeddings

