import numpy as np
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from scipy import sparse as sp
from dateutil.parser import parse
import torch

class DateSet(object):
    def __init__(self, path):
        self.train_path = "./data/"+ path + '.txt'
        self.path = ''
        user_train = self.data_partition()
        self.time_csv, self.t_map_csv = self.mappingAndSort(user_train)
        self.getItemtime()

    def data_partition(self, ):
        user_train = defaultdict(list)
        user_train_5 = defaultdict(list)
        with open(self.train_path, 'r') as f:
            for line in f.readlines():
                u, i, c, timestamp = line.rstrip().split(' ')
                u = int(u)
                i = int(i)
                c = int(c)
                try:
                    timestamp = int(timestamp)
                except:
                    timestamp = float(timestamp)
                user_train[u].append([i, c, timestamp])
        for u in user_train:
            if len(user_train[u]) > 4:
                user_train_5[u] = user_train[u]
        return user_train_5

    def mappingAndSort(self, user_train):

        user_csv = set()
        item_csv = set()
        cate_csv = set()
        day_csv = set()
        year_csv = set()
        month_csv = set()

        for u, info in user_train.items():
            # l = len(info)
            user_csv.add(u)
            for i in info:
                item_csv.add(i[0])
                cate_csv.add(i[1])
                t = datetime.fromtimestamp(i[2])
                
                day_csv.add(int(t.strftime('%j')))
                year_csv.add(t.year)
                month_csv.add(t.month)
        
        self.user_num, self.item_num,self.cate_num  = len(user_csv) + 1, len(item_csv) + 1, len(cate_csv) + 1
        self.day_num, self.year_num, self.month_num = len(day_csv)+1, len(year_csv)+1, len(month_csv)+1
        
        u_map_csv = dict(zip(user_csv, [i+1 for i in range(len(sorted(user_csv)))]))
        i_map_csv = dict(zip(item_csv, [i+1 for i in range(len(sorted(item_csv)))]))
        c_map_csv = dict(zip(cate_csv, [i+1 for i in range(len(sorted(cate_csv)))]))

        d_map_csv = dict(zip(day_csv, [i+1 for i in range(len(sorted(day_csv)))]))
        self.y_map_csv = dict(zip(year_csv, [i+1 for i in range(len(sorted(year_csv)))]))
        m_map_csv = dict(zip(month_csv, [i+1 for i in range(len(sorted(month_csv)))]))

        self.User_train = defaultdict(list)
        self.User_train_cate = defaultdict(list)
        for u, info in user_train.items():
            sorted_info = sorted(info, key=lambda x: x[2])
            sorted_info = list(map(lambda x: [i_map_csv[x[0]], [self.y_map_csv[datetime.fromtimestamp(x[2]).year], 
                                               m_map_csv[datetime.fromtimestamp(x[2]).month],
                                               d_map_csv[int(datetime.fromtimestamp(x[2]).strftime('%j'))]]], sorted_info))
            sorted_info_cate = sorted(info, key=lambda x: x[2])
            sorted_info_cate = list(map(lambda x: [c_map_csv[x[1]], [self.y_map_csv[datetime.fromtimestamp(x[2]).year], 
                                               m_map_csv[datetime.fromtimestamp(x[2]).month],
                                               d_map_csv[int(datetime.fromtimestamp(x[2]).strftime('%j'))]]], sorted_info_cate))

            self.User_train[u_map_csv[u]] = sorted_info
            self.User_train_cate[u_map_csv[u]] = sorted_info_cate


        print("data processing done...")
        return 1, 2

    def split_train_and_test(self, ):
        user_train = {}
        user_valid = {}
        user_test = {}
        user_test_all = {}
        for user in self.User_train:
            nfeedback = len(self.User_train[user])
            if nfeedback < 3:
                user_train[user] = self.User_train[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = self.User_train[user][:-2]
                user_valid[user] = []
                user_valid[user].append(self.User_train[user][-2])
                user_test[user] = []
                user_test[user].append(self.User_train[user][-1])
                user_test_all[user] = self.User_train[user][:-1]


        return [user_train, user_valid, user_test, self.user_num, self.item_num, self.year_num, self.month_num, self.day_num,user_test_all]

    def split_cate_train_and_test(self, ):
        user_train = {}
        user_valid = {}
        user_test = {}
        user_test_all = {}
        for user in self.User_train_cate:
            nfeedback = len(self.User_train_cate[user])
            if nfeedback < 3:
                user_train[user] = self.User_train_cate[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = self.User_train_cate[user][:-2]
                user_valid[user] = []
                user_valid[user].append(self.User_train_cate[user][-2])
                user_test[user] = []
                user_test[user].append(self.User_train_cate[user][-1])
                user_test_all[user] = self.User_train_cate[user][:-1]

        return [user_train, user_valid, user_test,self.cate_num,user_test_all]

    def getItemtime(self):
        adj_mat = sp.dok_matrix((self.item_num, self.year_num + self.month_num + self.day_num), dtype=np.float32)
        for u, info in self.User_train.items():
            for i in info:
                adj_mat[i[0], i[1][0]] += 1
                adj_mat[i[0], i[1][1] + self.year_num] += 1
                adj_mat[i[0], i[1][2] + self.year_num + self.month_num] += 1

        adj_mat = adj_mat.tocsr()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        adj_mat = d_mat.dot(adj_mat)
        self.adj_mat = self._convert_sp_mat_to_sp_tensor(adj_mat)

    #时间与项目进行卷积
    def UCGraph(self,data):
        adj_mats = sp.dok_matrix((self.user_num, self.cate_num), dtype=np.float32)
        for u, info in data.items():
            for i in info:
                adj_mats[u, i[0]] = 1

        adj_mat = sp.dok_matrix((self.user_num + self.cate_num, self.user_num + self.cate_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = adj_mats.tolil()
        adj_mat[:self.user_num, self.user_num:] = R
        adj_mat[self.user_num:, :self.user_num] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()
        norm_adj_mat = normalized_adj_single(adj_mat)
        
        norm_time_adj = norm_adj_mat.tocsr()
        norm_time_adj = self._convert_sp_mat_to_sp_tensor(norm_time_adj).float()
        return norm_time_adj
    

    def UIGraph(self,data):
        adj_mats = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
        for u, info in data.items():
            for i in info:
                adj_mats[u, i[0]] = 1
        adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = adj_mats.tolil()
        adj_mat[:self.user_num, self.user_num:] = R
        adj_mat[self.user_num:, :self.user_num] = R.T
        adj_mat = adj_mat.todok()
        
#         adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

#         rowsum = np.array(adj_mat.sum(axis=1))
#         d_inv = np.power(rowsum, -1).flatten()
#         d_inv[np.isinf(d_inv)] = 0.
#         d_mat = sp.diags(d_inv)

#         norm_time_adj = d_mat.dot(adj_mat)
#         norm_time_adj = norm_time_adj.dot(d_mat)
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            np.seterr(divide='ignore', invalid='ignore')
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()
        norm_adj_mat = normalized_adj_single(adj_mat)
        
        norm_time_adj = norm_adj_mat.tocsr()
        norm_time_adj = self._convert_sp_mat_to_sp_tensor(norm_time_adj).float()
        return norm_time_adj

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def date_conversation(self,year, day):
        year = int(year)
        day = int(day)
        first_day = datetime(year, 1, 1)
        wanted_day = first_day + timedelta(day - 1)
        wanted_day = datetime.strftime(wanted_day, '%Y%m%d')
        return wanted_day

    def get_year_key(self,value):
        return [k for k, v in self.y_map_csv.items() if v == value]

    def time_int(self,user_train,user_valid,args):
    # 购买时间序列
        his_time = np.zeros([len(user_train) + 1, args.maxlen])
        for i, row in enumerate(user_train):
            for j, val in enumerate(user_train[row]):
                if (args.maxlen - len(user_train[row]) + j) >= 0:
                    his_time[row][args.maxlen - len(user_train[row]) + j] = self.date_conversation(
                        self.get_year_key(user_train[row][j][1][0])[0], user_train[row][j][1][2])
    # 推荐时间
        user_rec = np.zeros([len(user_train) + 1, 1])
        for i, row in enumerate(user_valid):
            if len(user_valid[row]) > 0:
                user_rec[row][0] = self.date_conversation(self.get_year_key(user_valid[row][0][1][0])[0], user_valid[row][0][1][2])
    #时间差
        time_int = np.zeros([len(user_train) + 1, args.maxlen])
        for i, row in enumerate(user_rec):
            if row > 0:
                a = user_rec[i]
                a = parse(str(int(a[0])))
                for j, num in enumerate(his_time[i]):
                    b = num
                    if b > 0:
                        b = parse(str(int(b)))
                        day = (a - b).days
                        if day > 1024:
                            day = 1024
                        if day < 0:
                            day = 0
                        time_int[i][j] = day
        print("time interval processing done...")
        return time_int
