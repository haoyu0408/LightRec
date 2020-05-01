import numpy as np
import random as rd
import heapq
import scipy as sp
import scipy.sparse as ss
import scipy.io as sio

class Data(object):
    def __init__(self, train_file, test_file, batch_size, num_neg):
        self.batch_size = batch_size
        #get number of users and items
        #self.n_users, self.n_items = 0, 0
        self.num_neg = num_neg
        self.train_file = train_file
        self.test_file = test_file
        
        self.R = sio.loadmat(train_file)['train'].tocsr()
        self.test = sio.loadmat(test_file)['test'].tocsr()

        self.n_users, self.n_items = self.R.shape

        print(self.n_users)
        print(self.n_items)

        self.train_items, self.test_set = {}, {}
        for u in range(self.n_users):
            items_train = list(np.where(np.squeeze(self.R[u,:].toarray()))[0])
            #items_test = list(np.where(np.squeeze(self.test[u,:].toarray()))[0])
            self.train_items[u] = items_train
            #self.test_set[u] = items_test

#        self.item_not_in_train = {}
#        for u in range(self.n_users):
#            #pos_items = self.train_items[u]
#            neg_item_u = list(set(range(self.n_items)) - set(self.train_items[u]))
#            self.item_not_in_train[u] = neg_item_u

    def sample(self):
        neg_items = []
        all_item = list(set(range(self.n_items)))
        for _ in range(self.num_neg):
            for u in range(self.n_users):
                pos_items = self.train_items[u]
                num_pos = len(pos_items)
                #neg_item_u = list(set(range(self.n_items)) - set(self.train_items[u]))
                for _ in range(num_pos):
                    while True:
                        a = rd.sample(all_item, 1)
                        if a[0] not in pos_items:
                            neg_items += a
                            break
                #neg_items_u = self.item_not_in_train[u]
                #neg_items += rd.sample(neg_items_u, num_pos)
        temp = np.array(neg_items).reshape((self.num_neg,-1))
        #users, pos_items = np.where(self.R)
        r,s,d = ss.find(self.R)
        rs = np.array([r,s])
        rs = rs[:,rs[0,:].argsort()]
        users = rs[0,:]
        pos_items = rs[1,:]
        idx = np.array(range(len(pos_items)))
        np.random.shuffle(idx)
        users = list(users[idx])
        pos_items = list(pos_items[idx])
        temp = temp[:,idx]
        neg_items = list(np.squeeze(temp.reshape(1,-1)))
        return users, pos_items, neg_items

    
        return users, pos_items, neg_items
                
    def get_num_users_items(self):
        return self.n_users, self.n_items
