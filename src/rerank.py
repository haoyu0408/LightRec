# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:02:49 2019

@author: admin
"""
import pickle
import scipy.io as sio
import scipy.sparse as ss
import numpy as np
from ut import IO, Eval, Misc
from load_dataset import *
DIR = 'data/'+'ml-10m'+'/'
model_pickle = open("./RCE/data/ml-10m/ml-10m32.pkl",'rb')
params = pickle.load(model_pickle, encoding='latin1')
#data_generator = Data(train_file=DIR+'train.mat', test_file=DIR+'test.mat',batch_size=1024,num_neg=5)
train_file=DIR+'train.mat'
test_file=DIR+'test.mat'
R = sio.loadmat(train_file)['train'].tocsr()
test = sio.loadmat(test_file)['test'].tocsr()
U = params[0]
V = params[1]
user_for_test = sio.loadmat("./RCE/data/ml-10m/user.mat")['user'][0]
topk_mat = Eval.topk_search(R[user_for_test,:], U[user_for_test,:], V, 200)
#print(topk_mat.shape)
rerank_mat = np.zeros((69662,100))
user_embed = np.load('./RCE/U_model.npy')
item_embed = np.load('./RCE/V_model.npy')
for id_,u in enumerate(user_for_test):
    user = u
    item = topk_mat[id_,:]
    rerank_score = np.matmul(user_embed[u,:], item_embed[np.squeeze(item),:].T)
    rerank_score = np.squeeze(rerank_score)
#    print(rerank_score.shape)
#    print(rerank_mat[u,:].shape)
#    print(item[(-rerank_score).argsort()[:100]].shape)
    rerank_mat[u,:] = item[(-rerank_score).argsort()[:100]]
rerank = Eval.evaluate_topk(R[user_for_test,:], test[user_for_test,:], rerank_mat[user_for_test,:], 100)
print(Eval.format(rerank))