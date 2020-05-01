import tensorflow as tf
import utils as ut
import numpy as np
import pickle
from mylayer import *
from scipy.sparse import csr_matrix, eye, find

class model(object):
    def __init__(self, n_users, n_items, emb_dim, lr, batch_size, decay, num_neg, \
            num_codebook, rnn_model, att_model, codebook_size, is_quantization, is_distilling, params):
        self.model_name = 'RCE'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.decay = decay
        self.num_neg = num_neg
        self.num_codebook = num_codebook
        self.rnn_model = rnn_model
        self.att_model = att_model
        self.codebook_size = codebook_size
        self.is_quantization = is_quantization
        self.is_distilling = is_distilling
        self.params = params
        self.lr = lr
        

        self.users = tf.placeholder(tf.int32)
        self.pos_items = tf.placeholder(tf.int32)
        self.neg_items = tf.placeholder(tf.int32)
        #self.topk_items = tf.placeholder(tf.int32)
        self.t = tf.placeholder(tf.float32)


        self.layer = mylayer(num_neg=self.num_neg, emb_dim=self.emb_dim, num_codebook=self.num_codebook, \
            batch_size=self.batch_size, decay=self.decay, codebook_size=self.codebook_size, params=self.params[1])

        self.user_embeddings = tf.Variable(self.params[0], trainable=True)
        self.item_embeddings = tf.Variable(self.params[1], trainable=False)
        self.user_emb = tf.Variable(self.params[0], trainable=False)
#        self.user_embeddings = tf.get_variable('user_embeddings',shape=[self.n_users, self.emb_dim],\
#                                  initializer = tf.glorot_normal_initializer())
#        self.item_embeddings = tf.get_variable('item_embeddings',shape=[self.n_items, self.emb_dim],\
#                                  initializer = tf.glorot_normal_initializer())

        self.user_test = self.user_embeddings
        self.item_test = self.layer.core_model(self.item_embeddings, self.rnn_model, self.att_model, 1e-20, is_training=False)
        self.itemindex = self.layer.item_index(self.item_embeddings)
        self.sample_rating = tf.matmul(self.user_embeddings, self.layer.codebook[0], transpose_a=False, transpose_b=True)

        # self.var_list = [self.user_embeddings0, self.item_embeddings0, self.w1_user, self.w1_item, self.w2_user, self.w2_item]\
        #     +self.layer.get_var_list()
        # self.var_list = [self.user_embeddings, self.item_embeddings]\
        #     +self.layer.get_var_list()
        
        self.var_list = [self.user_embeddings]\
            +self.layer.get_var_list()
        self.loss = self.model_loss()
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        #self.extra_update_ops = self.layer.batch_norm
        self.updates = self.opt.minimize(self.loss, var_list=self.var_list)
        # self.updates = self.opt.minimize(self.loss, var_list=self.var_list)
    
    def model_loss(self):
        if self.is_quantization:
            u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            u_embeddings_ = tf.nn.embedding_lookup(self.user_emb, self.users)
            pos_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
            neg_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)
            pos_i = self.layer.core_model(pos_i_embeddings, self.rnn_model, self.att_model, self.t, is_training=True)
            neg_i = self.layer.core_model(neg_i_embeddings, self.rnn_model, self.att_model, self.t, is_training=True)
            
            topk_emb = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)
            quanti_topk_emb = self.layer.core_model(topk_emb, self.rnn_model, self.att_model, self.t, is_training=True)
            loss_quanti = self.layer.quanti_topk(u_embeddings, quanti_topk_emb, u_embeddings_, topk_emb)
            if self.num_neg>1:
                if self.is_distilling:
                    loss = self.layer.multi_bpr_loss(u_embeddings, pos_i, neg_i)+(self.full_dot_loss(pos_i,pos_i_embeddings)+self.full_dot_loss(neg_i,neg_i_embeddings))*1e-9#+0.2*loss_quanti
#                    loss = self.layer.multi_bpr_loss(u_embeddings, pos_i, neg_i)+0.01*self.layer.quanti_topk(u_embeddings, quanti_topk_emb, u_embeddings_, topk_emb)+\
#                        0*(tf.nn.l2_loss(pos_i-pos_i_embeddings)+tf.nn.l2_loss(neg_i-neg_i_embeddings))
#                    loss = self.layer.multi_bpr_loss(u_embeddings, pos_i, neg_i)\
#                        +1e-5*self.layer.dot_distilling(u_embeddings, pos_i, neg_i,u_embeddings_,pos_i_embeddings,neg_i_embeddings)
                else:
                    loss = self.layer.multi_bpr_loss(u_embeddings, pos_i, neg_i)
            else:
                loss = self.layer.bpr_loss(u_embeddings, pos_i, neg_i)
                #loss = self.layer.hinge_loss(u_embeddings, pos_i, neg_i, 0.1)
        else:
            u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            pos_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
            neg_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)
            if self.num_neg>1:
                loss = self.layer.multi_bpr_loss(u_embeddings, pos_i_embeddings, neg_i_embeddings)
            else:
                loss = self.layer.bpr_loss(u_embeddings, pos_i_embeddings, neg_i_embeddings)
        return loss
        
    def full_dot_loss(self, item, item_):
        u_full_embeddings = self.user_emb
        u_full_embeddings_quantization = self.user_embeddings
        score = tf.matmul(u_full_embeddings_quantization,item, transpose_a=False, transpose_b=True)
        score_ = tf.matmul(u_full_embeddings, item_, transpose_a=False, transpose_b=True)
        loss = tf.nn.l2_loss(score-score_)
        return loss
