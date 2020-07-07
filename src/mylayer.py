import tensorflow as tf
import numpy as np
import pickle
from scipy.sparse import csr_matrix, eye, find
from sklearn.cluster import KMeans

class mylayer(object):
    def __init__(self, num_neg, emb_dim, num_codebook, batch_size, decay, codebook_size, params):
        self.num_neg = num_neg
        self.codebook = []
        self.emb_dim = emb_dim
        self.num_codebook = num_codebook
        self.batch_size = batch_size
        self.decay = decay
        self.codebook_size = codebook_size
        self.params = params

        self.W = []
        self.M = []
        self.res = []
        self.proj = []
        self.rnn1 = tf.get_variable('rnn1',shape=[self.emb_dim, self.emb_dim],\
                                  initializer = tf.glorot_normal_initializer())
        self.rnn2 = tf.get_variable('rnn2',shape=[self.emb_dim, self.emb_dim],\
                                  initializer = tf.glorot_normal_initializer())
        label = []
        centers = []
        for i in range(self.num_codebook):
#            x = self.params
#            if i>0:
#                x = x-centers[i-1][label[i-1],:]
            self.W.append(tf.get_variable('W_%d' %i,shape=[self.emb_dim, self.emb_dim],\
                                  initializer = tf.glorot_normal_initializer()))
            self.M.append(tf.get_variable('M_%d' %i,shape=[self.emb_dim, int(self.emb_dim/4)],\
                                  initializer = tf.glorot_normal_initializer()))
            self.res.append(tf.get_variable('res_%d' %i,shape=[self.emb_dim, self.emb_dim],\
                                  initializer = tf.glorot_normal_initializer()))
#            temp = KMeans(n_clusters=256).fit(x)
#            centers.append(temp.cluster_centers_)
#            label.append(temp.labels_)
#            self.codebook.append(tf.Variable(temp.cluster_centers_))
            self.codebook.append(tf.get_variable('codebook_%d' %i,shape=[self.codebook_size[i], self.emb_dim],\
                                  initializer = tf.glorot_normal_initializer()))
            self.proj.append(tf.get_variable('proj_%d' %i,shape=[self.emb_dim, self.codebook_size[i]],\
                                  initializer = tf.glorot_normal_initializer()))
        self.batch_norm = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    def get_var_list(self):
        return [self.rnn1,self.rnn2]\
                +self.codebook+self.W+self.res+self.M+self.proj

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        regularizer = tf.nn.l2_loss(users)+ (tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items))
        regularizer = regularizer/self.batch_size
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_mean(maxi)) + self.decay * regularizer
        return loss

    def hinge_loss(self, users, pos_items, neg_items, margin):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        regularizer = tf.nn.l2_loss(users)+ (tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items))
        regularizer = regularizer/self.batch_size
        maxi = tf.maximum(0., tf.nn.sigmoid(neg_scores)-tf.nn.sigmoid(pos_scores)+margin)
        loss = tf.reduce_mean(maxi)+self.decay * regularizer
        return loss

    def multi_bpr_loss(self, users, pos_items, neg_items):
        regularizer = tf.nn.l2_loss(users)+ (tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items))
        regularizer = regularizer/self.batch_size
        users = tf.tile(users,[self.num_neg,1])
        pos_items = tf.tile(pos_items,[self.num_neg,1])
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        y_ = pos_scores-neg_scores
        #y_ = tf.reshape(y_,[self.num_neg,-1])
        #weight = tf.nn.softmax(tf.negative(y_),0)
        #maxi = tf.reduce_sum(tf.multiply(tf.stop_gradient(weight),tf.log(tf.nn.sigmoid(y_))),0)
        #maxi = tf.log(tf.nn.sigmoid(y_))
        maxi = tf.log_sigmoid(y_)
        loss = tf.negative(tf.reduce_mean(maxi)) + self.decay * regularizer
        return loss

    # def distilling_loss(self, users, pos_items, neg_items, pos_items_, neg_items_):
        # #users:batch*dim,pos_items:batch*dim,neg_items:(num_neg*batch)*dim
        # #pos_items_:batch*dim,neg_items_:(num_neg*batch)*dim
        # pos_scores = tf.reshape(tf.reduce_sum(tf.multiply(users, pos_items), axis=1),[1,-1])
        # neg_items = tf.reshape(neg_items, [self.num_neg,-1,self.emb_dim])
        # neg_scores = tf.reduce_sum(users*neg_items, axis=2)
        # quantization_score = tf.concat([pos_scores,neg_scores],axis=0)
        # quantization_score_pro = tf.nn.softmax(quantization_score)
        
        # pos_scores_ = tf.reshape(tf.reduce_sum(tf.multiply(users, pos_items_), axis=1),[1,-1])
        # neg_items_ = tf.reshape(neg_items_, [self.num_neg,-1,self.emb_dim])
        # neg_scores_ = tf.reduce_sum(users*neg_items_, axis=2)
        # non_quantization_score = tf.concat([pos_scores_,neg_scores_],axis=0)
        # non_quantization_score_pro = tf.nn.softmax(non_quantization_score)
        # maxi = tf.reduce_sum(tf.stop_gradient(non_quantization_score_pro)*tf.log(quantization_score_pro+1e-10), axis=0)
        # loss = tf.negative(tf.reduce_mean(maxi))
        # return loss
        
    # def dot_distilling(self, users, pos_items, neg_items, users_, pos_items_, neg_items_):
        # users = tf.tile(users,[self.num_neg,1])
        # users_ = tf.tile(users_,[self.num_neg,1])
        # pos_items = tf.tile(pos_items,[self.num_neg,1])
        # pos_items_ = tf.tile(pos_items_,[self.num_neg,1])
        # non_quanti_pos_score = tf.reduce_sum(tf.multiply(users_, pos_items_), axis=1)
        # quanti_pos_score = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        # non_quanti_neg_score = tf.reduce_sum(tf.multiply(users_, neg_items_), axis=1)
        # quanti_neg_score = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # loss = tf.nn.l2_loss(non_quanti_pos_score-quanti_pos_score)
        # #+tf.nn.l2_loss(non_quanti_neg_score-quanti_neg_score)
        # return loss
        
    def quanti_topk(self, users, topk_item_quanti, users_, topk_item):
        users_quanti = tf.tile(users,[self.num_neg,1])
        users_non_quanti = tf.tile(users_,[self.num_neg,1])
        users_quanti_score = tf.reduce_sum(tf.multiply(users_quanti, topk_item_quanti), axis=1)
        users_non_quanti_score = tf.reduce_sum(tf.multiply(users_quanti, topk_item), axis=1)
        users_quanti_score = tf.reshape(users_quanti_score,[self.num_neg,-1])
        users_non_quanti_score = tf.reshape(users_non_quanti_score,[self.num_neg,-1])
        #loss = tf.nn.l2_loss(tf.nn.softmax(users_non_quanti_score,dim=0)-tf.nn.softmax(users_quanti_score,dim=0))
        maxi = tf.reduce_sum(tf.nn.softmax(users_non_quanti_score/0.5,dim=0)*tf.nn.log_softmax(users_quanti_score/0.5,dim=0), axis=0)
        loss = tf.negative(tf.reduce_mean(maxi))
        return loss
#        topk_item_quanti = tf.reshape(topk_item_quanti, [self.num_neg,-1,self.emb_dim])
#        topk_item_quanti_score = tf.reduce_sum(users*topk_item_quanti, axis=2)
#        #topk_item_quanti_score_pro = tf.nn.softmax(topk_item_quanti_score)
#        
#        topk_item = tf.reshape(topk_item, [self.num_neg,-1,self.emb_dim])
#        topk_item_score = tf.reduce_sum(users_*topk_item, axis=2)
#        topk_item_score_pro = tf.nn.softmax(topk_item_score,axis=0)
#        maxi = tf.reduce_sum(tf.stop_gradient(topk_item_score_pro)*tf.nn.log_softmax(topk_item_quanti_score,axis=0), axis=0)
#        loss = tf.negative(tf.reduce_mean(maxi))
#        return loss

    def sample_gumbel(self, shape, eps=1e-20):
        U = tf.random_uniform(shape,minval=0,maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax( y / temperature)

    def gumbel_softmax(self, logits, temperature, hard=False):
        #y = self.gumbel_softmax_sample(logits, temperature)
        y = tf.nn.softmax( logits / temperature)
        y = tf.stop_gradient(tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)-y)+y
        if hard:
            y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
            y = y_hard
        return y

    def double_(self, x, w, y):
        return tf.matmul(tf.matmul(x, w), y, transpose_a=False, transpose_b=True)

    def M_distance(self, x, m, y):
        M = tf.matmul(m,m,transpose_a=False, transpose_b=True)
        return self.double_(x, M, y)

    def euclidean_distance(self, x, y):
        '''x:m*k
        y:n*k
        output:|x1-y1...x1-yn
               
               xm-y1...xm-yn|_F^2'''
        m = tf.shape(x)[0]
        #k = tf.shape(x)[1]
        n = tf.shape(y)[0]
        x_x = tf.matmul(tf.reshape(tf.reduce_sum(tf.multiply(x, x), 1), [-1,1]), tf.ones([1, n]))
        y_y = tf.matmul(tf.ones([m, 1]), tf.reshape(tf.reduce_sum(tf.multiply(y, y), 1), [1,-1]))
        x_y = tf.matmul(x, y, transpose_a=False, transpose_b=True)
        #return tf.exp(-(x_x+y_y-2*x_y))
        return -(x_x+y_y-2*x_y)

    def core_model(self, input_, rnn_model, att_model, t, is_training=True):
        x = input_
        output = []
        for i in range(self.num_codebook):
            if i>0:
                if rnn_model == 'res':
                    x = x - output[i-1]
                elif rnn_model == 'rnn':
                    #x = tf.matmul(x,self.res[i-1])-tf.matmul(output[i-1],self.res[i])
                    x = tf.matmul(x-output[i-1],self.rnn1)
            if att_model == 'factorization_bilinear':
                x_ = self.gumbel_softmax(self.M_distance(x, self.M[i], self.codebook[i]), t, hard=not is_training)
            elif att_model == 'bilinear':
                x_ = self.gumbel_softmax(self.double_(x, self.W[i], self.codebook[i]), t, hard=not is_training)
            elif att_model == 'dot':
                x_ = self.gumbel_softmax(tf.matmul(x, self.codebook[i], transpose_a=False, transpose_b=True), t, hard=not is_training)
            elif att_model == 'scale_dot':
                x_ = self.gumbel_softmax(tf.matmul(x, self.codebook[i], transpose_a=False, transpose_b=True)\
                    /tf.sqrt(tf.cast(self.emb_dim,dtype=tf.float32)), t, hard=not is_training)
            elif att_model == 'euclidean':
                x_ = self.gumbel_softmax(self.euclidean_distance(x, self.codebook[i]), t, hard=not is_training)
            elif att_model == 'kd':
                x_ = self.gumbel_softmax(tf.matmul(x, self.proj[i]), t, hard=not is_training)
            else:
                raise ValueError('Not supported attention mode')
            output.append(tf.matmul(x_, self.codebook[i]))
        return tf.add_n(output)

    def res_model(self, input_, t):
        x_ = self.gumbel_softmax(self.double_(input_, self.W[0], self.codebook[0]), t, hard=False)
        return tf.matmul(x_, self.codebook[0])
        # if is_training:
        #     output_ = tf.add_n(output)+input_
        # else:
        #     output_ = tf.add_n(output)
        # return output_

    def item_index(self, x):
        index_ = []
        for i in range(self.num_codebook):
            temp = self.gumbel_softmax(self.double_(x, self.W[i], self.codebook[i]) ,1e-20, hard=True)
            index_.append(tf.argmax(input=temp, dimension=1))
        return index_
