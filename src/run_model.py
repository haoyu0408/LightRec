from model import *
#from test_new import *
import pickle
from params_new import *
from load_dataset import *
import numpy as np
import scipy.io as sio
from ut import IO, Eval, Misc
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    model_pickle = open("./RCE/data/ml-10m/ml-10m32.pkl",'rb')
    user_for_test = sio.loadmat("./RCE/data/ml-10m/user.mat")['user'][0]
    params = pickle.load(model_pickle, encoding='latin1')
    rerank_user_embeddings2 = params[0]
    rerank_item_embeddings2 = params[1]
    data_generator = Data(train_file=DIR+'train.mat', test_file=DIR+'test.mat',batch_size=BATCH_SIZE,num_neg=NUM_NEG)
    USER_NUM, ITEM_NUM = data_generator.get_num_users_items()
#    rerank_user_embeddings1 = np.tanh(np.matmul(params[0], params[2]))
#    rerank_user_embeddings2 = np.tanh(np.matmul(rerank_user_embeddings1, params[3]))
#    rerank_item_embeddings1 = np.tanh(np.matmul(params[1], params[4]))
#    rerank_item_embeddings2 = np.tanh(np.matmul(rerank_item_embeddings1, params[5]))
    params = [rerank_user_embeddings2,rerank_item_embeddings2]
    RCE = model(n_users=USER_NUM, n_items=ITEM_NUM, emb_dim=EMB_DIM,
                     lr=LR, decay=DECAY, batch_size=BATCH_SIZE, num_neg = NUM_NEG, num_codebook = NUM_CODEBOOK, \
                         rnn_model = 'res', att_model = 'bilinear', codebook_size = CODEBOOK_SIZE, \
                             is_quantization = True, is_distilling = True, params = params)
    print(RCE.model_name)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    #print(sess.run(RCE.names))
    #best = 0
    iteration = int(np.floor(np.sum(np.sum(data_generator.R))/BATCH_SIZE))+1
    loss_list = []
    
    for epoch in range(N_EPOCH):

        users_set, pos_items_set, neg_items_set = data_generator.sample()
        #users_set, pos_items_set, neg_items_set = data_generator.negative_sample(sess, RCE)
        #users_set, pos_items_set, neg_items_set = data_generator.new_negative_sample(sess, model, 256)
        

        loss_all = 0
        #loss1 = 0
        for itera in range(iteration):
            if itera == iteration-1:
                reminder = BATCH_SIZE-int(np.sum(np.sum(data_generator.R))%BATCH_SIZE)
                users_hold = list(users_set[itera*BATCH_SIZE:])+list(users_set[0:reminder])
                pos_items_hold = list(pos_items_set[itera*BATCH_SIZE:])+list(pos_items_set[0:reminder])
                temp = np.array(neg_items_set).reshape((NUM_NEG,-1))
                temp1 = temp[:,itera*BATCH_SIZE:]
                temp2 = temp[:,0:reminder]
                neg_items_hold = list(np.squeeze(temp1.reshape((1,-1))))+list(np.squeeze(temp2.reshape((1,-1))))
                _, loss_value = sess.run([RCE.updates, RCE.loss], \
                    feed_dict={RCE.users: users_hold, RCE.pos_items: pos_items_hold, \
                        RCE.neg_items: neg_items_hold, RCE.t: 0.9})
            else:
                users_hold = users_set[itera*BATCH_SIZE:(itera+1)*BATCH_SIZE]
                pos_items_hold = pos_items_set[itera*BATCH_SIZE:(itera+1)*BATCH_SIZE]
                temp = np.array(neg_items_set).reshape((NUM_NEG,-1))
                temp1 = temp[:,itera*BATCH_SIZE:(itera+1)*BATCH_SIZE]
                neg_items_hold = list(np.squeeze(temp1.reshape((1,-1))))
                _, loss_value = sess.run([RCE.updates, RCE.loss], \
                    feed_dict={RCE.users: users_hold, RCE.pos_items: pos_items_hold, \
                        RCE.neg_items: neg_items_hold, RCE.t: 0.9})
            loss_all += loss_value
            #loss1 += loss_quanti
        print('Epoch %d training loss %f' % (epoch,loss_all))
        #print(loss1)
        if epoch%10 == 0:
            U,V = sess.run([RCE.user_test, RCE.item_test])
            np.save("U_model.npy", U)
            np.save("V_model.npy", V)
            m = Eval.evaluate_item(data_generator.R[user_for_test,:], data_generator.test[user_for_test,:], U[user_for_test,:], V)
            print(Eval.format(m))
            # topk_mat = Eval.topk_search(data_generator.R, U, V, 200)
            # rerank_mat = np.zeros((USER_NUM,100))
            # for u in range(USER_NUM):
            #     user = u
            #     item = topk_mat[u,:]
            #     rerank_score = sess.run([RCE.rerank_score], feed_dict={RCE.users:[u],RCE.pos_items:np.squeeze(item)})
            #     rerank_score = np.squeeze(rerank_score[0])
            #     #print(rerank_score[0])
            #     #print(rerank_score[0].shape)
            #     rerank_mat[u,:] = item[(-rerank_score).argsort()[:100]]
            #     #rerank_mat[u,:] = item[np.argpartition(rerank_score, -200)[-100:]]
            # rerank = Eval.evaluate_topk(data_generator.R, data_generator.test, rerank_mat, 100)
            # print(Eval.format(rerank))


if __name__ == '__main__':
    main()
