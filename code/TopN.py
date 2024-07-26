import sys
import numpy as np
from sklearn import preprocessing
#from data_utils import DataUtils
import random
import math
import os
from sklearn import metrics
import time
#from io import open
from sklearn.preprocessing import normalize
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support
import heapq

from sklearn.cluster import KMeans
from sklearn import metrics

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def top_N(test_u, test_v, test_rate, vectors, top_n,rho):
    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    stime = time.time()
    size = 0
    print("start computing metrics")
    print("%d pairs"%(len(test_rate)))
    for u in test_u:
        recommend_dict = {}
        if vectors.get(int(u)) is None:
            continue
        U = np.array(vectors[int(u)])
        for v in test_rate[u].keys():
            if vectors.get(int(v)) is None:
                continue
            V = np.array(vectors[int(v)])
            pre = U.dot(V.T)
            recommend_dict[int(v)] = sigmoid(float(pre) * rho)

        # tmp_r = sorted(recommend_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(recommend_dict),top_n)]
        # tmp_t = sorted(test_rate[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(test_rate[u]),top_n)]
        # tmp_r_list = []
        # tmp_t_list = []
        # for (item, rate) in tmp_r:
        #     tmp_r_list.append(item)

        # for (item, rate) in tmp_t:
        #     tmp_t_list.append(item)
        size += len(recommend_dict)
        if len(recommend_dict)==0:
            continue
        tmp_r_list = heapq.nlargest(min(len(recommend_dict),top_n), recommend_dict, key=recommend_dict.get)
        tmp_t_list = heapq.nlargest(min(len(test_rate[int(u)]),top_n), test_rate[int(u)], key=test_rate[int(u)].get)
        
        pre, rec = precision_and_recall(tmp_r_list,tmp_t_list)
        ap = AP(tmp_r_list,tmp_t_list)
        rr = RR(tmp_r_list,tmp_t_list)
        ndcg = nDCG(tmp_r_list,tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)

    etime = time.time()
    print("eval time: ", etime-stime)
    print("Size is ", size)


    precison = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    print(precison, recall)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1,map,mrr,mndcg

def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i+1
        dcg += 1/ math.log(rank+1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i+1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0

def RR(ranked_list, ground_list):

    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def precision_and_recall(ranked_list,ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list))
    rec = hits/(1.0 * len(ground_list))
    return pre, rec

def read_test_data(test_filename = '../data/bitcoin/bitcoin_topn.txt'):
    print("loading test data...")
    users,items,rates = set(), set(), {}
    with open(test_filename, "r") as fin:
        line = fin.readline()
        while line:
            user, item, rate = line.strip().split(',')
            user = int(user)
            item = int(item)
            if rates.get(user) is None:
                rates[user] = {}
            rates[user][item] = float(rate)
            users.add(user)
            items.add(item)
            line = fin.readline()

    # train_user_item_flag = {}
    # with open(train_filename, "r") as fin:
    #     line = fin.readline()
    #     while line:
    #         user, item, rate = line.strip().split()
    #         train_user_item_flag[user+item] = True
    #         line = fin.readline()

    return users, items, rates

def load_embedding(path = '../data/bitcoin/bitcoin_2.emb'):
    with open(path, 'r') as f:
        embeddings = f.readlines()
    origin_embedding = {}
    #embeddings = embeddings[1:-1]
    tot = 0
    for i in range(len(embeddings)):
        if tot == 0:
            tot += 1
            continue
        embeddings[i] = list(map(float, embeddings[i].split(' ')))
        origin_embedding[int(embeddings[i][0])] = embeddings[i][1:]
    
    return origin_embedding

def load_ctdne_embedding(path = '../data/bitcoin/bitcoin_ctdne_embedding'):
    with open(path, 'rb') as f:
        embedding = pickle.load(f)
    return embedding

def evaluate(embeddings = None, data = '', rho = 0.001, top_n = 10):
    test_user, test_item, test_rate = read_test_data('../data/' + str(data) + '/'+ str(data) +'_topn.txt')
    print('Start test...')
    embedding = {}
    for i in range(len(embeddings)):
        embedding[i] = embeddings[i]
    print(rho)
    f1, map, mrr, mndcg = top_N(test_user,test_item,test_rate,embedding,top_n,rho)
    print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f' % (round(f1,4), round(map,4), round(mrr,4), round(mndcg,4)))
