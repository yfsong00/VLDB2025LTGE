import random
import math
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import time
import pickle
from scipy.linalg import qr
import Incremental
import gc
import sys
import scipy
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

result = []
node_index = 0
edge_set = set()

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def load_edges(file):
    edges = []
    edge_set = set()
    with open(file, 'r') as f:
        tot = 0
        for l in f:
            x,y,t = l.strip().split(',')
            x = int(x)
            y = int(y)
            t = float(t)
            edge_set.add(str(x)+'and'+str(y))
            edges.append((x,y,t))
    return edges

def sample_neg_edges():
    global edge_set
    global node_index
    ok = 1
    #size_epoch += 1
    while ok:
        first_node = random.randint(1,node_index)  # pick a random node
        second_node = random.randint(1,node_index)
        if first_node==second_node or (str(first_node)+'and'+str(second_node)) in edge_set:
            continue
        ok = 0
    n_edge = (first_node, second_node, 0)
    return n_edge

def recompute_embedding(edges, node_index, k = 400, dimension = 32):
    print("Largest Motif Number is " + str(k))
    t = edges[0][2]
    T = edges[-1][2] - t
    label = 0
    Hash = []
    num_per_motif = float(T/k)
    for l in range(len(edges)):
        if edges[l][2] - t > num_per_motif:
            label+=1
            t = edges[l][2]
        Hash.append(label)
    tot = 0
    E_m = scipy.sparse.lil_matrix((int(node_index+1),k+1))
    for edge in edges:
        E_m[(int(edge[0]),Hash[tot])]+=(math.log(max(1.0,float(edge[2]))))
        E_m[(int(edge[1]),Hash[tot])]+=(math.log(max(1.0,float(edge[2]))))
    tot+=1
    for i, row in enumerate(E_m.rows):
        for j in row:
            if E_m[i, j] > 0:  # 确保元素大于零
                E_m[i, j] = np.log(E_m[i, j])
    E_m = scipy.sparse.csr_matrix(E_m)

    del Hash
    gc.collect()

    U, s, V = randomized_svd(E_m, n_components=dimension, n_iter=6)

    del E_m
    gc.collect()

    Sigma = np.diag(s)
    Embedding = np.dot(U,Sigma)
    t2 = time.perf_counter()
    
    return Embedding, V, num_per_motif

def linkpred(pos_edge, embedding, node_index, rho = 0.001):
    global data
    tot = 0

    y_true = []
    y_test = []
    y_test_pro = []
    pos_number = 0
    for u, v, prop in pos_edge:
        u = int(u)
        v = int(v)

        if int(u)>=len(embedding) or int(v)>=len(embedding):
            if u > node_index or v > node_index:
                continue
            else:
                y_test.append(1)
                y_test_pro.append(0.5)
        else:
            temp = np.dot(np.array(embedding[u]),np.array(embedding[v]))
            y_test_pro.append(sigmoid(temp*rho))
            if sigmoid(temp*rho)>=0.5:
                y_test.append(1)
            else:
                y_test.append(0)
        pos_number += 1
        y_true.append(1)
    neg_number = 0
    while neg_number < pos_number:
        test_neg_edge = sample_neg_edges()
        u = int(test_neg_edge[0])
        v = int(test_neg_edge[1])
        if int(u)>=len(embedding) or int(v)>=len(embedding):
            if u > node_index or v > node_index:
                continue
            else:
                y_test.append(1)
                y_test_pro.append(0.5)
        else:
            temp = np.dot(np.array(embedding[u]),np.array(embedding[v]))
            y_test_pro.append(sigmoid(temp*rho))
            if sigmoid(temp*rho)> 0.5:
                y_test.append(1)
            else:
                y_test.append(0)
        neg_number += 1
        y_true.append(0)
    print(pos_number)
    print(neg_number)
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_test_pro)
    ap = metrics.average_precision_score(y_true=y_true, y_score=y_test_pro, average='macro', sample_weight=None)
    print('Link prediction roc:', round(auc,4))
    print('Link prediction AP:', round(ap,4))




if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', default='wikitalk', help='data name.')

    parser.add_argument('--dimension', default=32, type=int, help='embedding size.')

    parser.add_argument('--k', default=400, type=int, help='number of motifs.')

    parser.add_argument('--save', default=1, type=int, help='save embedding')

    parser.add_argument('--rho', default=0.001, type=float, help='parameter rho')

    parser.add_argument('--split', default=0.2, type=float, help='save embedding')

    parser.add_argument('--split_each', default=0.02, type=float, help='save embedding')


    arg = parser.parse_args()

    split = arg.split
    split_each = arg.split_each
    usingtime = 0
    edges = load_edges('../data/' + str(arg.data) + '/'+ str(arg.data) + '_in.txt')
    for edge in edges[:int(len(edges)*split)]:
        node_index = max(node_index, int(edge[0]))
        node_index = max(node_index, int(edge[1]))

    print("Edges Load Done!")

    if arg.save != 1:
        origin_embedding = np.loadtxt('../data/' + str(arg.data) + '/'+ str(arg.data) +'_embedding.txt')
        origin_V = np.loadtxt('../data/' + str(arg.data) + '/'+ str(arg.data) +'_V.txt')
        with open('../data/' + str(arg.data) + '/'+ str(arg.data) +'_num_per_motif.txt') as f:
            for l in f:
                num_per_motif = l.strip().split(',')
    
    t1 = time.perf_counter()

    origin_embedding, origin_V, num_per_motif = recompute_embedding(edges[:int(len(edges)*split)],node_index)
    
    t2 = time.perf_counter()

    print("Origin_embedding Load Finished!\n")
    print("Time is "+ str(round(t2-t1,3)) + '\n')

    usingtime += t2-t1

    if arg.save == 1:
        np.savetxt('../data/' + str(arg.data) + '/'+ str(arg.data) +'_embedding.txt', origin_embedding, fmt='%.04f')
        np.savetxt('../data/' + str(arg.data) + '/'+ str(arg.data) +'_V.txt', origin_V, fmt='%.04f')
        with open('../data/' + str(arg.data) + '/'+ str(arg.data) +'_num_per_motif.txt', 'w') as f:
            f.write(str(num_per_motif))

    linkpred(edges[int(len(edges)*split):int(len(edges)*(split+split_each))], origin_embedding, node_index, arg.rho)
    
    #new_embeddings = origin_embedding
    #new_V = origin_V
    #MLP_new = MLP_p
    
    
    tot = 1
    while tot < 40:
        tot += 1
        for l in edges[int(len(edges)*split):int(len(edges)*(split+split_each))]:
            node_index = max(node_index,int(l[0]))
            node_index = max(node_index,int(l[1]))
        t4 = time.perf_counter()
        origin_embedding, origin_V = Incremental.Increse(origin_embedding, origin_V, edges[int(len(edges)*split):int(len(edges)*(split+split_each))], node_index, num_per_motif, arg.dimension)
        t5 = time.perf_counter()
        print("Incremental Time is "+ str(round(t5-t4,3)) + '\n')
        usingtime+=t5-t4
        split+=split_each
        test_pos_edges = edges[int(len(edges)*split):int(len(edges)*(split+split_each))]
        print("Split is: ", split)
        print("New_embedding_result is: \n")
        linkpred(test_pos_edges, origin_embedding, node_index, arg.rho)
    with open('../data/' + str(arg.data) + 'LDGE_IN_Result_AP', 'w') as f:
        for results in result:
            f.write(str(results[1])+',\n')
    with open('../data/' + str(arg.data) + 'LDGE_IN_Result_AUC', 'w') as f:
        for results in result:
            f.write(str(results[0])+',\n')
    with open('../data/' + str(arg.data) + 'LDGE_IN_Result_Time', 'w') as f:
        f.write('All time is ' + str(usingtime))
    print('All time is ' + str(usingtime))
        

    