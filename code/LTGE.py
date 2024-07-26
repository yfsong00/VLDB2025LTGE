import random
import math
import numpy as np
#from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pickle
import time
import gc
import sys
from sys import getsizeof
import scipy
import TopN

edge_set = set()
node_number = 0
edge_number = 0

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def load_edges(file):
    edges = []
    global edge_set
    global node_number
    global edge_number
    with open(file, 'r') as f:
        tot = 0
        for l in f:
            if tot == 0:
                node_size, edge_size = l.strip().split(',')
                node_size = int(node_size)
                node_number = node_size
                edge_size = int(edge_size)
                edge_number = edge_size
                print(str(node_size))
                print(str(edge_size))
                tot+=1
                continue
            else:
                x,y,t = l.strip().split(',')
                x = int(x)
                y = int(y)
                t = float(t)
                edge_set.add(str(x)+'and'+str(y))
                edges.append((x,y,t))
    return np.array(edges)

def load_test_edges(file):
    global node_number
    edges = []
    with open(file, 'r') as f:
        tot = 0
        for l in f:
            x,y,t = l.strip().split(',')
            x = int(x)
            y = int(y)
            node_number = max(node_number, x)
            node_number = max(node_number, y)
            t = float(t)
            edges.append((x,y,t))
    return np.array(edges)

def sample_neg_edges():
    global edge_set
    global node_number
    ok = 1
    #size_epoch += 1
    while ok:
        first_node = random.randint(1,node_number)  # pick a random node
        second_node = random.randint(1,node_number)
        if first_node==second_node or (str(first_node)+'and'+str(second_node)) in edge_set:
            continue
        ok = 0
    n_edge = (first_node, second_node, 0)
    return n_edge


def recompute_embedding(edges, k = 400, dimension = 32):
    global node_number
    global edge_number
    print("Largest Motif Number is " + str(k))
    t = edges[0][2]
    T = edges[-1][2] - t
    label = 0
    Hash = []
    for l in range(edge_number):
        if edges[l][2] - t > T/k:
            label+=1
            t = edges[l][2]
        Hash.append(label)
    tot = 0
    E_m = scipy.sparse.lil_matrix((int(node_number+1),k+1))
    for edge in edges:
        E_m[(int(edge[0]),Hash[tot])]+=max(1.0,float(edge[2]))
        E_m[(int(edge[1]),Hash[tot])]+=max(1.0,float(edge[2]))
        tot+=1
    for i, row in enumerate(E_m.rows):
        for j in row:
            if E_m[i, j] > 0: 
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
    
    return Embedding


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
    print('Link prediction roc:', round(auc,3))
    print('Link prediction AP:', round(ap,3))

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', default='superuser', help='data name.')

    parser.add_argument('--dimension', default=32, type=int, help='embedding size.')

    parser.add_argument('--k', default=400, type=int, help='number of motifs.')

    parser.add_argument('--task', default='linkprediction', help='downstream task')
    
    parser.add_argument('--save', default=1, type=int, help='save embedding')

    parser.add_argument('--rho', default=0.001, type=float, help='parameter rho')

    arg = parser.parse_args()
    
    train_edges = load_edges("../data/"+ str(arg.data) + "/"+ str(arg.data) + "_final.txt")
    
    print("Edges Load Done!")

    if arg.task == 'linkprediction':
        if arg.save == 0:
            origin_embedding = np.loadtxt('../data/' + str(arg.data) + '/'+ str(arg.data) +'_embedding.txt')
        else:
            t1 = time.perf_counter()
            
            origin_embedding = recompute_embedding(train_edges, arg.k, arg.dimension)
            
            t2 = time.perf_counter()
            
            print("Origin_embedding Load Finished!\n")
            print("Embedding time is "+ str(round(t2-t1,3)) + '\n')

        if arg.save != 0:

            np.savetxt('../data/' + str(arg.data) + '/'+ str(arg.data) +'_embedding.txt', origin_embedding, fmt='%.04f')
            print("Save embedding Done.")
        
        test_edges = load_test_edges("../data/"+ str(arg.data) + "/"+ str(arg.data) + "_test.txt")

        for edge in test_edges:
            edge_set.add(str(edge[0])+'and'+str(edge[1]))
            node_number=max(node_number,edge[0])
            node_number=max(node_number,edge[1])
        
        linkpred(test_edges, origin_embedding, len(origin_embedding)-1, arg.rho)

    elif arg.task == 'recommendation':

        t1 = time.perf_counter()

        origin_embedding = recompute_embedding(train_edges, arg.k, arg.dimension)

        t2 = time.perf_counter()

        print("Origin_embedding Load Finished!\n")
        print("Time is "+ str(round(t2-t1,3)) + '\n')

        TopN.evaluate(origin_embedding, arg.data, arg.rho)

