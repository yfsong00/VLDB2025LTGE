import networkx as nx
import numpy as np
import pickle
import random
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

edge_set = set()
edges = []
def load_data_as_graph(path='network_data/superuser.txt', weight_idx=0, time_idx=2, bipartite = 0, transfer = 0):
    global edges
    global edge_set
    user = {}
    item = {}
    user_total = 0
    total_edges = 0
    tot = 0
    with open(path) as f:
        for line in f:
            if tot == 0:
                tot+=1
                continue
            total_edges += 1
            if total_edges % 100000000 == 0:
                print(str(total_edges) + "Edges have been read" + '\n')
            tokens = line.strip().split(",")
            u = str(tokens[0])
            v = str(tokens[1])
            if transfer:
                time = int(float(tokens[time_idx]))
            else:
                time = int(float(tokens[time_idx]))
            if weight_idx:
                weight = float(tokens[weight_idx])
                edges.append((u, v, {'weight':weight, 'time': time}))
            else:
                edges.append((u, v, {'time': time}))
            edge_set.add(str(u)+'and'+str(v))
        f.close()
    edges = sorted(edges, key=lambda x: x[2]['time'], reverse=False)
    inital_time = edges[0][2]['time']
    if bipartite:
        for l in range(len(edges)):
            u = str(edges[l][0])
            v = str(edges[l][1])
            if weight_idx:
                weight = float(edges[l][2]['weight'])
            time = int(edges[l][2]['time']) - inital_time + 2
            if user.__contains__(u):
                u = user[u]
            else:
                user_total += 1
                user[u] = user_total
                u = user_total
            if item.__contains__(v):
                v = item[v]
            else:
                user_total += 1
                item[v] = user_total
                v = user_total
            if weight_idx:
                edges[l] = (u, v, {'weight':weight, 'time': time})
            else:
                edges[l] = (u, v, {'time': time})
    else:
        for l in range(len(edges)):
            u = int(edges[l][0])
            v = int(edges[l][1])
            if weight_idx:
                weight = float(edges[l][2]['weight'])
            time = int(edges[l][2]['time'])- inital_time + 2
            if user.__contains__(u):
                u = user[u]
            else:
                user_total += 1
                user[u] = user_total
                u = user_total
            if user.__contains__(v):
                v = user[v]
            else:
                user_total += 1
                user[v] = user_total
                v = user_total
            if weight_idx:
                edges[l] = (u, v, {'weight':weight, 'time': time})
            else:
                edges[l] = (u, v, {'time': time})
    print("Read Done")
    return user_total




def create_embedding_and_training_data(nodes_number, train_edges_fraction=0.7, test_start_fraction = 0.85):
    global edges
    node_number = 0
    num_edges = len(edges)
    
    num_train_edges = int(train_edges_fraction * num_edges)
    num_test_edges = int(test_start_fraction * num_edges)
    
    train_edges = edges[:num_train_edges+1]
    
    for edge in train_edges:
        node_number = max(node_number, int(edge[0]))
        node_number = max(node_number, int(edge[1]))
    
    test_edges = edges[num_test_edges+1:]
    print(len(train_edges))

    return node_number, train_edges, test_edges


def main(args):
    global edges
    
    nodes_number = 0
    nodes_number_for_train = 0
    
    path = '../data/' + str(arg.data) + '/'+ str(arg.data) +'.txt'
    if arg.data == 'bitcoin':
        transfer = 1
    else:
        transfer = 0

    if args.topn:
        nodes_number = load_data_as_graph(path=path, weight_idx=2, time_idx=3, bipartite = arg.bipartite, transfer = transfer)
    else:
        nodes_number = load_data_as_graph(path=path, weight_idx=0, time_idx=2, bipartite = arg.bipartite, transfer = transfer)
    
    nodes_number_for_train, train_edges, test_edges = create_embedding_and_training_data(nodes_number, train_edges_fraction= arg.train, test_start_fraction = arg.test)
    
    print("Create Done!")
    
    save_path = '../data/' + str(arg.data) + '/'+ str(arg.data)

    with open(save_path + '_final.txt', 'w') as f:
        f.write(str(nodes_number_for_train)+','+str(len(train_edges))+'\n')
        for u,v,t in train_edges:
            f.write(str(u)+","+str(v)+","+str(t['time'])+'\n')
    with open(save_path + '_test.txt', 'w') as f:
        for u,v,t in test_edges:
            f.write(str(u)+","+str(v)+","+str(t['time'])+'\n')

    if arg.incremental:
        with open(save_path + '_in.txt', 'w') as f:
            for u,v,t in edges:
                f.write(str(u)+","+str(v)+","+str(t['time'])+'\n')

    if arg.topn:
        with open(save_path + '_topn.txt', 'w') as f:
            for u,v,t in test_edges:
                f.write(str(u)+","+str(v)+","+str(t['weight'])+'\n')

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', default='superuser', help='data name.')

    parser.add_argument('--bipartite', default=0, type = int, help='bipartite graph')
    
    parser.add_argument('--train', default=0.7, type = float, help='train split fraction.')
    
    parser.add_argument('--test', default=0.7, type = float, help='test start fraction.')

    parser.add_argument('--topn', default=0, type = int, help='topn recommendation')

    parser.add_argument('--incremental', default=0, type = int, help='test incremental')

    arg = parser.parse_args()

    main(arg)
