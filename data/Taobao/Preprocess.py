import numpy as np
import pickle
import random
import time
import pandas as pd
path = 'Origin.txt'
result = 'Taobao.txt'
#labels = 'Tianchi_label.txt'
item = []
user = []
label = []
times = []
item_dic = {}
user_dic = {}
weight = []
total = 0
total_edges = 0
with open(path) as f:
    with open(result,'w') as output:
        for l in f:
            total_edges+=1
            if total_edges % 25000000 == 0:
                print(str(total_edges) + " edges have been read")
            s = l.strip().split('\t')
            p=str(s[0])
            if p in item_dic:
                item.append(int(item_dic[p]))
            else:
                item_dic[p] = total
                total += 1
                item.append(int(item_dic[p]))
            p=str(s[1])
            if p in user_dic:
                user.append(int(user_dic[p]))
            else:
                user_dic[p] = total
                total += 1
                user.append(int(user_dic[p]))
            p = s[2]
            if p =='alipay':
                p = 4.0
            elif p == 'cart':
                p = 3.0
            elif p =='collect':
                p = 2.0
            else:
                p = 1.0
            weight.append(float(p))
            '''if p =='alipay':
                label.append(1)
            else:
                label.append(0)'''
            date_time = str(s[3])
            timeArray = time.strptime(date_time, "%Y-%m-%d %H:%M:%S")    
            times.append(int(time.mktime(timeArray)))
        print("Read Finish")
        print("Total edges are " + str(len(item)) + '\n')
        print("Total nodes are " + str(total)+ '\n')
        with open('Tianchi_attr.txt', 'w') as f2:
            f2.write("Total edges are " + str(len(item)) + '\n')
            f2.write("Total nodes are " + str(total)+ '\n')
        for i in range(len(item)):
            if i % 10000000 == 0:
                print(str(i) + " edges has finished")
            output.write(str(item[i])+' '+str(user[i])+' '+str(weight[i])+' '+str(times[i])+'\n')
        output.close()
    '''with open(labels,'w') as output_label:
        for i in range(len(user)):
            if i % 10000000 == 0:
                print(str(i) + " labels has finished")
            output_label.write(str(user[i])+' '+str(label[i])+'\n')
        output_label.close()
    f.close()'''
