# -*- coding: utf-8 -*-
"""Attribute_reduction.py
author:carter
date:2017/12/8
粗糙集属性约简
"""


import pandas as pd
import numpy as np
import itertools
#load data
data = pd.read_csv('data_for_attribute', sep='\s+')
def comIND(a):
    US = []
    for i in a:
        p = []
        for name,group in data.groupby(i):
            p.append(set(list(group.index)))
        US.append(p)
    count = len(US)-1
    IND = []
    if len(a) <= 1:
        IND = US[0]
    if len(a)==2:
        for i in US[0]:
            for j in US[1]:
                if set(i) & set(j):
                    IND.append(set(i)&set(j))
    if len(a)>2:
        for i in US[0]:
            for j in US[1]:
                if set(i) & set(j):
                    IND.append(set(i)&set(j))
        for p in range(2,  count+1):
            IND_ = []
            for i in IND:
                for j in US[p]:
                    if set(i) & set(j):
                        IND_.append(set(i) & set(j))
            IND = IND_
    return IND

def comU():
    U = set(range(len(data)))
    return U

def comPOS(a, IND):
    POS = set()
    for i in IND:
        if i & a == i:
            POS=POS|i
    return POS


def comPOSforPQ(p,q):
    INDforp = comIND(p)
    INDforq = comIND(q)
    POSpq = set()
    for i in INDforq:
        j = comPOS(i, INDforp)
        POSpq = POSpq | j
    return POSpq


def comGama(p, q):
    gama = abs(len(comPOSforPQ(p,q)))/abs(8)
    return gama



#def reduction():
tiaojian = []
for t in range(len(list(data.columns))-1):
    tiaojian.append(list(data.columns)[t])
R = []
R_value = []
for i in range(1,len(data.columns)+1):
    for j in list(itertools.combinations(tiaojian, i)):
        R.append(''.join(list(j)))
        R_value.append(comGama(list(j),list(data.columns)[-1]))
wholeP_index = R.index(''.join(tiaojian))
wholeP = R_value[wholeP_index]
maxP = wholeP
for i in range(len(R_value)):
    if i != wholeP_index:
        if R_value[i] > maxP:
            maxP = R_value[i]
        if R_value[i] == wholeP:
            maxP = R_value[i]
            break
print('----------数据为-----------')
print(data)
print('----------依赖度-----------')
for j in range(len(R)):
    print(R[j],R_value[j])

print('-----根据依赖度约简后------')
print(R[i], maxP)

columns = list(data.columns)
data = np.array(data)
Matrix = np.zeros((len(data), len(data))).astype(str)
ID = []
for i in range(len(data)):
    for j in range(0, i):
        k = []
        if data[i][-1] != data[j][-1]:
            for p in range(len(columns)-1):
                if data[i][p] != data[j][p]:
                    k.append(columns[p])
        ID.append(k)
        Matrix[i][j] = ''.join(k)
Matrix = pd.DataFrame(Matrix)
print('---------区分矩阵----------')
print(Matrix)
ID_set = []
last_ID_set = []
for i in range(len(ID)):
     ID_set.append(set(ID[i]))
for i in range(len(ID_set)):
    flag = 0
    if ID_set[i]:
        for j in range(1, len(ID_set)):
            if ID_set[j]:
                if ID_set[i] != ID_set[j]:
                    if ID_set[i] & ID_set[j] == ID_set[j]:
                        flag = 1
                        break
    if flag == 0:
        if ID_set[i] and ID_set[i] not in last_ID_set:
            last_ID_set.append(ID_set[i])

print('-----------最小约简----------')
print(last_ID_set)
final_ID = []
for i in last_ID_set:
    final_ID.append(list(i)[0])
print('-------------选取------------')
print(final_ID)
final_data = pd.read_csv('data_for_attribute', sep='\s+')
cloumns = list(final_data)
for i in list(final_data):
    if (i not in final_ID) and (i != cloumns[-1]):
        del final_data[i]
print('-----------约简后----------')
final_data_ = np.array(final_data)
final_data = final_data.drop_duplicates()
print(final_data)

# 规则约简
for i in range(len(final_data_[0])-1):
    kind = np.unique(final_data.iloc[:, i])
    col = final_ID.copy()
    col.append(list(final_data.columns)[-1])
    del col[i]
    for name, group in final_data.groupby(col):
        if len(list(np.unique(group.iloc[:, i]))) == len(kind):
            index = group.index
            final_data_[index[0]][i] = '/'
            if len(index) == 0:
                continue
            for j in range(1, len(index)):
                final_data_[index[j]][0] = '*'

#打印规则
print('-----------------提取规则----------------')
for i in range(len(final_data_)):
    for j in range(len(final_data_[0])-2):
        flag = 1
        if final_data_[i][j] == '*':
            flag = 0
            break
        if final_data_[i][j] == '/':
            break
        print(final_data_[i][j], '+ ', end='')
    if flag == 1:
        if final_data_[i][-2] != '/':
            print(final_data_[i][-2], end='')
    print('--->', final_data_[i][-1])