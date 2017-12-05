# -*- coding: utf-8 -*-
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
        for p in range(2, count):
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
print(maxP)
for i in range(len(R_value)):
    if i != wholeP_index:
        if R_value[i] > maxP:
            maxP = R_value[i]
        if R_value[i] == wholeP:
            maxP == R_value[i]
            break
print('约简后：', R[i],maxP)

columns = list(data.columns)
data = np.array(data)
print(data)
Matrix = np.zeros((len(data),len(data))).astype(str)
ID = []
for i in range(len(data)):
    for j in range(0, i):
        k = []
        print(data[i][-1],data[j][-1])
        if data[i][-1] != data[j][-1]:
            for p in range(len(columns)-1):
                print(data[i][p],data[j][p])
                if data[i][p] != data[j][p]:
                    k.append(columns[p])
        ID.append(k)
        Matrix[i][j] = ''.join(k)
Matrix = pd.DataFrame(Matrix)
print(Matrix)
