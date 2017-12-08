# -*- coding: utf-8 -*-
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import time
data_g = pd.read_table('glassscaled.txt', sep=',')
data_h = pd.read_table('heart.txt', sep='\s+')
for i in range(13):
     data_h[str(i)] = (data_h[str(i)]-min(data_h[str(i)]))/(max(data_h[str(i)])-min(data_h[str(i)]))
for i in range(9):
    data_g[str(i)] = (data_g[str(i)]-min(data_g[str(i)]))/(max(data_g[str(i)])-min(data_g[str(i)]))
print(data_g)
print(data_h)
'''1'''
def h(c):
    data_h_train = data_h.iloc[:, :-1]
    data_h_label = data_h.iloc[:, -1]
    print(data_h_label)
    X_h_train,X_h_test,y_h_train,y_h_test = train_test_split(data_h_train, data_h_label,test_size=0.3, random_state=0)
    start = time.time()
    # 利用SVC训练
    print('SVC begin')
    clf1 = svm.SVC(C=pow(10,c))
    clf1.fit(X_h_train, y_h_train)
    # 返回accuracy
    accuracy = clf1.score(X_h_test, y_h_test)
    end = time.time()
    print("accuracy: {}, time elaps:{}".format(accuracy, int(end-start)))
    return accuracy
'''2'''
def g(c):
    data_g_train = data_g.iloc[:, :-1]
    data_g_label = data_g.iloc[:, -1]
    print(data_g_label)
    X_g_train,X_g_test,y_g_train,y_g_test = train_test_split(data_g_train, data_g_label,test_size=0.3, random_state=0)
    start = time.time()
    # 利用SVC训练
    print('SVC begin')
    clf1 = svm.SVC(C=pow(10,c))
    clf1.fit(X_g_train, y_g_train)
    # 返回accuracy
    accuracy = clf1.score(X_g_test, y_g_test)
    end = time.time()
    print("accuracy: {}, time elaps:{}".format(accuracy, int(end-start)))
    return accuracy
s=[]
for i in range(1,15):
    s_=g(i)
    s.append(s_)
plt.plot(range(1,15),s)
plt.show()