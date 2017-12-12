#-*- coding: utf-8 -*-
"""greedy.py
author:carter
date:2017/12/1
bp神经网络拟合预测数据
"""

import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
#******************************读入数据**********************
data = pd.read_table('housing.txt', sep='\s+')
for i in data:
    data[i] = (data[i]-min(data[i]))/(max(data[i])-min(data[i]))
data.drop('3',1)
data.drop('8',1)
label = data.iloc[:, -1]
train = data.iloc[:, :-1]
pca = PCA(n_components=2,copy=True)
#train = pca.fit_transform(train)
label = data.iloc[:, -1]
print(label)
X_train,X_test,y_train,y_test=train_test_split(train,label,test_size=0.3,random_state=0)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)





#**********************************配置神经网络*****************
num = len(X_train) #训练样本总数
num_in=len(X_train[0]) #输入节点数
num_out=1 #输出节点数
num_hide=math.sqrt(num_in) #隐层节点数
w0 = 0.2*np.random.random((num_in, int(num_hide)))-0.1 #初始化输入层权值矩阵
w1 = 0.2*np.random.random((int(num_hide),num_out))-0.1 #初始化隐层权值矩阵
b0 = np.zeros((int(num_hide))) #初始化输入层偏置
b1 = np.zeros((int(num_out))) #初始化隐层偏置
lr_in = 0.01
lr_hide = 0.1
err_door = 0.01
beta_in = 0.9
beta_hide = 0.9
####################################函数定义#################################
def sigmoid(x):
    vector = []
    for i in x:
        vector.append(1/(1+math.exp(-i)))
    return np.array(vector)

def err(e):
    e=np.array(e)
    return 0.5*np.dot(e,e.T)

def predict(X_test, w0,w1,b0,b1):
    out = []
    for i in X_test:
        out.append((np.dot((np.dot(i, w0)+b0), w1))+b1)
    return out
#训练网络
flag=1
count=0
the_E=[]
while(flag):

    E = 0
    for k in range(len(X_train)):
        #print(k)
        #print(k)
        #t_label = np.zeros(num_out)
        #t_label[y_train[k]] = 1
        #前向过程
        hide_value = np.dot(X_train[k], w0)+b0  #隐层输出
        hide_r = sigmoid(hide_value)  #隐层输出激活值
        out_value = np.dot(hide_r, w1) +b1 #输出层输出
        out_r = sigmoid(out_value)  #输出层激活值

        #后向过程
        e = y_train[k] - out_r
        out_delta = e * out_r * (1 - out_r) #输出层delta
        hide_delta = hide_r*(1-hide_r)*np.dot(w1, out_delta)
        for i in range(len(out_r)):
            w1[:, i] += lr_in*out_delta[i]*hide_r  #隐层到输出层权值矩阵更新
        for i in range(len(hide_delta)):
            w0[:, i] += lr_hide*hide_delta[i]*X_train[k]  #输入层到隐层权值矩阵更新
        b0 += lr_in*hide_delta #输入层偏置更新
        b1 += lr_hide*out_delta #隐层偏置更新
        E = E + err(e)
    E = E / num
    print(count,abs(E))
    count +=1
    the_E.append(abs(E))
    if count==1000:
        break
    if math.sqrt(abs(E))<err_door:
        flag=0
#
#####################################################################################
out=[]
for count in range(len(X_test)):
    hid_value = np.dot(X_test[count], w0) + b0     # 隐层值
    hid_act = sigmoid(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w1) + b1
    out_r = sigmoid(out_value)
    out.append(out_r)
print(y_test)
print(out)

x=range(1000)
plt.figure()
plt.plot(x,the_E)
plt.show()