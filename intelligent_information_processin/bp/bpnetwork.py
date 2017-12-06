#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
import dill
import time
import matplotlib.pylab as plt
#******************************读入数据**********************
time_ = []
acc_ = []
time1 = time.time()
print('读取数据...')
data = pd.read_csv('train.csv')
train = data.iloc[:, 1:]
label = data.iloc[:, 0]
X_train,X_test,y_train,y_test=train_test_split(train,label,test_size=0.3,random_state=0)
X_train = np.array(X_train/256)
X_test = np.array(X_test/256)
y_test = np.array(y_test)
y_train = np.array(y_train)
dill.dump_session('loaddata.plk')
time2 = time.time()
print('数据读取完毕...  用时：%f' % (time2 - time1))

dill.load_session('loaddata.plk')
#**********************************配置神经网络*****************
num = len(X_train) #训练样本总数
num_in=len(X_train[0]) #输入节点数
num_out=10 #输出节点数
num_hide=math.sqrt(num_in) #隐层节点数
w0 = 0.2*np.random.random((num_in, int(num_hide)))-0.1 #初始化输入层权值矩阵
w1 = 0.2*np.random.random((int(num_hide),num_out))-0.1 #初始化隐层权值矩阵
b0 = np.zeros((int(num_hide))) #初始化输入层偏置
b1 = np.zeros((int(num_out))) #初始化隐层偏置
lr_in = 0.01
lr_hide = 0.01
err_door = 0.05

####################################函数定义#################################
def sigmoid(x):
    vector = []
    for i in x:
        vector.append(1/(1+math.exp(-i)))
    return np.array(vector)

def err(e):
    e=np.array(e)
    return 0.5*np.dot(e,e)

def predict(X_test, w0,w1,b0,b1):
    out = []
    for i in X_test:
        out.append((np.dot(np.dot(i, w0)+b0), w1)+b1)
    return out
#训练网络
lr_hide_ = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9]
lr_in_ = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9]
for pp in range(len(lr_hide_)):
    lr_in = lr_in_[pp]
    lr_hide = lr_hide_[pp]
    E=err_door+1
    count = 1
    time1 = time.time()
    while(E>err_door):
        E=0
        time3 = time.time()
        for k in range(len(X_train)):
            #print(k)
            t_label = np.zeros(num_out)
            t_label[y_train[k]] = 1
            #前向过程
            hide_value = np.dot(X_train[k], w0)+b0  #隐层输出
            hide_r = sigmoid(hide_value)  #隐层输出激活值
            out_value = np.dot(hide_r, w1) +b1 #输出层输出
            out_r = sigmoid(out_value)  #输出层激活值

            #后向过程
            e = t_label - out_r
            out_delta = e * out_r * (1 - out_r) #输出层delta
            hide_delta = hide_r*(1-hide_r)*np.dot(w1, out_delta)
            for i in range(len(out_r)):
                w1[:, i] += lr_in*out_delta[i]*hide_r  #隐层到输出层权值矩阵更新
            for i in range(len(hide_delta)):
                w0[:, i] += lr_hide*hide_delta[i]*X_train[k]  #输入层到隐层权值矩阵更新
            b0 += lr_in*hide_delta #输入层偏置更新
            b1 += lr_hide*out_delta #隐层偏置更新
            E = E+err(e)
        E=E/num
        time4 = time.time()
        print('---iter %d ---: %f %f s' %(count, E, (time4-time3)))
        count = count+1


    time2 = time.time()
    time_.append(float(time2-time1))
    print('训练用时：%f' % (time2-time1))
    dill.dump_session('2.plk')
    #####################################################################################
    dill.load_session('2.plk')
    print('保存网络完毕...\n开始读入测试数据...')
    time5 = time.time()
    dill.load_session('2.plk')
    right = np.zeros(10)
    numbers = np.zeros(10)
    print('读取测试数据完毕...')                                  # 以上读入测试数据
    # 统计测试数据中各个数字的数目
    for i in y_test:
        numbers[i] += 1
    for count in range(len(X_test)):
        hid_value = np.dot(X_test[count], w0) + b0     # 隐层值
        hid_act = sigmoid(hid_value)                # 隐层激活值
        out_value = np.dot(hid_act, w1) + b1             # 输出层值
        out_act = sigmoid(out_value)                # 输出层激活值
        if np.argmax(out_act) == y_test[count]:
            right[y_test[count]] += 1
    time6 = time.time()
    result = right/numbers
    sum = right.sum()
    print('每个数字测试结果正确率：', result)
    print('准确率：', sum/len(X_test), '测试用时：%f' % (time6-time5))
    acc_.append(sum/len(X_test))
dill.dump_session('3.plk')
dill.load_session('3.plk')
plt.figure(1)
plt.subplot(211)
plt.plot(lr_hide_, time_, 'b')
plt.ylabel('收敛时间/s')
plt.subplot(212)
plt.plot(lr_hide_, acc_, 'r')
plt.ylabel('准确率')
plt.xlabel('学习率')
plt.show()