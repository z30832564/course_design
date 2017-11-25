# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import dill
from sklearn.cross_validation import train_test_split
data = pd.read_csv('data.csv')
train = data.iloc[:, :-1]
label = data.iloc[:, -1]
X_train,X_test,y_train,y_test=train_test_split(train,label,test_size=0.3,random_state=0)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
'''print(X_train)
print(X_test)
def f(n):
    if n >= 0:
        return 1
    else:
        return 0
def jiance(a, b):
    flag = 1
    for i in range(0, len(a)):
        if a[i] == b[i]:
            pass
        else:
            flag = 0
    if flag == 1:
        return 0
    else:
        return 1


for hang in range(len(X_train)):
    print(hang)
    for lie in range(len(X_train[0])):
        if X_train[hang][lie] > 120:
            X_train[hang][lie] = 1
        else:
            X_train[hang][lie] = 0

print(X_train)
print(y_train)
w0 = range(len(X_train[0]))
b0 = 0
flag1 = 1
count = 1
while(flag1):
    a = []
    for i in range(0, len(X_train)):
        print('第', count, '次迭代：')
        num = np.dot(w0, X_train[i].T)+b0
        a.append(f(num))
        e = y_train[i]-f(num)
        print('a=', a[i])
        print('e=', e)
        if e == 0:
            pass
        else:
            w0 = w0+np.dot(e,X_train[i].T)
            b0 = b0+e
        count = count +1
    flag1 = jiance(a, y_train)
    print('a=',a)
    print('b0=',b0)
    print(w0)
    print('**************************')
dill.dump_session('1.plk')
print('**************-------------********************')
dill.load_session('1.plk')'''
output = []
'''for hang in range(len(X_test)):
    print(hang)
    for lie in range(len(X_test[0])):
        if X_test[hang][lie] > 120:
            X_test[hang][lie] = 1
        else:
            X_test[hang][lie] = 0'''
def f(n):
    if n >= 0:
        return 1
    else:
        return 0
w0=[-788, 2677.6]
b0=-7035
print(w0)
print(X_test[0])
for i in range(len(X_test)):
    output.append(f(np.dot(w0, X_test[i].T)+b0))

print(output)
count=0
for i in range(len(y_test)):
    if y_test[i]==output[i]:
        count=count+1
print(float(count)/float(len(y_test)))
