#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#生成非线性可分的数据集
def create_nolinear_data(n):
    np.random.seed(1)
    x_11 = 50 + np.random.randint(0, 100, (n, 1))
    x_12 = 50 + np.random.randint(0, 100, (n, 1))
    x_13 = -50 - np.random.randint(0, 100, (n, 1))
    x_14 = -50 - np.random.randint(0, 100, (n, 1))
    x_21 = 50 + np.random.randint(0, 100, (n, 1))
    x_22 = -50 - np.random.randint(0, 100, (n, 1))
    x_23 = -50 - np.random.randint(0, 100, (n, 1))
    x_24 = 50 + np.random.randint(0, 100, (n, 1))

    plus_samples_1 = np.hstack([x_11, x_12, np.ones((n,1))])
    plus_samples_2 = np.hstack([x_13, x_14, np.ones((n,1))])
    minus_samples_1 = np.hstack([x_21, x_22, -np.ones((n,1))])
    minus_samples_2 = np.hstack([x_23, x_24, -np.ones((n,1))])
    plus_samples = np.vstack([plus_samples_1, plus_samples_2])
    minus_samples = np.vstack([minus_samples_1, minus_samples_2])
    samples = np.vstack([plus_samples, minus_samples])
    np.random.shuffle(samples)#混洗数据

    return samples
def showData2D(data):
    x=[]
    y=[]
    X=[]
    Y=[]
    for i in data:
        if i[2]==-1:
            x.append(i[0])
            y.append(i[1])
        else:
            X.append(i[0])
            Y.append(i[1])
    ax=plt.subplot()
    ax.scatter(x,y,c='r')
    ax.scatter(X,Y,c='b')
    plt.show()




data = create_nolinear_data(100)
showData2D(data)


#net1随机感知器层
def net1(net1_in):
    def f(a):
        for i in range():
            if i>0:

    w0=np.random.randint(-100, 100, (1,3))
    b0 = np.random.randint(-100, 100)
    num = np.dot(w0,net1_in)+b0
