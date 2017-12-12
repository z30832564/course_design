import numpy as np
# import csv
import operator
from pandas import DataFrame, Series
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame


def knn(inX, dataSet, labels, k):
    '''
    knn算法
    :param inX: 输入待判断数据
    :param dataSet: 训练样本数据集
    :param labels: 训练样本标签集
    :param k:  几近邻
    :return: labels
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def knn_judge(result_s, label_true):
    error = 0.00
    for i in range(len(label_true)):
        if result_s[i] != label_true[i]:
            error = error + 1
    print('错误数：', error)
    error_rate = error / len(label_true)
    print('错误率：', error_rate)

data = pd.read_table('heart.txt', sep='\s+')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test = np.array(X_test)
X_train = np.array(X_train)
y_test = np.array(y_test)
y_train = np.array(y_train)

result = []
for i in range(len(X_test)):
    print(knn(X_test[i], X_train, y_train, 5))
    result.append(knn(X_test[i], X_train, y_train, 5))

knn_judge(result, y_test)