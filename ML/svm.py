#-*- coding: utf-8 -*-
'''
author:carter
date:2017/12/6
svm for course design
'''
import pandas as pd
import numpy as np
import random
#加载数据
def loadDataSet(filename):
    data = pd.read_table(filename, sep='\s+')
    dataMat =data.iloc[:, :-1]
    lableMat = list(data.iloc[:, -1])
    for i in range(len(lableMat)):
        if lableMat[i] == 1:
            lableMat[i] = -1
        elif lableMat[i] == 2:
            lableMat[i] = 1
    return dataMat, lableMat

#辅助函数1
def selectJrand(i, m):
    '''
    :param i:第一个alpha下标
    :param m: 所有alpha的数目
    :return: 第二个alpha的下标
    '''
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

#辅助函数2
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMatin, label, C, toler, maxIter):
    '''
    :param dataMatin:训练集
    :param label: 训练集输出
    :param C: 常数，松弛因子
    :param toler: 容错率
    :param maxIter: 最大循环次数
    :return: b, alphas
    '''
    #设置参数
    dataMatrix = np.array(dataMatin)
    labelMat = np.array([label]).T
    b = 0
    m = len(dataMatrix)
    n = len(dataMatrix[0])
    alphas = np.zeros((m, 1))
    iter = 0

    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.dot((alphas*labelMat).T, np.dot(dataMatrix, dataMatrix[i].T)) + b)
            Ei = fxi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fxj = float(np.dot((alphas*labelMat).T, np.dot(dataMatrix, dataMatrix[j].T)) + b)
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L=H')
                    continue
                eta = 2.0 * np.dot(dataMatrix[i], dataMatrix[j].T) - np.dot(dataMatrix[i], dataMatrix[j].T) -\
                      np.dot(dataMatrix[j], dataMatrix[j].T)
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold < 0.00001):
                    print('j not moving enough')
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold)*np.dot(dataMatrix[i], dataMatrix[i].T) - labelMat[j]*(alphas[j] - alphaJold) * np.dot(dataMatrix[i], dataMatrix[j].T)
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold)*np.dot(dataMatrix[i], dataMatrix[j].T) - labelMat[j]*(alphas[j] - alphaJold) * np.dot(dataMatrix[j], dataMatrix[j].T)
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed: %d" % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        #else:
         #   iter = 0
        print("iteration number: %d" % iter)
    return b, alphas
dataMatin, label = loadDataSet('heart.txt')
b, alphas = smoSimple(dataMatin, label, 1000, 0.001, 40)
print(b,alphas)

#