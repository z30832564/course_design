#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
#read_data
################################################################
data = pd.read_table('heart.txt', sep='\s+')
data = data.iloc[:10, 0:12]
for i in range(len(data.iloc[0])):
    data[str(i)] = (data[str(i)]-min(data[str(i)]))/(max(data[str(i)])-min(data[str(i)]))
data = np.array(data)
data = data
print('--------------------数据为:----------------------')
data=np.array([[1,1],
               [1,2],
              [2,3],
               [10000,10100],
               [10000,10110]])

print(data)
#一些函数
################################################################
def similar(a, b):
    mum1 = 0
    mum2 = 0
    for i in range(len(a)):
        mum1 += (a[i])**2
        mum2 += (b[i])**2
    a=np.array(a)
    b=np.array(b)
    son = np.dot(a, b.T)
    return float(son)/float(math.sqrt(mum1)*math.sqrt(mum2))

def minmax(a, b):
    a = list(a)
    b = list(b)
    minVector = []
    for i in range(len(a)):
        minVector.append(min([a[i], b[i]]))
    maxNumber = max(minVector)
    return maxNumber

def muti(a, b):
    mutiMatrix = np.zeros((len(a), len(a[0])))
    for i in range(len(a)):
        for j in range(len(a[0])):
            mutiMatrix[i][j] = minmax(a[i], b.T[j])
    return mutiMatrix

def involution(a, n):
    b = muti(a,a)
    for i in range(n-1):
        b = muti(b, a)
    return b

def combine(a, b):
    result = np.zeros((len(a), len(a[0])))
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[i][j] = max([a[i][j], b[i][j]])
    return result


def closure(a):
    result = combine(a, involution(a, 2))
    for i in range(len(a)-2):
        result = combine(result, involution(a, i+3))
    return result



#构造相似度矩阵
################################################################
similarMatrix = np.ones((len(data),len(data)))
for i in range(len(data)):
    for j in range(i, len(data)):
        similarMatrix[i][j] = similarMatrix[j][i] = similar(data[i], data[j])
print('--------------------相似度矩阵为:----------------------')
print(similarMatrix)


#聚类分析
################################################################
tr = closure(similarMatrix)
print('---------------------传递闭包为：----------------------')
print(tr)
lambdaVector = (np.unique(tr))
for i in range(len(lambdaVector)):
    lambdaVector[i] = round(lambdaVector[i],4)
lambdaVector = list(set(lambdaVector))
print('--------------------lambda集合为：---------------------')
print(lambdaVector)
r = []
R = []
for i in lambdaVector:
    if i <= 1:
        r=[]
        for p in range(len(tr)):
            for k in range(p, len(tr[0])):
                if round(tr[p][k], 4)>=i:
                    flag = 0
                    for j in r:
                        if p in j:
                            j.append(k)
                            for j in range(len(r)):
                                r[j] = list(set(r[j]))
                            flag = 1
                            break
                        if k in j:
                            j.append(p)
                            for j in range(len(r)):
                                r[j] = list(set(r[j]))
                            flag =1
                            break
                    if flag == 0:
                        r.append([p, k])
                        for j in range(len(r)):
                            r[j] = list(set(r[j]))
                else:
                    pass
        #for j in range(len(r)):
            #r[j] = list(np.array(r[j]) + 1)
        print(i,':')
        print(r)
        R.append(r)

print('-------------------动态聚类图----------------')
for i in range(len(data)):
    print(i, '    ', end='')
print('\n')
log_ = np.zeros((len(lambdaVector)+1, 2*len(data)))
for i in range(len(log_[0])):
    if i % 2 == 0:
        log_[0][i] = 1
com = []
com_l = 1
max_num = 1
log_for_max = []
log_for_maxnum = []
for i in range(len(lambdaVector)-2, -1, -1):
    log_[len(lambdaVector) - i -1] = log_[0]
    for k in log_for_max:
        log_[len(lambdaVector) - i - 1][k] = 0
    for j in range(len(R[i])):
        if len(R[i][j]) > com_l:
            com_l += 1
            min_num = int(min(R[i][j]))
            r_copy = R[i][j].copy()
            for k in log_for_maxnum:
                if k in r_copy:
                    r_copy[r_copy.index(k)] = -1
            max_num = max(r_copy)
            for q in range(min_num*2-1, max_num*2-2):
                if log_[len(lambdaVector)-i-1][q] == 1:
                    log_[len(lambdaVector) - i - 1][q] = 3
                    continue
                log_[len(lambdaVector)-i - 1][q] = 2
    log_for_max.append(max_num*2-2)
    log_for_maxnum.append(max_num)
    log_for_max = list(set(log_for_max))
log_[-1][-2] = 1
for i in range(1, len(log_)):
    for j in range(-2, len(log_[0])-2):
        if log_[i][j] == 1:
            print(' | ', end='')
        if log_[i][j] == 0:
            print('   ', end='')
        if log_[i][j] == 2:
            print('___', end='')
        if log_[i][j] == 3:
            print('_!_', end='')
    print('----', lambdaVector[i-1], end='\n')

#寻找最好lambda时一些函数
############################################################
#计算分子
def comdif(lambdaV):
    index = lambdaVector.index(lambdaV)
    r_son = len(R[index])
    son = 0
    for i in range(len(R[index])):
        class_son = []
        for j in R[index][i]:
            class_son.append(data[j])
        class_son = np.array(class_son)
        son = son + len(R[index][i]) * np.dot((meanplace(class_son) - meanplace(data)), (meanplace(class_son)-meanplace(data)).T)
    son = float(son/(r_son-1))
    return son
#计算分母
def comSame(lambdaV):
    index = lambdaVector.index(lambdaV)
    r_mother = len(R[index])
    mother = 0
    for i in range(len(R[index])):
        for p in range(len(R[index])):
            class_mother_one = []
            class_mother_two = []
            for j in R[index][i]:
                class_mother_one.append(data[j])
            for j in R[index][p]:
                class_mother_two.append(data[j])
            class_mother_one = np.array(class_mother_one)
            class_mother_two = np.array(class_mother_two)
            if len(data)-r_mother == 0:
                continue
            mother = mother + np.dot((meanplace(class_mother_one)-meanplace(class_mother_two)), (meanplace(class_mother_one)-meanplace(class_mother_two)).T)
            mother = float(mother/(len(data)-r_mother))
    return mother
def meanplace(a):
    mp = np.zeros(len(data[0]))
    for i in range(len(a)):
        mp += a[i]
    mp = np.array(list(mp/len(a)))
    return mp
#寻找最好lambda
##################################################################
def findBestLambda():
    F = []
    for i in lambdaVector:
        if comSame(i) != 0:
            F.append(float(comdif(i)/comSame(i)))
    BestLambda = lambdaVector[F.index(max(F))]
    print(F)
    print('-----------------------最优lambda为------------------')
    print(BestLambda)
    print('----------------------此时聚类结果为：---------------')
    print(R[F.index(max(F))])
findBestLambda()
