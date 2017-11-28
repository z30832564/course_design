import pandas as pd
import numpy as np
import math
#read_data
################################################################
data = pd.read_table('heart.txt', sep='\s+')
data = data.iloc[0:10, 0:10]
for i in range(len(data.iloc[0])):
    data[str(i)] = (data[str(i)]-min(data[str(i)]))/(max(data[str(i)])-min(data[str(i)]))
data = np.array(data)
data = data
print('--------------------数据为:----------------------')
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



#寻找最好lambda时一些函数
############################################################
#计算距离
def distance(a,b):
    a = list(np.array(a) - np.array(b))
    dis = 0
    for i in a :
        dis += i**2
    dis = math.sqrt(dis)
    return dis
#计算同类距离
def same(a):
    dis = 0
    for i in range(len(a)-1):
        dis += distance(data[a[i]], data[a[i+1]])
    return dis
#计算样本中心
def meanplace(a):
    mp = np.zeros(len(data[0]))
    for i in range(len(a)):
        mp += data[a[i]]
    mp = list(mp/len(a))
    return mp
#寻找最好lambda
##################################################################
def findBestLambda():
    Gs = []
    for i in R:
        sameDis = 0
        for j in i:
            if len(j)>1:
                sameDis += same(j)
        difDis = 0
        mean_of_dif = []
        for j in i:
            if len(j)>1:
                mean_of_dif.append(meanplace(j))
            else:
                mean_of_dif.append(data[j[0]])
        for k in range(len(mean_of_dif)):
            for p in range(len(mean_of_dif)):
                difDis += distance(mean_of_dif[k], mean_of_dif[p])
        difDis = difDis/2
        G = difDis-sameDis
        Gs.append(G)
    BestLambda = lambdaVector[Gs.index(max(Gs))]
    print('-----------------------最优lambda为------------------')
    print(BestLambda)
    print('----------------------此时聚类结果为：---------------')
    print(R[Gs.index(max(Gs))])
findBestLambda()
