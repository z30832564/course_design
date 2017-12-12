#-*- coding: utf-8 -*-
'''
author:carter
data:2017/12/11
禁忌搜索算法解决图分区问题
'''

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math
def creatData(m, n):
    mypoint = []
    for i in range(m):
        mypoint_x = random.randint(100, 200)
        mypoint_y = random.randint(100, 200)
        mypoint.append([mypoint_x, mypoint_y])
    dpoint = []
    for i in range(n):
        dpoint_x = random.randint(0, 100)
        dpoint_y = random.randint(0, 100)
        dpoint.append([dpoint_x, dpoint_y])
    return np.array(mypoint), np.array(dpoint)

def distance(a, b):
    d = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    return d

#评价函数
def findBestPath(orders,data):
    DISTANCE = []
    for j in range(len(orders)):
        DIS = 0
        for i in range(len(orders[0])-1):
            DIS += distance(data[orders[j][i]], data[orders[j][i+1]])
        DIS += distance(data[orders[j][-1]], data[orders[j][0]])
        DISTANCE.append(DIS)
    index = DISTANCE.index(min(DISTANCE))
    return orders[index], index

def sitisfy(path, data):
    dis = 0
    for i in range(len(path) - 1):
        dis += distance(data[path[i]], data[path[i + 1]])
    dis += distance(data[path[-1]], data[path[0]])
    return dis
def upDateTS(ts_table):
    for i in range(len(ts_table)):
        for j in range(len(ts_table[0])):
            if ts_table[i][j] != 0:
                ts_table[i][j] -= 1
def changeMaxMin(a, b):
    if a > b:
        return b, a
    if a < b:
        return a, b

def comMean(a):
    x_ = []
    y_ = []
    for i in range(len(a)):
        x_.append(a[i][0])
        y_.append(a[i][1])
    x_mean = sum(x_)/len(a)
    y_mean = sum(y_)/len(a)
    return x_mean, y_mean

def divideData(k, mypoint, data):
    fix_x, fix_y = comMean(mypoint)
    index1 = []
    index2 = []
    data1 = []
    data2 = []
    for i in range(len(data)):
        y = k*(data[i][0] - fix_x) + fix_y - data[i][1]
        if y >= 0:
            index1.append(i)
            data1.append(data[i])
        if y < 0:
            index2.append(i)
            data2.append(data[i])
    return np.array(data1), np.array(data2), index1, index2, fix_x, fix_y

#禁忌函数
def ts(len_table, data, maxIter):
    '''

    :param len_table:禁忌表长度
    :param data: 现行数据
    :return: order 最优的替换
    '''
    #配置禁忌表
    ts_table = np.zeros((len(data), len(data)))
    best_path = list(range(len(data)))
    random.shuffle(best_path)
    print(best_path)
    most_beat_path = best_path
    best_dis = sitisfy(best_path, data)
    count = maxIter
    while count:
        orders = []
        change = []
        #产生邻域解
        '''for i in range(k):
            flag = 1
            while(flag):
                change_one_ = random.randint(0, len(data)-1)
                change_two_ = random.randint(0, len(data)-1)
                if change_one_ != change_two_:
                    change_one, change_two = changeMaxMin(change_one_, change_two_)
                    flag = 0
                    #if ts_table[change_one][change_two] == 0:
                        #flag = 0'''
        for i in range(len(best_path)):
            for j in range(i+1, len(best_path)):
                change_one = i
                change_two = j
                path_ly = []
                for i in best_path:
                    path_ly.append(i)
                t = path_ly[change_one]
                path_ly[change_one] = path_ly[change_two]
                path_ly[change_two] = t
                orders.append(path_ly)
                change.append([change_one, change_two])


        flag = 1
        while flag:
            best_path, best_index = findBestPath(orders, data)
            if sitisfy(best_path, data) < best_dis:
                best_dis = sitisfy(best_path, data)
                most_beat_path = best_path
                flag = 0
            elif ts_table[change[best_index][0]][change[best_index][1]] == 0:
                flag = 0
            else:
                del orders[best_index]
        upDateTS(ts_table)
        ts_table[change[best_index][0]][change[best_index][1]] = len_table
        print(count, best_dis, best_path)
        count = count - 1
    return most_beat_path

if __name__ == '__main__':
    mypoint, dpoint = creatData(2, 20)
    data = np.row_stack((mypoint, dpoint))
    k = 1
    data1, data2, index1, index2,fix_x, fix_y = divideData(k, mypoint, data)
    print(data1)
    print(data2)
    print(fix_x)
    print(fix_y)
    def C(m, n):
        m_1 = m
        for i in range(m, 0, -1):
            m_1 = m_1 * i
        n_1 = n
        for i in range(n, 0, -1):
            n_1 = n_1 * i
        n_m_1 = n-m
        for i in range(n-m, 0, -1):
            n_m_1 = n_m_1 * i
        return int(n_1/(m_1*n_m_1))
    path1 = ts(3, data1, 500)
    path2 = ts(3, data2, 500)
    x = range(200)
    y = []
    for i in range(200):
        y.append(k*(i-fix_x)+fix_y)
    plt.plot(x, y, 'r--')
    if len(data1) > 0:
        plt.scatter(data1.T[0], data1.T[1])
    if len(data2) > 0:
        plt.scatter(data2.T[0], data2.T[1])


    for i in range(len(path1)-1):
        step_one = [data1[path1[i]][0], data1[path1[i+1]][0]]
        step_two = [data1[path1[i]][1], data1[path1[i+1]][1]]
        plt.plot(step_one, step_two, 'y--')

    step_one = [data1[path1[-1]][0], data1[path1[0]][0]]
    step_two = [data1[path1[-1]][1], data1[path1[0]][1]]
    plt.plot(step_one, step_two, 'y--')

    for i in range(len(path2)-1):
        step_tree = [data2[path2[i]][0], data2[path2[i+1]][0]]
        step_four = [data2[path2[i]][1], data2[path2[i+1]][1]]
        plt.plot(step_tree, step_four, 'y--')

    step_one = [data2[path2[-1]][0], data2[path2[0]][0]]
    step_two = [data2[path2[-1]][1], data2[path2[0]][1]]
    plt.plot(step_one, step_two, 'y--')
    plt.show()

