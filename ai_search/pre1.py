# -*- coding:utf-8 -*-
import pandas as pd
import jieba
import os
import math
from nltk.corpus import stopwords
def GetListOfStopWords(filepath):
    f_stop = open(filepath)
    try:
        f_stop_text = f_stop.read()
        #f_stop_text = (f_stop_text, 'utf-8')
    finally:
        f_stop.close()
    f_stop_seg_list = f_stop_text.split('\n')

    return f_stop_seg_list
def getData():
    data = []
    n = 0
    terms = []
    stopwords = GetListOfStopWords('stopwords.txt')
    dir = os.listdir('data')
    for file in dir:
        n = n + 1
        _lrc = []
        _lrc.append(''.join(file[0:file.find('.')]))
        ofile = open('data/'+file, 'r')
        for line in ofile:
            lrc = list(jieba.cut(line))
            while ' ' in lrc:
                lrc.remove(' ')
            while '\n' in lrc:
                lrc.remove('\n')
            if len(lrc) != 0:
                _lrc.extend(lrc)
            for i in stopwords:
                if i in _lrc:
                    _lrc.remove(i)
        data.append(_lrc)
        terms.extend(_lrc)
    return n, data, terms

def getIdf(n, data, terms):
    IDF = {}
    for i in terms:
        df = 0
        for j in range(len(data)):
            if i in data[j]:
                df = df +1
        idf = math.log(n/df, 2)
        IDF[i] = idf
    return IDF

n, data, terms = getData()
IDF = getIdf(n, data, terms)
print(IDF)
querry = list(jieba.cut('林俊杰思念'))



