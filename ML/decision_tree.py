#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np


data_g = pd.read_table('glassscaled.txt', sep=',')
data_h = pd.read_table('heart.txt', sep='\s+')
print(data_g)



def Statistics(a,r):
    countl = 0
    counth = 0
    for i in a:
        if i <= r:
            countl++
        if i > r:
            counth++
    return countl/(countl+counth), counth/(countl+counth)
def comAndChoseI(a):
    a = list(a)
    a_unique = a.unique()
    for i in a_unique:
        pl, qh = Statistics(a, i)

