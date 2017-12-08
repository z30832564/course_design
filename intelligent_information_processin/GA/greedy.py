# -*- coding: utf-8 -*-

"""greedy.py
author:carter
date:2017/12/8
贪心算法解决TSP问题
"""

import sys
import random
import math
import time
import tkinter
import threading
from functools import reduce


class MyTSP(object):
    def __init__(self, root, width=800, height=600, n=30):
        self.root = root
        self.width = width
        self.height = height
        self.n = n
        self.canvas = tkinter.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#ffffff",
            xscrollincrement=1,
            yscrollincrement=1
        )
        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.title("贪心算法解决旅行商问题")
        self.__r = 3
        self.__t = None
        self.__lock = threading.RLock()

        self.__bindEvents()
        self.new()

    def __bindEvents(self):
        self.root.bind("n", self.new)
        self.root.bind("s", self.stop)

    def title(self, s):
        self.root.title(s)

    def new(self, evt=None):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.clear()
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点图片对象
        for i in range(self.n):
            x = random.random() * (self.width - 60) + 30
            y = random.random() * (self.height - 60) + 30
            #x = random.random() * self.width
            #y = random.random() * self.height
            self.nodes.append((x, y))
            node = self.canvas.create_oval(x - self.__r,
                                           y - self.__r, x + self.__r, y + self.__r,
                                           fill="#000000",
                                           outline="#000000",
                                           tags="node",
                                           )
            self.nodes2.append(node)

        self.order = range(self.n)
        for i in range(len(self.order)-1):
            self.line(self.nodes[self.order[i]], self.nodes[self.order[i+1]])
    def distance(self, p1, p2):
        #"得到当前顺序下连线总长度"
        p1 = list(p1)
        p2 = list(p2)
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def line(self, p1, p2):
        #"将节点按 order 顺序连线"
        self.canvas.create_line(p1, p2, fill="#000000", tags="line")
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    def stop(self, evt):
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()

        self.canvas.delete("line")
        self.final_order = []
        self.final_order.append(1)
        m = len(self.order)-1
        while m:
            self.index_of_minDis = []
            self.minDis = []
            for i in self.order:
                if i not in self.final_order:
                    self.d = self.distance(self.nodes[self.final_order[-1]], self.nodes[self.order[i]])
                    self.index_of_minDis.append(i)
                    self.minDis.append(self.d)
            self.final_order.append(self.index_of_minDis[self.minDis.index(min(self.minDis))])
            print(self.final_order)
            m = m - 1
        for i in range(len(self.final_order)-1):
            self.line(self.nodes[self.final_order[i]], self.nodes[self.final_order[i+1]])
        self.line(self.nodes[self.final_order[-1]], self.nodes[self.final_order[0]])

    def mainloop(self):
        self.root.mainloop()



if __name__ == "__main__":
    MyTSP(tkinter.Tk()).mainloop()