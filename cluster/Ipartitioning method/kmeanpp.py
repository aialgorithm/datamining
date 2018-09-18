#kmeans++聚类算法
# https://www.cs.cmu.edu/~./enron/

import random
import numpy as np

class KmeanppCluster():
    
    def __init__(self, k):
        self.k = k
        self.data_dict = {}
        self.data = []
        self.center_dict = {}
        self.new_center = [0.0 for k in range(self.k)]
        # load and standlize
        self.loaddata()
        # random center
        self.center = self.init_center(k)
        #数据格式初始化
        for i in range(k):
            self.center_dict.setdefault(i, {})
            self.center_dict[i].setdefault("center", self.center[i])
            self.center_dict[i].setdefault("items_dist", {})
            self.center_dict[i].setdefault("classify", [])
        
    def init_center(self, k):
        center = [[self.data[i][r] for i in range(len((self.data))) if i > 0]  
                  for r in random.sample(range(len(self.data)), k)]
        # Kmean ++
        
        return sorted(center)
    
    def get_median(self, alist):
        """get median of list"""
        tmp = list(alist)
        tmp.sort()
        alen = len(tmp)
        if (alen % 2) == 1:
            return tmp[alen // 2]
        else:
            return (tmp[alen // 2] + tmp[(alen // 2) - 1]) / 2 

    def standlize(self, column):
        median = self.get_median(column)
        asd = sum([abs(x - median) for x in column]) / len(column)
        result = [(x - median) / asd for x in column]
        return result
                    
    def loaddata(self):
        lista = []
        filename = "C:/doc/ch8/dogs.csv.txt"
        with open(filename , "r") as fileobject:
            lines = fileobject.readlines()
        header = lines[0].split(",")
        self.data = [[] for i in range(len(header))]
        for line in lines[1:]:
            line = line.split(",")
            for i in range(len(header)):
                if i == 0:
                    self.data[i].append(line[i])
                else:
                    self.data[i].append(float(line[i]))                    
        for col in range(1, len(header)):
            self.data[col] = self.standlize(self.data[col])
        #data_dict  data对应Key    
        for i in range(0, len(self.data[0])):
            for col in range(1, len(self.data)):
                self.data_dict.setdefault(self.data[0][i], [])
                self.data_dict[self.data[0][i]].append(self.data[col][i])
        # print(self.data_dict)    
            
    def kcluster(self):
        for i in range(20):
            class_dict = self.count_distance()
            self.locate_center(class_dict)
            # print(new_center,self.center)
            print(self.new_center)
            print(self.center)
            print(self.center_dict)
            print("----------------%d----------------"%i)
            
            #if self.center == self.new_center:
                #break
            #else:
                #self.center = self.new_center
            self.center = self.new_center
            if i < 19 :    
                for j in range(self.k):
                    self.center_dict[j]["center"] = self.center[j]
                    self.center_dict[j]["items_dist"] = {}
                    self.center_dict[j]["classify"] = []            
        
    #数据结构，算法原理 
    #self.center_dict{k:{{center:[]},{distance：{item：0.0}，{classify:[]}}}}
    def count_distance(self):
        min_list = [99999]
        class_dict = {}
        for i in range(1, len(self.data[0])):
            class_dict.setdefault(self.data[0][i], 0)
            min_list.append(99999)
        for k, itema in self.center_dict.items():
            #遍历column
            for lista in self.data[1:]:
                j = 0 
                #遍历row，j为center[j]，曼哈顿方法计算距离
                for i in range(0, len(lista)):
                    itema["items_dist"].setdefault(self.data[0][i], 0.0)
                    itema["items_dist"][self.data[0][i]]  += abs(lista[i] - itema["center"][j])
                j += 1
            # 分类 {item:clss}
            #error : i from 1 
            for i in range(0, len(self.data[0])):
                if itema["items_dist"][self.data[0][i]] < min_list[i]:
                    min_list[i] = itema["items_dist"][self.data[0][i]]
                    class_dict[self.data[0][i]] = k
        return class_dict
        
    def locate_center(self, class_dict):
        # class_dict {'Boston Terrier': 0, 'Brittany Spaniel': 1, 
        #加入分类的列表
        for item_name, k in class_dict.items():
            self.center_dict[k]["classify"].append(item_name)
        #print(class_dict)
        #print(self.center_dict)
        for k, item in self.center_dict.items():
            self.new_center[k] = 0.0
            for classa in item["classify"]:
                 self.new_center[k] += np.array(self.data_dict[classa])
            self.new_center[k] /= len(item["classify"])
            # error 单一中心点排序
            self.new_center[k] = list(self.new_center[k])
        # 排序，方便对比前后差异
        self.new_center = sorted(self.new_center)

            
                 
a = KmeanppCluster(3)
a.kcluster()          
    
