class Bayes():    

    def __init__(self):
        self.data = []
        #i=1
        #self.data = self.loaddb("C:\\doc\\ch6\\house-votes\\hv-0%d"%i)
        #[['both', 'sedentary', 'moderate', 'yes', 'i100'],
        self.tenfold()
        #self.prior, self.p_b = self.count_num()
        
    def loaddb(self,filename):
        data_list = []
        with open (filename, "r") as fileobject:
            for line in fileobject.readlines():
                data_list.append(line.strip("\n").split("\t"))
        return data_list
    
    def count_num(self):
        prior = {}
        p_b = {}
        for line in self.data:
            #计数条件概率
            if line[0] not in p_b:
                p_b[line[0]] = {}
            for i in range(len(line)):
                if i not in p_b[line[0]]:
                    p_b[line[0]][i] = {}
                if line[i] not in p_b[line[0]][i]:
                    p_b[line[0]][i][line[i]] = 0
                #"y","n"默认为0
                if "y" not in p_b[line[0]][i]:
                    p_b[line[0]][i]["y"] = 0
                if "n" not in p_b[line[0]][i]:
                    p_b[line[0]][i]["n"] = 0
                p_b[line[0]][i][line[i]] += 1                
            #计数 Y,N的先验概率
            for i in range(len(line)):
                if i not in prior:
                    prior[i] = {}
                if line[i] not in prior[i]:
                    prior[i][line[i]] = 0
                prior[i][line[i]] += 1        
        # Y,N的先验概率    
        for value in prior.values():
            for category, num in value.items():
                value[category] = round(num / len(self.data), 3)
        # 条件概率 ， 也可以通过 高斯分布
        for cate, value_cate in p_b.items():
            for num, value in value_cate.items():
                if num != 0 :
                    for k, v in value.items():
                        #value[k] = v / value_cate[0][cate] 
                        #m=2,p参数 machine learning
                        value[k] = round((v + 2 * prior[num][k]) / (value_cate[0][cate] + 2) , 3)
        return prior, p_b
    
    #为什么要叫“朴素贝叶斯”呢？我们之所以能将多个概率进行相乘是因为这些概率都是具有独立性的。
    #P(i100|健康，中等水平、热情一般，适应） = P(健康|i100)P(中等水平|i100)P(热情一般|i100)P(适应|i100)*0.4
    
    def tenfold(self):
        currency = []
        #分桶，处理数据集
        for i in range(1,11):
            if i != 10:
                self.test = self.loaddb("C:\\doc\\ch6\\house-votes\\hv-0%d"%i)
            else:
                self.test = self.loaddb("C:\\doc\\ch6\\house-votes\\hv-%d"%i)
            for k in range(1,11):
                if k != i:
                    if k != 10:
                        self.data += self.loaddb("C:\\doc\\ch6\\house-votes\\hv-0%d"%k)
                    else:
                        self.data += self.loaddb("C:\\doc\\ch6\\house-votes\\hv-%d"%k)
            self.prior, self.p_b = self.count_num()
            # 测试桶i的贝叶斯概率
            right = 0
            for line in self.test:
                predict = self.classify(line)
                if predict == line[0]:
                    right += 1
                #print(predict ,line[0])
            currency.append(right / len(self.test))             
        print(sum(currency)/len(currency))
            
    def classify(self, line):
        p_list = []
        for k1, v1 in self.p_b.items():
            mut = 1
            for i in range(len(line)):
                if i != 0:
                    mut *= v1[i][line[i]] 
            p_list.append([mut * self.prior[0][k1], k1])
        #print(max(p_list))
        return max(p_list)[1]
            
            
a = Bayes()
