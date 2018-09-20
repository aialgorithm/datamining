## 前言
数据挖掘是通过对大量数据的清理及处理以发现信息，并将这原理应用于分类，推荐系统，预测等方面的过程。本文基于《面向程序员数据挖掘指南》的理解，扩展学习后总结为这入门级的文章，不足之处还请赐教。谢谢！



## 一、数据挖掘过程
1.数据选择

在分析业务需求后，需要选择应用于需求业务相关的数据。明确业务需求并选择好业务针对性的数据是数据挖掘的先决条件。

2.数据预处理

选择好的数据会有噪音，不完整等缺陷，需要对数据进行清洗，集成，转换以及归纳。

3.数据转换

根据选择的算法，对预处理好的数据转换为特定数据挖掘算法的分析模型。

4.数据挖掘

使用选择好的数据挖掘算法对数据进行处理后得到信息。

5.解释与评价

对数据挖掘后的信息加以分析解释，并应用于实际的工作领域。

## 二、数据挖掘常用算法简介

### 1.关联分析算法
关联规则在于找出具有最小支持度阈值和最小置信度阈值的不同域的数据之间的关联。在关联规则的分析算法研究中，算法的效率是核心的问题。
经典的算法有：Apriori算法，AprioriTid算法，FP-growth算法；

### 2.分类算法
决策树算法：以树形结构表示分类或者决策集合，产生规则或者发现规律。主要有ID3算法，C4.5算法， SLIQ算法， SPRINT算法， RainForest算法；

朴素Bayes分类算法：利用Bayes定理概率统计的方法，选择其中概率比较大的类别进行分类；

CBA(Classification Based on Association)算法：基于关联规则的分类算法；

MIND(Mining in Database)算法 ：采用数据库中用户定义的函数(user-definedfunction，简称UDF)来实现分类的算法；

神经网络分类算法：利用训练集对多个神经的网络进行训练，并用训练好的模型对样本进行分类；

粗集理论：粗集理论的特点是不需要预先给定某些特征或属性的数量描述，而是直接从给定问题出发，通过不可分辨关系和不可分辨类确定问题的近似域,从而找出问题中的内在规律；

遗传算法：遗传算法是模拟生物进化过程，利用复制(选择)、交叉(重组)和变异(突变)3个基本方法优化求解的技术；



### 3.聚类算法
聚类分析与分类不同，聚类分析处理的数据对象的类是未知的。聚类分析就是将对象集合分组为由类似的对象组成 的多个簇的过程。分为3类方法：

Ipartitioning method(划分方法) 给定1个N个对象或者元组的数据库，1个划分方法构建数据的K个划分，每1个划分表示1个聚簇，并且K<N。经典算法是K-MEAN(K平均值)；

hierarchical method(层次方法)
对给定数据对象集合进行层次的分解，经典算法是BIRTH算法；

grid based method(基于网格的方法) 这种方法采用一个多分辨率的网格数据结构。将空间量化为有限数目的单元，这些单元形成了网格结构，所有聚类分析都在网格上进行。常用的算法有STING，SkWAVECLUSTER和 CLIQUE；


### 小结
随着数据量的日益积累以及数据库种类的多样化，各种数据挖掘方法作用范围有限，都有局限性，因此采用单一方法难以得到决策所需的各种知识。但它们的有机组合具有互补性，多方法融合将成为数据挖掘算法的发展趋势。


## 三、数据挖掘算法实现
### 1、相关知识

#### (1)距离度量：在数据挖掘中需要明确样本数据相似度，通常可以计算样本间的距离，如下为常用距离度量的介绍。
样本数据以：
![样本数据](https://user-gold-cdn.xitu.io/2018/9/6/165acaff8aac01cf?w=605&h=362&f=png&s=12159)

![坐标](https://user-gold-cdn.xitu.io/2018/9/6/165ad72a029f3d22?w=605&h=362&f=png&s=12159)


**曼哈顿距离：** 也称曼哈顿街区距离，就如从街区的一个十字路口点到另一个十字路口点的距离，
二维空间（多维空间按同理扩展）用公式表示为
![](https://user-gold-cdn.xitu.io/2018/9/6/165af852e320fea3?w=300&h=30&f=png&s=500)

![](https://user-gold-cdn.xitu.io/2018/9/6/165af7c0321440e0?w=605&h=356&f=png&s=15230)


**欧氏距离**：表示为点到点的距离。二维空间（多维空间按同理扩展）的公式表示为
![](https://user-gold-cdn.xitu.io/2018/9/6/165af8375b661141?w=846&h=99&f=png&s=23089)

![](https://user-gold-cdn.xitu.io/2018/9/6/165af84b783a086b?w=596&h=356&f=png&s=15492)

**闵可夫斯基距离**：是一组距离方法的概括，当 p=1 既是曼哈顿距离，当 p=2 既是欧氏距离。当p越大，单一维度的差值对整体的影响就越大。
![](https://user-gold-cdn.xitu.io/2018/9/6/165af8eebd55abee?w=155&h=57&f=png&s=2367)

闵可夫斯基距离（包括欧氏距离，曼哈顿距离）的优缺点：


优点：应用广泛。

缺点：无法考虑各分量的单位以及各分量分布（方差，期望）的差异性。（其中个分量的单位差异可以使用数据的标准化来消除，下面会有介绍。）

**余弦相关系数**：样本数据视为向量，通过两向量间的夹角余弦值确认相关性，数值范围[-1，1]。 -1表示负相关，0表示无关，1表示正相关。

![](https://user-gold-cdn.xitu.io/2018/9/7/165b211e1adf40e8?w=353&h=48&f=png&s=3937)
余弦相关系数的优缺点：

优点：余弦相似度与向量的幅值无关，只与向量的方向相关，在文档相似度（TF-IDF）和图片相似性（histogram）计算上都有它的身影；
而且在样本数值稀疏的时候仍可以使用。

缺点：余弦相似度受到向量的平移影响，上式如果将 x 平移到 x+1, 余弦值就会改变。(可以理解为受样本的起始标准的影响，接下来介绍的皮尔逊相关系数可以消除这个影响)

 **皮尔逊相关系数**：计算出了样本向量间的相关性，数值范围[-1，1]。
![](https://user-gold-cdn.xitu.io/2018/9/7/165b29dfceee4ae8?w=704&h=68&f=png&s=6525)

考虑计算的遍历的次数，有一个替代公式可以近似计算皮尔逊相关系数：

![](https://user-gold-cdn.xitu.io/2018/9/7/165b2a5756a5a4a3?w=533&h=189&f=png&s=25601)

 皮尔逊相关系数优点：可消除每个分量标准不同（分数膨胀）的影响，具有平移不变性和尺度不变性。
 
#### (2)数据标准化：
各分量计算距离而各分量的单位尺度差异很大，可以使用数据标准化消除不同分量间单位尺度的影响，常用的方法有三种：

**min-max 标准化**：将数值范围缩放到（0,1）,但没有改变数据分布。max为样本最大值，min为样本最小值。

![](https://user-gold-cdn.xitu.io/2018/9/7/165b2bdd4a7cd845?w=128&h=62&f=png&s=1391)
**z-score 标准化**：将数值范围缩放到0附近, 但没有改变数据分布。u是平均值，σ是标准差。

![](https://user-gold-cdn.xitu.io/2018/9/7/165b2bc6bd1995f7?w=95&h=47&f=png&s=1114)
**修正的标准z-score**：修正后可以减少样本数据异常值的影响。将z-score标准化公式中的均值改为中位数，将标准差改为绝对偏差。
![](https://user-gold-cdn.xitu.io/2018/9/13/165d1d0e453e3047?w=173&h=99&f=png&s=5152)
其中asd绝对偏差：u为中位数，card(x)为样本个数
![](https://user-gold-cdn.xitu.io/2018/9/13/165d19a89ad874e4?w=231&h=75&f=png&s=8321)
#### (3) 算法的效果评估：

十折交叉验证：将数据集随机分割成十个等份，每次用9份数据做训练集，1份数据做测试集，如此迭代10次。十折交叉验证的关键在于较平均地分为10份。

N折交叉验证又称为留一法：用几乎所有的数据进行训练，然后留一个数据进行测试，并迭代每一数据测试。留一法的优点是：确定性。




### 2、算法入门——协同过滤推荐算法

#### 代码实现、数据集及参考论文 [电影推荐——基于用户、物品的协同过滤算法](https://github.com/liaoyongyu/datamining/tree/master/recommendation_algorithms)
```
...
示例：
r = Recommendor()

print("items base协同推荐 slope one")
#items base协同推荐算法 Slope one
r.slope_one_recommendation('lyy')

print("items base协同推荐 cos")
#items base协同推荐算法  修正余弦相似度 
r.cos_recommendation('lyy')

print("users base协同推荐")
#userbase协同推荐算法 
r.user_base_recommendation("lyy")

```
#### (1)基于用户的协同推荐算法
这个方法是利用相似用户的喜好来进行推荐：如果要推荐一个乐队给你，会查找一个和你类似的用户，然后将他喜欢的乐队推荐给你。

算法的关键在于找到相似的用户，迭代计算你与每个用户对相同乐队的评分距离，来确定谁是你最相似的用户，距离计算可以用曼哈顿距离，皮尔斯相关系数等等。
![](https://user-gold-cdn.xitu.io/2018/9/7/165b33cfd31f1d8a?w=648&h=257&f=png&s=58602)
基于用户的协同推荐算法算法的缺点：

扩展性：随着用户数量的增加，其计算量也会增加。这种算法在只有几千个用户的情况下能够工作得很好，但达到一百万个用户时就会出现瓶颈。稀疏性：大多数推荐系统中，物品的数量要远大于用户的数量，因此用户仅仅对一小部分物品进行了评价，这就造成了数据的稀疏性。比如亚马逊有上百万本书，但用户只评论了很少一部分，于是就很难找到两个相似的用户了。

#### (2)基于物品的协同推荐算法
基于用户的协同过滤是通过计算用户之间的距离找出最相似的用户（需要将所有的评价数据在读取在内存中处理进行推荐），并将相似用户评价过的物品推荐给目标用户。而基于物品的协同过滤则是找出最相似的物品（通过构建一个物品的相似度模型来做推荐），再结合用户的评价来给出推荐结果。

基于物品的协同推荐算法常用有如下两种：

#### 修正余弦相似度算法：
以物品的评分作为物品的属性值，通过对比物品i,j的工有的用户相对评分的计算相关性s(i,j)。与皮尔逊相关系数的原理相同，共有用户对物品的每一评分R(u,j)，R(u,i)需要减去该用户评分的平均值R(`u)而消除分数膨胀。

![](https://user-gold-cdn.xitu.io/2018/9/7/165b4596a0bf0e1f?w=399&h=128&f=png&s=17472)
修正余弦相似度的优点：通过构建物品模型的方式，扩展性好，占用内存小；消除分数膨胀的影响；

修正余弦相似度的缺点：稀疏性，需要基于用户的评分数据；

#### Slope One推荐算法：

第一步，计算平均差值：

dev(i,j)为遍历所有共有物品i，j的共有用户u的评分平均差异。

card(Sj,i(X))则表示同时评价过物品j和i的用户数。
![slopeone](https://user-gold-cdn.xitu.io/2018/9/7/165b456e6ef82636?w=276&h=74&f=png&s=9386)


第二歩，使用加权的Slope One算法：

PWS1(u)j表示我们将预测用户u对物品j的评分。

求合集i属于S(u)-j,用户u所含的所有物品i（除了j以外）。

dev(i,j)为遍历所有共有物品i，j的共有用户u的评分平均差异。

C(ji)也就是card(Sj,i(X))表示同时评价过物品j和i的用户数。
![](https://user-gold-cdn.xitu.io/2018/9/9/165bd05c5b75f331?w=320&h=116&f=png&s=12784)


Slope One算法优点：算法简单；扩展性好，只需要更新共有属性的用户评价，而不需要重新载入整个数据集。

Slope One算法的缺点：稀疏性，需要基于用户的评分数据；

### 3、分类算法
#### (1)基于物品特征值的KNN分类算法
#### 代码实现 [鸢尾花KNN分类算法](https://github.com/liaoyongyu/datamining/tree/master/classify/KNN/)
```
...

 # KNN算法
    def knn(self, oj_list):
        weight_dict = {"Iris-setosa":0.0, "Iris-versicolor":0.0, "Iris-virginica":0.0}
        for atuple in oj_list:
            weight_dict[atuple[1]] += (1.0 / atuple[0])
        rel_class = [(key, value) for key, value in weight_dict.items()]
        #print(sorted(rel_class, key=lambda x:x[1], reverse=True))
        rel_class = sorted(rel_class, key=lambda x:x[1], reverse=True)[0][0]
        return rel_class
        
...
```
前面我们讨论的协同推荐算法需要在用户产生的各种数据上面进行分析，因此也称为社会化过滤算法，而这种算法通常有数据的稀疏性，算法可扩展性以及依赖于用户的数据的缺点，而基于物品特征值分类算法可以改善这些问题。算法分为两步：

第一步、选取特征值

算法的关键在于挑取有代表区分意义的特征及分值。以Iris花的示例，选取花萼长度，	花萼宽度，花瓣长度，花瓣宽度特征值。



![](https://user-gold-cdn.xitu.io/2018/9/10/165c3bc0f875a4f2?w=468&h=193&f=png&s=13889)


第二歩、计算距离

比如计算测试集与训练集特征值之间的曼哈顿距离，得到k个最近邻后并通过加权后的结果预测分类。 

KNN分类算法的缺点：无法对分类结果的置信度进行量化；是被动学习的算法，每次测试需要需要遍历所有的训练集后才能分类。

#### (2)贝叶斯分类算法
#### 代码实现 [区分新闻类别朴素贝叶斯分类算法](https://github.com/liaoyongyu/datamining/blob/master/classify/Bayes/)
```
...
def train_data(self):
        #训练组的条件概率
        for word in self.vocabulary:
            for category,value in self.prob.items():
                if word not in self.prob[category]:
                    count = 0
                else :
                    count = self.prob[category][word]
                #优化条件概率公式
                self.prob[category][word] = (count + 1) / (self.total[category] + len(self.vocabulary)) 
                
...

```
贝叶斯分类算法是基于概率的分类算法。相比于KNN分类算法，它是主动学习的算法，它会根据训练集建立一个模型，并用这个模型对新样本进行分类，速度也会快很多。
贝叶斯分类算法的理论基础是基于条件概率的公式（应用于现实中P(X|Y&Z)不直观得出，而P(Y|X)*P(Z|X)比较直观得出），并假设已存在的子事件(y,z...实际应用中会有多个)间是相互独立的（因此也称为朴素贝叶斯），当y，z事件假设为独立便有：


![](https://user-gold-cdn.xitu.io/2018/9/12/165cb97ac388a376?w=308&h=47&f=png&s=2796)
如下举例推测买牛奶和有机食品，再会买绿茶的概率：

![](https://user-gold-cdn.xitu.io/2018/9/13/165d1ff2e5208d1e?w=1546&h=38&f=png&s=11379)


第一步：计算先验概率及条件概率

先验概率：为单独事件发生的概率，如P(买绿茶)，P(有机食品)

条件概率（后验概率）：y事件已经发生，观察y数据集后得出x发生的概率。如P(买有机食品|买绿茶)，通过以下公式计算（nc表示y数据集下x的发生频数，n为y数据集的总数）：

![](https://user-gold-cdn.xitu.io/2018/9/12/165cb8547529dd0e?w=109&h=53&f=png&s=2600)
上式存在一个缺陷，当一个条件概率 P(y|x)为0时，整体的预测结果P(x) * P(y|x) * P(z|x)只能为0，这样便不能更全面地预测。

修正后的条件概率：（公式摘自Tom Mitchell《机器学习》。m是一个常数，表示等效样本大小。决定常数m的方法有很多，我们这里可以使用预测结果的类别来作为m，比如投票有赞成和否决两种类别，所以m就为2。p则是相应的先验概率，比如说赞成概率是0.5，那p(赞成)就是0.5。）：

![](https://user-gold-cdn.xitu.io/2018/9/12/165cb83cfc13aaf1?w=152&h=60&f=png&s=4167)

第二歩：根据贝叶斯公式做出预测


![](https://user-gold-cdn.xitu.io/2018/9/12/165cb98290a241b9?w=308&h=47&f=png&s=2796)

由公式计算比较y&z事件发生下，不同x事件发生的概率差异，如得出P（x=喜欢），P（x=不喜欢） 的概率大小，预测为概率比较大的事件。
因为P(y)*p(z)在上式都一样，因此公式可以简化为计算概率最大项而预测分类：
![](https://user-gold-cdn.xitu.io/2018/9/12/165cba5c9f486819?w=308&h=31&f=png&s=2113)

贝叶斯算法的优点：能够给出分类结果的置信度；它是一种主动学习算法，速度更快。

贝叶斯算法的缺点：需要特定格式；数值型数据需要转换为类别计算概率或用高斯分布计算概率；

#### (2)逻辑回归分类算法
#### 代码实现 [区分猫的图片](https://github.com/liaoyongyu/datamining/tree/master/classify/NeutralNetwork)
注：逻辑回归分类算法待后续加入网络层，更新为神经网络分类算法。

```
...
# cost函数，计算梯度
def propagate(w, b, X, Y):
    m = X.shape[1]      
    A = sigmoid(np.dot(w.T, X) + b)            
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))        
    dw = 1 / m * np.dot(X, (A - Y).T)  
    db = 1 / m * np.sum(A - Y) 
...    
```

逻辑回归分类算法实现了输入特征向量X，而输出Y（范围0~1）预测X的分类。

第一步，得到关于X线性回归函数

可以通过线性回归得到WX + b，其中W是权重，b是偏差值。但不能用本式表述预测的值，因为输出Y的值需要在（0~1）区间；

第二歩，通过激活函数转换

激活函数的特点是可以将线性函数转换为非线性函数，并且有输出值有限，可微分，单调性的特点。本例使用sigmoid，使输出为预测值Y=sigmoid（WX+b）；

第三歩，构建Cost函数

训练W，b更好的预测真实的类别需要构建Cost代价函数，y^为sigmoid(WX+b)的预测分类值，y为实际分类值（0或者1）：

![](https://user-gold-cdn.xitu.io/2018/9/12/165cdb16135ada50?w=621&h=73&f=png&s=19581)

其中L(y^,y)称为损失函数
![](https://user-gold-cdn.xitu.io/2018/9/12/165cdba3f27220e1?w=321&h=39&f=png&s=8091)
训练的目的就是为了让L(y^,y)足够小，也就是当y实际分类值为1时，y^要尽量偏向1。y实际分类值为0时，y^尽量小接近0。

第四步，梯度下降得到Cost函数的极小值

![](https://user-gold-cdn.xitu.io/2018/9/12/165cdc1436948f7a?w=364&h=230&f=png&s=48409)
通过对W,b两个参数求偏导，不断迭代往下坡的的位置移动（对w，b值往极小值方向做优化，其中α为学习率控制下降的幅度），全局最优解也就是代价函数（成本函数）J (w,b)这个凸函数的极小值点。
![](https://user-gold-cdn.xitu.io/2018/9/12/165cdc3fca61705f?w=275&h=154&f=png&s=10722)
第五步、通过训练好的W,b预测分类。
![](https://user-gold-cdn.xitu.io/2018/9/12/165cd7df7fd9b48c?w=1036&h=804&f=png&s=175335)


### 4、聚类算法

#### (1)层次聚类
#### 代码实现 [狗的种类层次聚类](https://github.com/liaoyongyu/datamining/tree/master/cluster/hierarchical%20method)

层次聚类将每条数据都当作是一个分类，每次迭代的时候合并距离最近的两个分类，直到剩下一个分类为止。




#### (2)K-means++聚类
#### 代码实现 [Kmean++聚类](https://github.com/liaoyongyu/datamining/blob/master/cluster/Ipartitioning%20method/kmeanpp.py)
注：Kmean算法与Kmean++区别在于初始的中心点是直接随机选取k各点。
```
        ...
        #kmean初始化随机k个中心点
        #random.seed(1)
        #center = [[self.data[i][r] for i in range(1, len((self.data)))]  
                  #for r in random.sample(range(len(self.data)), k)]
            
        # Kmean ++ 初始化基于距离份量随机选k个中心点
        # 1.随机选择一个点
        center = []
        center.append(random.choice(range(len(self.data[0]))))
        # 2.根据距离的概率选择其他中心点
        for i in range(self.k - 1):
            weights = [self.distance_closest(self.data[0][x], center) 
                     for x in range(len(self.data[0])) if x not in center]
            dp = [x for x in range(len(self.data[0])) if x not in center]
            total = sum(weights)
            #基于距离设定权重
            weights = [weight/total for weight in weights]
            num = random.random()
            x = -1
            i = 0
            while i < num :
                x += 1
                i += weights[x]
            center.append(dp[x])
        ... 

```

k-means++算法可概括为：

（1）基于各点到中心点得距离分量，依次随机选取到k个元素作为中心点：
先随机选择一个点。重复以下步骤，直到选完k个点。

计算每个数据点dp(n)到各个中心点的距离（D），选取最小的值D(dp)；
![](https://user-gold-cdn.xitu.io/2018/9/20/165f5eafc5c036a9?w=145&h=202&f=png&s=6769)
根据D(dp)距离所占的份量来随机选取下一个点作为中心点。
![](https://user-gold-cdn.xitu.io/2018/9/20/165f5eb1eb0f4f7e?w=150&h=233&f=png&s=9001)

（2）根据各点到中心点的距离分类；

（3）计算各个分类新的中心点。
重复(2、3)，直至满足条件。


------------------------------------------------------------

## 参考文献

[数据挖掘算法概述](https://github.com/liaoyongyu/datamining/blob/master/classify/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E7%AE%97%E6%B3%95%E7%A0%94%E7%A9%B6%E4%B8%8E%E7%BB%BC%E8%BF%B0.pdf)

[面向程序员数据挖掘指南](https://wizardforcel.gitbooks.io/guide-to-data-mining/content/)



