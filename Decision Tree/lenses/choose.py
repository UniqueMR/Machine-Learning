from entropy import Entropy
from split import splitDataSet

"""选择最好的数据集划分方式"""
def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1 #求样本的特征数量（减去标签）
    baseEntropy = Entropy(dataset) #计算原始香农熵
    bestInfoGain = 0.0 #最优信息增益
    bestFeature = -1 #最优特征索引     
    for i in range(numFeatures): #遍历循环所有特征
        #提取当前特征下的所有样本取值
        featList = [example[i] for example in dataset] 
        #当前特征的所有可能取值（利用set互斥的特性）
        uniqueVals = set(featList) 
        newEntropy = 0.0 #按照当前特征划分的新信息熵
        for value in uniqueVals: #计算信息增益
            #划分后特征取值均为value的子集
            subDataSet = splitDataSet(dataset, i, value)  
            #该子集规模占总数据集比例（概率）
            prob = len(subDataSet)/float(len(dataset)) 
            #添加该子集对新香农熵的贡献
            newEntropy += prob * Entropy(subDataSet)   
        infoGain = baseEntropy - newEntropy #信息增益 
        # 实时更新最佳划分特征           
        if (infoGain > bestInfoGain):        
            bestInfoGain = infoGain        
            bestFeature = i             
    return bestFeature #返回最终的最佳划分特征id
