from choose import chooseBestFeatureToSplit
from split import splitDataSet
from majority import majorityCnt

"""创建决策树"""
def createTree(dataset, labels):
    classList = [example[-1] for example in dataset] #获取所有样本的标签

    #递归出口1，列表标签取值全部相同
    if classList.count(classList[0]) == len(classList): 
        return classList[0] #返回标签取值

    #递归出口2，列表元素并非全部相同
    if len(dataset[0]) == 1: 
        return majorityCnt(classList) #返回出现次数最多的标签

    bestFeat = chooseBestFeatureToSplit(dataset) #找到最佳划分属性
    bestFeatLabel = labels[bestFeat] #获取label的名称
    mytree = {bestFeatLabel:{}} #初始化决策树
    del(labels[bestFeat]) #在标签列表中删除当前最优的特征
    #得到最优特征下的所有数值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues) #得到最优特征下的所有可能取值
    for value in uniqueVals:
        subLabels = labels[:] #还没有作为划分属性的特征
        #在依据当前划分属性得到的数据子集中递归创建决策树
        mytree[bestFeatLabel][value] = \
            createTree(splitDataSet(dataset, bestFeat,value), subLabels)
    return mytree #返回最终得到的决策树
