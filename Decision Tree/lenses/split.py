"""按照给定特征划分数据集"""
def splitDataSet(dataset, axis, value):
    '''
    parameters:
        dataset: 待划分的数据集
        axis: 用来划分数据集的特征id
        value: 当前特征下的取值
    '''
    retDataset = [] #存储划分后的数据集
    for feature in dataset:
        if feature[axis] == value:
            reduced = feature[:axis] 
            #去掉特征axis
            reduced.extend(feature[axis+1:]) 
            #将符合条件的加入返回列表
            retDataset.append(reduced)
    return retDataset