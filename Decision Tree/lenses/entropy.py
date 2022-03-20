from math import log

"""计算香农熵"""
def Entropy(dataset):
    nums = len(dataset) #获取样本数量
    labels = {} #创建字典，保存每个标签出现的次数
    for feature in dataset: #对每组样本进行统计
        currentLabel = feature[-1] #提取标签信息（最后一列）
        #如果标签不在字典中，添加标签
        if currentLabel not in labels.keys():
            labels[currentLabel] = 0
        labels[currentLabel] += 1 #label计数
    entropy = 0.0
    for key in labels: #计算香农熵
        probability = float(labels[key])/nums #标签出现的概率
        entropy -= probability * log(probability, 2) #计算香农熵
    return entropy