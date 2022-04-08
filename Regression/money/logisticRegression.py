import numpy as np
from numpy import random

def logisticFunction(inputV):
    '''
    logistic函数
    :param inputV: logistic函数输入
    :return: logistic函数值
    '''
    return 1.0 / (1.0 + np.exp(-inputV))

def logisticRegression_SGD(trainSet, labels, eta = 0.01, max_iter = 5000):
    '''
    :param trainSet: 训练集
    :param labels: 训练集的y值
    :param eta: 学习率，步长
    :param iterTime: 最大迭代次数
    :return: 权重；权重更新记录，用户观测是否收敛
    '''
    sampleSize = len(labels) #样本总数
    featureSize = len(trainSet[0]) + 1 #特征总数
    #随机分配初始权重(w[0]为偏置，w[1:]分别对应各个特征的权重)
    weights = random.rand(featureSize) 
    weightsRecord = [[x] for x in weights] # 权重更新记录
    print('initial weights: ', weights)
    count = 0
    #使用SGD更新权重
    while(count < max_iter):
        sample = random.randint(0, sampleSize - 1) #随机样本索引
        #梯度计算  
        update = logisticFunction(-labels[sample]
         * (np.dot(weights[1:], trainSet[sample]) + weights[0]))
        #对权重进行更新
        weights[1:] = weights[1:] - \
            eta * update * (-labels[sample] * trainSet[sample])
        weights[0] = weights[0] - eta * update * (-labels[sample])
        count += 1
        #每更新500次记录权重
        if count % 500 == 0:
            for i in range(featureSize):
                weightsRecord[i].append(weights[i])
    #将记录权重保存
    fout = open(r'./data/weightRecord.txt', 'w', encoding='utf-8')
    for i in range(featureSize):
        fout.write(','.join([str(i) for i in weightsRecord[i]]) + '\n')
    fout.close()
    return weights, weightsRecord