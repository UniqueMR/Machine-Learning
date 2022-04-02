import numpy as np 
from collections import Counter

def bayesModelTrain(train_x, train_y):
    '''
    贝叶斯分类器模型训练
    param:
        train_x: 训练集图片
        train_y: 训练集标签
    '''
    #step1: 计算先验概率
    totalNum = train_x.shape[0]
    classNumDic = Counter(train_y) #对各个标签的出现次数进行统计
    #得到各个标签的先验概率分布
    prioriP = np.array([classNumDic[i]/totalNum for i in range(10)])

    #step2: 计算类条件概率
    oldShape = train_x.shape
    #将图片像素展开（由二维变为一维）
    train_x.resize((oldShape[0], oldShape[1]*oldShape[2]))
    #创建统计像素和的列表（长度为像素个数）
    posteriorNum = np.empty((10, train_x.shape[1]))
    #创建统计各个像素位置取值为1的概率的列表（长度为像素的个数）
    posteriorP = np.empty((10, train_x.shape[1]))
    #对数字0~8分别进行统计
    for i in range(10):
        posteriorNum[i] = train_x[np.where(train_y == i)].sum(axis=0)
        # 拉普拉斯平滑
        posteriorP[i] = (posteriorNum[i] + 1) / (classNumDic[i] + 2)
    train_x.resize(oldShape) #将结果还原为图片格式
    return prioriP, posteriorP #返回先验概率和条件概率