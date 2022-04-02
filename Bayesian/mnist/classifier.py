import numpy as np
import math

def bayesClassifier(test_x, prioriP, posteriorP):
    '''
    使用贝叶斯分类器进行分类(极大似然估计)
    param:
        test_x: 待预测图片
        prioriP: 先验概率列表
        posteriorP: 条件概率分布列表
    '''
    oldShape = test_x.shape
    #将测试图片展开成一维形式
    test_x.resize(oldShape[0]*oldShape[1])
    classP = np.empty(10)
    #计算数字0~9各自可能出现的概率
    for j in range(10):
        #根据朴素贝叶斯，计算各个像素取值的条件概率乘积
        # （转化为log形式后相加）
        temp = sum([math.log(1-posteriorP[j][x]) if test_x[x] ==
                    0 else math.log(posteriorP[j][x]) 
                    for x in range(test_x.shape[0])])
        #将条件概率和先验概率相乘（转化为log形式后相加）
        classP[j] = np.array(math.log(prioriP[j]) + temp) 
    test_x.resize(oldShape)
    #返回概率最大的情况下的索引值，即为得到的数字类别
    return np.argmax(classP)