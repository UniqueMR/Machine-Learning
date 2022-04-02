from classifier import bayesClassifier
import numpy as np

def modelEvaluation(test_x, test_y, prioriP, posteriorP):
    '''
    对贝叶斯分类器的模型进行评估
    param:
        test_x: 测试集图片
        test_y: 测试集标签
        prioriP: 先验概率列表
        posteriorP: 条件概率分布列表
    '''
    bayesClassifierRes = np.empty(test_x.shape[0])
    for i in range(test_x.shape[0]):
        bayesClassifierRes[i] = \
            bayesClassifier(test_x[i], prioriP, posteriorP)
    return bayesClassifierRes, \
        (bayesClassifierRes == test_y).sum() / test_y.shape[0]   