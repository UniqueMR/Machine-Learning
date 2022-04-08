import pandas as pd
import numpy as np
from logisticRegression import logisticFunction

def performence_BankNodeData(weights):
    '''
    :param weights: 模型的权重
    :return: None
    '''
    df = pd.read_csv(r'./data/test.txt', header=None)
    testSet = df.loc[:][[0, 1, 2, 3]].values # shape = 26, 4, dtype = float64
    pre = np.dot(testSet, weights[1:]) + weights[0]
    res = pre.copy()
    for i in range(pre.__len__()):
        print('predict\t:', np.where(logisticFunction(pre[i]) > 0.5, 1, 0),\
             '(', logisticFunction(pre[i]), ')')
        res[i] = np.where(logisticFunction(pre[i]) > 0.5, 1, 0)
    df['label'] = res
    df.columns = ['factor1','factor2','factor3','factor4','true/false']
    df.to_csv('./data/result.csv')

