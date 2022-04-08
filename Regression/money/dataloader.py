import pandas as pd
import numpy as np 

def load_BankNodeData():
    '''
    加载钞票数据集的训练集
    :return: 训练集的特征，训练集的label
    '''
    #导入训练集
    df = pd.read_csv(r'./data/train.txt', header=None)

    #导入训练数据
    trainSet = np.array(df.loc[:][[0, 1, 2, 3]]) 
    print('train set: \n', trainSet)

    #导入训练标签
    labels = df.loc[:][4].values 
    labels = np.where(labels == 1, 1, -1) #[1,0]转[1,-1]
    print('lebel values: \n', labels)
    return trainSet, labels