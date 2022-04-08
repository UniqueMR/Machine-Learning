import pandas as pd
import matplotlib.pyplot as plt

def plotWeightTrend():
    '''
    绘制权重更新的变化趋势
    :return: 
    '''
    df = pd.read_csv(r'./data/weightRecord.txt', header=None)
    featureSize = df.values.shape[0]
    iter_n = df.values.shape[1]
    for i in range(featureSize):
        plt.plot(range(iter_n), df.loc[i], lw = 1.5, label = 'w_' + str(i))
    plt.legend(loc = 'upper left')
    plt.show()