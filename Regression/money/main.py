from dataloader import load_BankNodeData
from logisticRegression import logisticRegression_SGD
from plot import plotWeightTrend
from classifier import performence_BankNodeData

if __name__ == '__main__':
    trainSet, labels = load_BankNodeData()
    weights, weights_record = logisticRegression_SGD(trainSet, labels, 0.1, 1500000)
    plotWeightTrend()
    performence_BankNodeData(weights)