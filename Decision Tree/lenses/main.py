from creatTree import createTree
from plot import createPlot

if __name__ == '__main__':
    fr = open('./data/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    myTree_lenses = createTree(lenses, lensesLabels)
    createPlot(myTree_lenses)