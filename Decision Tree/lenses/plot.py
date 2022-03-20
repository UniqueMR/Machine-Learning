from matplotlib import pyplot as plt
from leafs import getNumLeafs
from depth import getTreeDepth
#绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle = "<-")
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords ='axes fraction', xytext = centerPt,
                            textcoords = 'axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
 
#创建绘图区，计算树形图的全局尺寸
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
 
#标注有向边属性
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation = 30)
 
#绘制决策函数
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
    leafNode = dict(boxstyle = "round4", fc = "0.8")
    numLeafs = getNumLeafs(myTree)
    defth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondeDict = myTree[firstStr]  
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondeDict.keys():
        if type(secondeDict[key]) is dict:
            plotTree(secondeDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondeDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
