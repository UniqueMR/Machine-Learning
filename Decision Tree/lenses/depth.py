#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0 
    firstStr = list(myTree.keys())[0] 
    secondDict = myTree[firstStr] 
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   
            thisDepth = 1
        if thisDepth > maxDepth: 
            maxDepth = thisDepth
    return maxDepth