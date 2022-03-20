#获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0 #初始化叶子
    firstStr = list(myTree.keys())[0] 
    secondDict = myTree[firstStr]   
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':       
            numLeafs += getNumLeafs(secondDict[key]) 
        else:   
            numLeafs +=1
    return numLeafs