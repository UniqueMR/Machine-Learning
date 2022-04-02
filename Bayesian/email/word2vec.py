def setOfWords2Vec(vocabList,inputSet): 
    """
        将某一文档转换成词向量（词集模型），邮件中出现的单词标记为1
        ，该向量中所含数值数目与词汇表中词汇数目相同。
        param:
            vocabList: 所有邮件的词汇表（词汇集合）
            inputSet: 某份邮件的词汇列表
    """
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            #1表示词向量该位置对应的词汇表中的单词，出现在inpust文档中
            returnVec[vocabList.index(word)]=1
    return returnVec

def bagOfWords2Vec(vocabList,inputSet): 
    """
        将某一文档转换成词向量（词集模型），
        记录邮件中各个单词出现的次数
        ，该向量中所含数值数目与词汇表中词汇数目相同。
        param:
            vocabList: 所有邮件的词汇表（词汇集合）
            inputSet: 某份邮件的词汇列表
    """
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec
