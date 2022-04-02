import numpy as np 

def probability(trainMatrix,trainCategory):
    """
        对两种邮件词汇概率分布进行统计
        param:
            trainMatrix: 所有文档的词向量矩阵
            trainCategory: 所有文档的类别标签所构成的向量
    """
    numTrainDocs=len(trainMatrix)  #总文档数
    numWords=len(trainMatrix[0])  #所有词的数目
    pAbusive=sum(trainCategory)/float(numTrainDocs) #垃圾邮件的先验概率
    #正常邮件的词向量的和（每个单词总共出现的次数）
    p0Num=np.ones(numWords) 
    #垃圾邮件的词向量的和（每个单词总共出现的次数）
    p1Num=np.ones(numWords) 
    p0Deom=2.0 #正常邮件词条总计数值
    p1Deom=2.0 #垃圾邮件词条总计数值
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]  #向量相加
            #所有垃圾邮件中出现的词条的总计数值
            p1Deom+=sum(trainMatrix[i]) 
        else:
            p0Num+=trainMatrix[i]
            p0Deom+=sum(trainMatrix[i])
    #在垃圾邮件条件下词汇表中单词的出现概率
    p1Vect=np.log(p1Num/p1Deom) 
    #在正常邮件条件下词汇表中单词出现的概率
    p0Vect=np.log(p0Num/p0Deom) 
    return p0Vect,p1Vect,pAbusive