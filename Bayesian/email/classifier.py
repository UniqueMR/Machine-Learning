import numpy as np

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass): 
    """
        用朴素贝叶斯对邮件进行分类
        param: 
            vec2Classify: 待分类的词向量（词袋形式）
            p0Vec: 正常邮件的词汇概率分布向量（对数形式）
            p1Vec: 垃圾邮件的词汇概率分布向量（对数形式）
            pClass: 垃圾邮件的先验概率
    """
    #使用朴素贝叶斯计算分类为垃圾邮件的概率（对数相加）
    p1=sum(vec2Classify*p1Vec)+np.log(pClass) 
    #使用朴素贝叶斯计算分类为正常邮件的概率（对数相加）
    p0=sum(vec2Classify*p0Vec)+np.log(1-pClass)
    if p1>p0:
        return 1  #分类为垃圾邮件
    else:
        return 0 #分类为正常邮件