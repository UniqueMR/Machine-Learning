from textParse import textParse
from createVocabList import creatVocabList
from word2vec import bagOfWords2Vec
from classifier import classifyNB
from probability import probability
import numpy as np 

def spamTest():
    """
        使用交叉验证的方法对朴素贝叶斯分类器进行测试
        保存分类模型的词汇表以及三个概率值，避免判断时重复求值
    """
    docList = []  # 文档（邮件）矩阵
    classList = []  # 类标签列表
    #得到邮件矩阵（词语列表形式）和标签列表
    for i in range(1, 26):
        wordlist = textParse(open('data/spam/{}.txt'.format(str(i))).read())
        docList.append(wordlist)
        classList.append(1)
        wordlist = textParse(open('data/ham/{}.txt'.format(str(i))).read())
        docList.append(wordlist)
        classList.append(0)
    vocabList = creatVocabList(docList)  # 所有邮件内容的词汇表
    import pickle
    file=open('data/vocabList.txt',mode='wb')  #存储词汇表
    pickle.dump(vocabList,file)
    file.close()
    # 对需要测试的邮件，根据其词表fileWordList构造向量
    # 随机构建40训练集与10测试集
    trainingSet = list(range(50))
    testSet = []
    #采用随机抽取10个样本的方式构建测试集
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []  # 训练集
    trainClasses = []  # 训练集中向量的类标签列表
    for docIndex in trainingSet:
        # 使用词袋模式构造的向量组成训练集
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #得到垃圾邮件和正常邮件的词汇概率分布和垃圾邮件的先验概率
    p0v,p1v,pAb=probability(trainMat,trainClasses)
    #用以存储分类器的三个概率
    file=open('data/threeRate.txt',mode='wb') 
    pickle.dump([p0v,p1v,pAb],file)
    file.close()
    errorCount=0
    #使用朴素贝叶斯分类器对测试集中的每个邮件进行分类，并统计错误率
    for docIndex in testSet:
        wordVector=bagOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0v,p1v,pAb)!=classList[docIndex]:
            errorCount+=1
    return float(errorCount)/len(testSet)

if __name__ == '__main__':
    print('分类错误率为:' + str(spamTest()))