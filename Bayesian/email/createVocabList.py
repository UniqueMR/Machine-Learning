def creatVocabList(dataset):
    """
        将文档矩阵中的所有词构成词汇表
        param:
            dataset: 所有邮件的词语列表
    """
    vocabSet=set([])
    for document in dataset:
        #通过求并集的方式，得到所有邮件中所有可能出现的词汇
         vocabSet=vocabSet|set(document)  
    return list(vocabSet)