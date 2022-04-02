import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    """
    将整数序列编码为二进制矩阵
    param:
        sequences: 读取得到的所有文档（词汇编号格式）
        dimension: 词汇表容量，默认值为10000
    """
    #创建矩阵存储每篇文章的词向量
    results = np.zeros((len(sequences), dimension))
    #使用词集模型构建词向量 
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1 #将出现的词语的索引位置为1
    return results