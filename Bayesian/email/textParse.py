def textParse(bigString):
    """
        对邮件中的文本进行分割，返回由一系列词语组成的列表
        param:
            bigString: 打开的邮件文本文件
    """
    import re
    listOfTokens=re.split('\W',bigString) #匹配非字母数字下划线
    #若文本中有URL，对其进行切分时，会得到很多词，
    #为避免该情况，限定字符创的长度
    return [tok.lower() for tok in listOfTokens if len(tok)>2] 