import numpy as np

def dataloader(url):
    '''
    读取文本
    '''
    with open(url, "rb") as fr:
        #以空格为单位对文本进行划分
        data_n = [inst.decode().strip().split(' ') for inst in fr.readlines()]
        data = [[int(element) for element in line] for line in data_n]
    return np.array(data)