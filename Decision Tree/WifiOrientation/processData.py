def process(data, bssid):
    """对原数据进行处理，得到新的数据集"""
    """
    parameters:
        data: 原始数据
        bssid: BSSID所有可能取值组成的列表
    """
    dataset = [] #存储处理得到的数据集（不包含标签）
    tag = [] #存储处理得到的数据标签
    feature = [] #存储相同时刻检测到的所有BSSIDLabel
    sample = [] #存储处理得到的单个数据样本
    len_dataset = list(data['finLabel'])[-1] #处理得到的数据集长度
    len_data = len(data)

    i = 1 #用于索引处理得到的数据集（索引finLabel）
    j = 0 #用于索引原数据

    #当两个索引均不超出范围时，对原始数据进行遍历
    while i <= len_dataset and j < len_data:
        #对相同时刻的数据进行整合
        if data['finLabel'][j] == i:        
            feature.append(data['BSSIDLabel'][j])
            j += 1 #原始数据索引后移
        #相同时刻的数据整合完毕
        else:
            for item in bssid: #遍历BSSID的所有可能取值
                #BSSID被检测到，对应属性为1
                if item in feature:
                    sample.append(1)
                #BSSID未被检测到，对应属性为0
                else:
                    sample.append(0)
            dataset.append(sample) #加入处理得到的新数据集样本
             #加入处理得到的新数据集标签
            tag.append(data['RoomLabel'][j-1])
            sample = []
            feature = []
            i += 1

    #最后一次整合完成后的结果并没有被加入，需要再进行一次加入操作
    for item in bssid:
        if item in feature:
            sample.append(1)
        else:
            sample.append(0)
    dataset.append(sample)
    tag.append(data['RoomLabel'][j-1])
    return dataset, tag #返回处理得到的新数据集和标签列表