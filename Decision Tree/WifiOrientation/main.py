import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report #混淆矩阵分类报告
from processData import process

train_data = pd.read_csv('./data/TrainDT.csv') #原始训练数据
test_data = pd.read_csv('./data/TestDT.csv') #原始测试数据

BSSID = train_data['BSSIDLabel'] #获取所有出现的BSSID
bssid = set(BSSID) #统计BSSID的所有可能取值

#处理得到新的数据集和测试集
train_dataset, train_tag = process(train_data, bssid)
test_dataset, test_tag = process(test_data, bssid)

Dtc = DecisionTreeClassifier(max_depth=10, random_state=8) #创建决策树   
Dtc.fit(train_dataset, train_tag) #使用训练集训练决策树
#使用得到的决策树，对测试集进行预测
pre = Dtc.predict(test_dataset).tolist() 
print(classification_report(test_tag, pre)) #输出测试结果