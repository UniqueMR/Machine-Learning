from dataloader import dataloader
from vectorization import vectorize_sequences
from sklearn.linear_model import LogisticRegression
import numpy as np 

#载入数据集与测试集
train_data = dataloader("data/train_data.txt")
train_labels = dataloader("data/train_labels.txt")
test_data = dataloader("data/test_data.txt")

#构建词向量
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')

#构建伯努利贝叶斯分类器，进行训练与预测
model = LogisticRegression()
model.fit(x_train, y_train)
y_test = model.predict(x_test)
print('Training Score: %.2f' % model.score(x_train,y_train))

#保存预测结果
result = open('./data/result.txt','w')
for item in y_test:
    result.write(str(int(item))+'\n')
result.close()
