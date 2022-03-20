import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

#导入训练数据与测试数据
train_data = pd.read_csv('./data/train.csv') 
test_data = pd.read_csv('./data/test.csv')

#去除训练集中的多余属性
train_data.drop('PassengerId', axis=1, inplace=True)   
train_data.drop('Name', axis=1, inplace=True)
train_data.drop('Ticket', axis=1, inplace=True)
train_data.drop('Fare', axis=1, inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
train_data.drop('Embarked', axis=1, inplace=True)

id = test_data['PassengerId'].tolist() #获取测试集中的乘客id

#去除测试集中的多余属性
test_data.drop('PassengerId', axis=1, inplace=True)  
test_data.drop('Name', axis=1, inplace=True)
test_data.drop('Ticket', axis=1, inplace=True)
test_data.drop('Fare', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Embarked', axis=1, inplace=True)

#用数值1来代替male，用0来代替female
train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 1       
train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 0
test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 1       
test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 0

#数据中有一些是空的，用均值来填充缺失值
train_data.fillna(train_data['Age'].mean(), inplace=True)   
test_data.fillna(test_data['Age'].mean(), inplace=True)    

#划分训练集与训练标签
train = train_data.copy()
train.drop('Survived', axis=1, inplace=True)
train_tag = train_data['Survived']

#构建决策树模型，max_depth数最大深度
Dtc = DecisionTreeClassifier(max_depth=5, random_state=8)   
Dtc.fit(train, train_tag) #使用训练集对决策树进行训练      
pre = Dtc.predict(test_data).tolist() #使用决策树对测试集进行预测              

#存储测试结果
res = {'PassengerId':id, 'Survived':pre}
res = pd.DataFrame(res)
res.to_csv('./test.csv',index=False)

#决策树可视化
dot_data = export_graphviz(Dtc, \
    feature_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'], \
        class_names='Survived')
graph = graphviz.Source(dot_data)       
graph.render("tree")

