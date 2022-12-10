import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold #k折交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.metrics import make_scorer
import os

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


df = data[['地址','价格','所属地区','房屋朝向','所在楼层','装修情况','套内面积','是否配备电梯','交易权属','多少人关注','房子总面积','建筑年代','楼型']]

df.drop(['所在楼层','房屋朝向','装修情况','所属地区','楼型','是否配备电梯','交易权属','地址'],axis=1,inplace=True)

y = df['价格'].values.tolist()
x = df.drop(['价格'],axis=1)
x = x.values.tolist()
corrDf = df.corr()   #这里多写了一点，是相关系数，运行起来较慢可以删除
print(corrDf['价格'].sort_values(ascending =False))
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.15,random_state=0)

def performance_metric(y_true, y_predict):
    from sklearn.metrics import r2_score
    score = r2_score(y_true,y_predict)
    return score

model = fit_model(train_x,train_y)
def model_imge(model):
    #joblib.dump(model,'./keep_model/model.pkl')
    pre_y = model.predict(test_x)
    #计算决策系数r方
    r2 = performance_metric(test_y,pre_y)  
    print(r2)
    plt.figure(figsize=(10,9),dpi=100)
    plt.plot(test_y,label='实际值')
    plt.plot(pre_y,label='预测值')
    plt.legend()
    plt.title('GBRT模型实际值与预测值对比')
    plt.show()
model_imge(model)
