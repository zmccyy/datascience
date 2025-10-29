from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
import matplotlib
import joblib
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.tree._export import export_text
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import  scienceplots
import seaborn as sns
plt.style.use('science')
plt.rcParams.update({"text.usetex": False})  # 禁用 LaTeX
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

data:pd.DataFrame = pd.read_csv('HR_datascience.csv');
data.head()

data.shape

# 查看是否有重复列
data.duplicated().sum()

#删除重复行
data = data.drop_duplicates().reset_index().drop('index',axis=1)
data.duplicated().sum()

# 查看有无缺失列
data.isna().sum()

# 我们想统计薪水分布,画饼图
df_salary = data['salary'].value_counts()
print(df_salary)
colors = ['#88bdf8','#94dcc3','#eef6ae'] #设置不同扇区的颜色
fig:plt.Figure = plt.figure(figsize=(10,8))
ax1:plt.Axes = fig.add_subplot(111)
ax1.pie(x=df_salary,labels=df_salary.index,autopct='%.2f%%', textprops={'fontsize': 18},colors=colors,explode=[0,0.0,0.2])
plt.savefig('figure1.svg')
plt.show()

# 绘制箱型图
data_box_filitered = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years'] #挑出想要画箱型图的数据
print(len(data_box_filitered))#一共7个
df2 = data[data_box_filitered]
df2.head()

#画图
plt.cla()
fig2:plt.Figure = plt.figure(figsize=(15,8))
colors = ['pink', 'lightblue', 'lightgreen', 'red', 'purple', 'orange', 'yellow']# 准备好不同的颜色
#在3*3的画布上画7张图
for i in range (1,8):
    ax:plt.Axes =fig2.add_subplot(3,3,i)
    ax.boxplot(
        x = df2[data_box_filitered[i-1]],
        patch_artist=True,
        boxprops={'facecolor':colors[i-1]},
    )
    ax.set_title(data_box_filitered[i-1],fontsize=15)
plt.tight_layout()
plt.savefig('figure2.svg')
plt.show()

# 使用seaborn工具画图
plt.cla()
sns.pairplot(data,hue='left',kind='reg',diag_kind='kde')
plt.savefig('figure3_png.png',dpi=300)
plt.show()

# 先将字符串数据进行独热编码
data.head()
#发现有department 和salary
# data['Department'].value_counts()
#进行独热编码
dumm_Department = pd.get_dummies(data['Department'],prefix='Department').astype(int)
dumm_Salary = pd.get_dummies(data['salary'],prefix='salary').astype('int')
#删除原有的对象
dropped = data.drop(['Department','salary'],axis=1)
#JOIN按行拼接
data_new =dropped.join([dumm_Department,dumm_Salary],how = 'outer')
data_new.head()

#相关性热力矩阵
plt.cla()
fig3 = plt.figure(figsize=(10,8))
corr = data_new.corr()
sns.heatmap(corr,annot=True,cmap='rainbow',fmt='.2f')
plt.savefig('figure4.svg')
plt.show()

X = data_new.drop('left',axis=1)
y = data_new['left'].astype(int)
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=42)
#对部分数值型数据进行标准化
numerical_columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                     'average_montly_hours', 'time_spend_company']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
X_train.head()

logreg =LogisticRegression(max_iter=1000,random_state=42)
#训练模型
logreg.fit(X_train,y_train)
#预测
y_pred = logreg.predict(X_test)
# 误差矩阵
class_report = metrics.classification_report(y_test,y_pred)
conf_martix = metrics.confusion_matrix(y_test,y_pred)
print(class_report)
print(conf_martix)

plt.cla()
fig4:plt.Figure = plt.figure(figsize=(10,8))
display = ConfusionMatrixDisplay(confusion_matrix=conf_martix,display_labels=['0','1'])
ax1:plt.Axes  = fig4.add_subplot(111)
plt.title('Confusion Matrix')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
display.plot()
plt.savefig('figure5.svg')
plt.show()

# 随机森林model
# 寻找max_depth
d_scores = []
for i in range(1,20):
    RF = RandomForestClassifier(n_estimators=15,max_depth=i,random_state=42,criterion='entropy')
    RF.fit(X_train,y_train)
    d_scores.append(RF.score(X_test,y_test))
depth = d_scores.index(max(d_scores))
print('决策树深度: ',depth,'最优值为: ',max(d_scores))

# 按照最优深度,找最优决策树数目
n_scores = []
for i in range (1,105):
     RF = RandomForestClassifier(n_estimators=i,max_depth=14,random_state=42,criterion='entropy')
     RF.fit(X_train,y_train)
     n_scores.append(RF.score(X_test,y_test))
n_tree = n_scores.index(max(n_scores))
print('最优决策树数目: ',n_tree,'最优值为: ',max(n_scores))

#建立随机森林模型
rf = RandomForestClassifier(n_estimators=29,max_depth=14, random_state=42)
#模型训练
rf.fit(X_train, y_train)
#进行预测
y_pred_rf = rf.predict(X_test)
#报告
class_report_rf = metrics.classification_report(y_test, y_pred_rf)
conf_matrix_rf = metrics.confusion_matrix(y_test, y_pred_rf)
print(class_report_rf,conf_matrix_rf)

plt.cla()
fig5:plt.Figure = plt.figure(figsize=(10,8))
display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf,display_labels=['0','1'])
ax1:plt.Axes  = fig5.add_subplot(111)
plt.title('Confusion Matrix')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
display.plot()
plt.savefig('figure6.svg')
plt.show()

#拿到主要特征
feature_importances = rf.feature_importances_
#创建dataframe,储存重要影响因子
features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})
#对影响因子进行排序
features_df = features_df.sort_values(by='Importance', ascending=False)
#绘图
plt.cla()
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('figure7.svg')
plt.show()
