from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
import matplotlib
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
import scienceplots
import seaborn as sns

plt.style.use('science')
plt.rcParams.update({"text.usetex": False})  # 禁用 LaTeX
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义绘制混淆矩阵的函数
def plot_confusion_matrix(cm, labels, title='Confusion Matrix', filename=None):

    """
    绘制并显示混淆矩阵的可视化图表

    该函数用于创建一个混淆矩阵的热力图，可以直观地展示模型预测结果的准确性。
    支持自定义标题，并可以选择将图表保存为图片文件。
    参数:
    cm -- 混淆矩阵，二维数组形式，表示实际类别与预测类别的对应关系
    labels -- 类别标签列表，用于在图表中显示各类别的名称
    title -- 图表的标题，默认为'Confusion Matrix'
    filename -- 可选参数，如果提供，图表将保存为指定文件名的图片

    返回:
    无返回值，直接显示或保存图表
    """
    plt.figure(figsize=(10, 8))  # 创建一个新的图形窗口，设置大小为10x8英寸
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # 创建混淆矩阵显示对象
    disp.plot()  # 绘制混淆矩阵
    plt.title(title)  # 设置图表标题
    if filename:  # 如果提供了文件名
        plt.savefig(filename)  # 将图表保存为图片文件
    plt.show()  # 显示图表

# 加载数据
data: pd.DataFrame = pd.read_csv('HR_datascience.csv')
data.head()

data.shape

# 查看是否有重复列
data.duplicated().sum()

# 删除重复行
data = data.drop_duplicates().reset_index().drop('index', axis=1)
data.duplicated().sum()

# 查看有无缺失列
data.isna().sum()

# 新增：检查目标变量left的分布
left_counts = data['left'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(left_counts, labels=['Not Left', 'Left'], autopct='%1.1f%%', startangle=90, colors=['#94dcc3', '#eef6ae'])
plt.title('Distribution of Left')
plt.savefig('figure_left_dist.svg')
plt.show()

# 我们想统计薪水分布,画饼图
df_salary = data['salary'].value_counts()
print(df_salary)
colors = ['#88bdf8', '#94dcc3', '#eef6ae']  # 设置不同扇区的颜色
fig: plt.Figure = plt.figure(figsize=(10, 8))
ax1: plt.Axes = fig.add_subplot(111)
ax1.pie(x=df_salary, labels=df_salary.index, autopct='%.2f%%', textprops={'fontsize': 18}, colors=colors,
        explode=[0, 0.0, 0.2])
plt.savefig('figure1.svg')
plt.show()

# 绘制箱型图
data_box_filitered = ['satisfaction_level', 'last_evaluation', 'number_project',
                      'average_montly_hours', 'time_spend_company', 'Work_accident',
                      'promotion_last_5years']  # 挑出想要画箱型图的数据
print(len(data_box_filitered))  # 一共7个
df2 = data[data_box_filitered]
df2.head()

# 画图
plt.cla()
fig2: plt.Figure = plt.figure(figsize=(15, 8))
colors = ['pink', 'lightblue', 'lightgreen', 'red', 'purple', 'orange', 'yellow']  # 准备好不同的颜色
# 在3*3的画布上画7张图
for i in range(1, 8):
    ax: plt.Axes = fig2.add_subplot(3, 3, i)
    ax.boxplot(
        x=df2[data_box_filitered[i - 1]],
        patch_artist=True,
        boxprops={'facecolor': colors[i - 1]},
    )
    ax.set_title(data_box_filitered[i - 1], fontsize=15)
plt.tight_layout()
plt.savefig('figure2.svg')
plt.show()

# 使用seaborn工具画图
plt.cla()
sns.pairplot(data, hue='left', kind='reg', diag_kind='kde')
plt.savefig('figure3_png.png', dpi=300)
plt.show()

# 先将字符串数据进行独热编码
data.head()
# 发现有department 和salary
# data['Department'].value_counts()
# 进行独热编码
dumm_Department = pd.get_dummies(data['Department'], prefix='Department').astype(int)
dumm_Salary = pd.get_dummies(data['salary'], prefix='salary').astype('int')
# 删除原有的对象
dropped = data.drop(['Department', 'salary'], axis=1)
# JOIN按行拼接
data_new = dropped.join([dumm_Department, dumm_Salary], how='outer')
data_new.head()

# 相关性热力矩阵
plt.cla()
fig3 = plt.figure(figsize=(10, 8))
corr = data_new.corr()
sns.heatmap(corr, annot=True, cmap='rainbow', fmt='.2f')
plt.savefig('figure4.svg')
plt.show()

X = data_new.drop('left', axis=1)
y = data_new['left'].astype(int)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
# 对部分数值型数据进行标准化
numerical_columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                     'average_montly_hours', 'time_spend_company']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
X_train.head()

logreg = LogisticRegression(max_iter=1000, random_state=42)
# 训练模型
logreg.fit(X_train, y_train)
# 预测
y_pred = logreg.predict(X_test)
# 误差矩阵
class_report = metrics.classification_report(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(class_report)
print(conf_matrix)

plt.cla()
# 使用函数绘制混淆矩阵
plot_confusion_matrix(conf_matrix, ['0', '1'], 'Confusion Matrix for Logistic Regression', 'figure5.svg')

# 随机森林model
# 使用GridSearchCV进行超参数调优
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

RF = RandomForestClassifier(random_state=42, criterion='entropy')
grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# 使用最佳参数的模型
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# 评估
class_report_rf = metrics.classification_report(y_test, y_pred_rf)
conf_matrix_rf = metrics.confusion_matrix(y_test, y_pred_rf)
print(class_report_rf)
print(conf_matrix_rf)

# 绘制混淆矩阵
plot_confusion_matrix(conf_matrix_rf, ['0', '1'], 'Confusion Matrix for Random Forest', 'figure6.svg')

# 绘制ROC曲线
y_pred_rf_proba = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_rf_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Random Forest')
plt.legend(loc="lower right")
plt.savefig('figure_roc.svg')
plt.show()

# 拿到主要特征
feature_importances = best_rf.feature_importances_
# 创建dataframe,储存重要影响因子
features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})
# 对影响因子进行排序
features_df = features_df.sort_values(by='Importance', ascending=False)
# 绘图
plt.cla()
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('figure7.svg')
plt.show()