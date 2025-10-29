import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import multiprocessing
from contextlib import contextmanager
from pandas import Series, DataFrame
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
import scienceplots
import seaborn as sns
from joblib import Parallel, delayed

# 配置优化
n_cores = max(1, multiprocessing.cpu_count() - 1)  # 保留一个核心给系统
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)

plt.style.use('science')
plt.rcParams.update({"text.usetex": False})
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

@contextmanager
def timer(name):
    """计时上下文管理器"""
    start = time.time()
    yield
    end = time.time()
    print(f"{name} 耗时: {end - start:.2f} 秒")

# 定义绘制混淆矩阵的函数
def plot_confusion_matrix(cm, labels, title='Confusion Matrix', filename=None):
    """
    绘制并显示混淆矩阵的可视化图表

    参数:
    cm -- 混淆矩阵，二维数组形式
    labels -- 类别标签列表，用于在图表中显示
    title -- 图表的标题，默认为'Confusion Matrix'
    filename -- 可选参数，如果提供，图表将保存为指定文件名的图片

    返回:
    无返回值，直接显示或保存图表
    """
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.close()  # 关闭图形释放内存

# 优化数据读取
def load_data_optimized(filepath):
    """优化数据加载"""
    # 指定数据类型，减少内存使用
    dtype_spec = {
        'satisfaction_level': 'float32',      # 员工满意度水平
        'last_evaluation': 'float32',         # 最后一次评估分数
        'number_project': 'int8',             # 参与项目数
        'average_montly_hours': 'int16',      # 平均每月工作小时数
        'time_spend_company': 'int8',         # 在公司工作年限
        'Work_accident': 'int8',              # 是否发生过工伤事故
        'promotion_last_5years': 'int8',      # 过去5年是否获得过晋升
        'left': 'int8'                        # 是否已离职（目标变量）
    }
    
    # 只读取需要的列
    usecols = list(dtype_spec.keys()) + ['Department', 'salary']
    
    data = pd.read_csv(filepath, dtype=dtype_spec, usecols=usecols)
    return data

# 快速评估函数
def quick_evaluate_rf(X_train, y_train, X_test, y_test):
    """使用预设的较好参数快速训练"""
    rf = RandomForestClassifier(
        n_estimators=100,           # 决策树数量
        max_depth=15,               # 树的最大深度
        min_samples_split=5,        # 分割内部节点所需的最小样本数
        min_samples_leaf=2,         # 叶节点所需的最小样本数
        n_jobs=n_cores,             # 使用并行计算的核心数
        random_state=42             # 随机种子，确保结果可重现
    )
    
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    print(f"快速评估准确率: {score:.4f}")
    return rf

# 随机森林超参数搜索优化
def optimize_rf_fast(X_train, y_train):
    """使用随机搜索替代网格搜索"""
    # 定义超参数搜索空间
    param_dist = {
        'n_estimators': [50, 100, 150],              # 决策树数量
        'max_depth': [10, 15, 20, None],             # 树的最大深度
        'min_samples_split': [2, 5],                 # 分割内部节点所需的最小样本数
        'min_samples_leaf': [1, 2],                  # 叶节点所需的最小样本数
        'max_features': ['sqrt', 'log2']             # 寻找最佳分割时考虑的特征数量
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=n_cores)
    
    # 使用随机搜索，迭代次数更少（相比网格搜索更高效）
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=20,      # 随机搜索20种参数组合
        cv=3,                           # 3折交叉验证
        scoring='accuracy',             # 以准确率为评估指标
        n_jobs=n_cores,                 # 使用并行计算的核心数
        random_state=42,                # 随机种子
        verbose=1                       # 输出搜索过程信息
    )
    
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

# 特征选择优化
def select_features_fast(X, y, k=15):
    """快速特征选择"""
    # 使用单变量统计测试选择k个最佳特征
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features

# 绘制箱型图（原始版本）
def plot_boxplots_original(data, save_path='figure2.svg'):
    """绘制原始版本的箱型图"""
    # 选择要绘制箱型图的数值变量
    data_box_filtered = ['satisfaction_level', 'last_evaluation', 'number_project',
                         'average_montly_hours', 'time_spend_company', 'Work_accident',
                         'promotion_last_5years']
    
    df2 = data[data_box_filtered]
    
    plt.cla()
    fig2 = plt.figure(figsize=(15, 8))
    # 为每个变量设置不同的颜色
    colors = ['pink', 'lightblue', 'lightgreen', 'red', 'purple', 'orange', 'yellow']
    
    # 在3x3的子图中绘制7个箱型图
    for i in range(1, 8):
        ax = fig2.add_subplot(3, 3, i)
        # 对于只有0/1值的变量使用不同的可视化方法
        if data_box_filtered[i-1] in ['Work_accident', 'promotion_last_5years']:
            # 使用条形图而不是箱型图来显示0/1变量的分布
            values = df2[data_box_filtered[i-1]]
            unique_vals, counts = np.unique(values, return_counts=True)
            bars = ax.bar(unique_vals, counts, color=colors[i-1], alpha=0.7)
            ax.set_title(data_box_filtered[i-1], fontsize=15)
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            # 在条形图上添加数值标签
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       str(count), ha='center', va='bottom')
        else:
            ax.boxplot(
                x=df2[data_box_filtered[i-1]],
                patch_artist=True,
                boxprops={'facecolor': colors[i-1]},  # 设置箱型图颜色
            )
            ax.set_title(data_box_filtered[i-1], fontsize=15)  # 设置子图标题
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 绘制相关性热力图（原始版本）
def plot_correlation_heatmap_original(data_new, save_path='figure4.svg'):
    """绘制原始版本的相关性热力图"""
    plt.cla()
    fig3 = plt.figure(figsize=(10, 8))
    # 计算所有特征之间的相关系数
    corr = data_new.corr()
    # 使用热力图可视化相关性矩阵，颜色深浅表示相关性强弱
    sns.heatmap(corr, annot=True, cmap='rainbow', fmt='.2f')
    plt.savefig(save_path)
    plt.close()

# 绘制配对图（原始版本）
def plot_pairplot_original(data, save_path='figure3.png'):
    """绘制原始版本的配对图"""
    plt.cla()
    # 使用seaborn绘制配对图，hue='left'表示根据不同离职状态使用不同颜色
    sns.pairplot(data, hue='left', kind='reg', diag_kind='kde')
    plt.savefig(save_path, dpi=300)
    plt.close()

# 并行处理箱型图
def plot_boxplots_parallel(data, columns, save_path='figure2.svg'):
    """并行绘制箱型图"""
    def plot_single_box(col, ax, color):
        ax.boxplot(data[col], patch_artist=True, boxprops={'facecolor': color})
        ax.set_title(col, fontsize=12)
        ax.set_ylabel('')
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    axes_flat = axes.ravel()
    colors = sns.color_palette("husl", len(columns))
    
    # 并行处理绘图
    Parallel(n_jobs=n_cores)(
        delayed(plot_single_box)(col, axes_flat[i], colors[i])
        for i, col in enumerate(columns[:len(axes_flat)])
    )
    
    # 隐藏多余的子图
    for i in range(len(columns), len(axes_flat)):
        fig.delaxes(axes_flat[i])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 主程序流程
def optimized_workflow():
    """优化的完整工作流程"""
    
    with timer("数据加载"):
        data = load_data_optimized('HR_datascience.csv')
    
    with timer("数据预处理"):
        # 优化重复值处理
        data = data.drop_duplicates(ignore_index=True)
        
        # 检查有无缺失列
        print("缺失值统计:")
        print(data.isna().sum())
        
        # 新增：检查目标变量left的分布（图1：离职员工分布饼图）
        # 这张图显示公司员工的离职情况，帮助我们了解数据集中的类别分布是否平衡
        left_counts = data['left'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(left_counts, labels=['Not Left', 'Left'], autopct='%1.1f%%', startangle=90, colors=['#94dcc3', '#eef6ae'])
        plt.title('Distribution of Left')
        plt.savefig('figure_left_dist.svg')
        plt.close()
        
        # 我们想统计薪水分布,画饼图（图2：员工薪水分布饼图）
        # 这张图显示员工薪水水平的分布情况，帮助了解公司薪酬结构
        df_salary = data['salary'].value_counts()
        print(df_salary)
        colors = ['#88bdf8', '#94dcc3', '#eef6ae']
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111)
        ax1.pie(x=df_salary, labels=df_salary.index, autopct='%.2f%%', textprops={'fontsize': 18}, colors=colors,
                explode=[0, 0.0, 0.2])
        plt.savefig('figure1.svg')
        plt.close()
        
        # 绘制箱型图 (保持与原始文件一致)（图3：各数值变量的分布箱型图）
        # 这张图显示各个数值变量的分布情况，包括中位数、四分位数和异常值
        # 帮助我们了解数据的分布特征和识别潜在的异常值
        data_box_filtered = ['satisfaction_level', 'last_evaluation', 'number_project',
                             'average_montly_hours', 'time_spend_company', 'Work_accident',
                             'promotion_last_5years']
        plot_boxplots_original(data, 'figure2.svg')
        
        # 快速编码（为了生成figure4.svg，我们需要完整的编码数据）
        data_full_encoded = pd.get_dummies(data, columns=['Department', 'salary'], dtype=int)
        # 绘制相关性热力图 (保持与原始文件一致)（图4：特征相关性热力图）
        # 这张图显示各特征之间的相关性，颜色越深表示相关性越强
        # 帮助我们理解特征之间的关系，识别可能的多重共线性问题
        plot_correlation_heatmap_original(data_full_encoded, 'figure4.svg')
        
        # 绘制配对图 (保持与原始文件一致，但添加采样以提高性能)（图5：特征间关系配对图）
        # 这张图显示不同特征之间的两两关系，不同颜色代表不同的离职状态
        # 帮助我们直观地观察哪些特征对离职预测可能更重要
        if len(data) > 2000:  # 如果数据量较大，则采样
            plot_data = data.sample(2000, random_state=42)
        else:
            plot_data = data
        plot_pairplot_original(plot_data, 'figure3_png.png')
        
        # 特征和目标分离
        X = data_full_encoded.drop('left', axis=1)
        y = data_full_encoded['left']
        
        # 保存特征名称
        feature_names = X.columns.tolist()
        
        # 快速特征选择
        X_selected, selected_features = select_features_fast(X, y, k=20)
        
        # 数据划分
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 对部分数值型数据进行标准化
        # 由于SelectKBest返回的是numpy数组，我们需要使用索引来识别数值列
        # 获取原始数据中数值列的索引
        numerical_indices = [i for i, col in enumerate(feature_names) if col in selected_features and 
                            not col.startswith(('Department', 'salary'))]
        
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        if numerical_indices:  # 只有当存在数值列时才进行标准化
            X_train_scaled[:, numerical_indices] = scaler.fit_transform(X_train_scaled[:, numerical_indices])
            X_test_scaled[:, numerical_indices] = scaler.transform(X_test_scaled[:, numerical_indices])
    
    # 逻辑回归模型
    with timer("逻辑回归模型训练"):
        logreg = LogisticRegression(max_iter=1000, random_state=42)
        logreg.fit(X_train_scaled, y_train)
        y_pred = logreg.predict(X_test_scaled)
        
        # 误差矩阵
        class_report = metrics.classification_report(y_test, y_pred)
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        print(class_report)
        print(conf_matrix)
        
        # 使用函数绘制混淆矩阵（图6：逻辑回归模型的混淆矩阵）
        # 这张图直观地显示了逻辑回归模型的预测结果与实际结果的对比
        # 对角线上的值表示正确预测的数量，非对角线上的值表示错误预测的数量
        plot_confusion_matrix(conf_matrix, ['0', '1'], 'Confusion Matrix for Logistic Regression', 'figure5.svg')
    
    # 随机森林模型
    with timer("随机森林模型训练"):
        # 使用快速评估
        print("快速模型训练...")
        rf_quick = quick_evaluate_rf(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # 使用优化的超参数搜索
        print("使用随机搜索进行超参数优化...")
        best_rf = optimize_rf_fast(X_train_scaled, y_train)
        y_pred_rf = best_rf.predict(X_test_scaled)
        
        # 评估
        class_report_rf = metrics.classification_report(y_test, y_pred_rf)
        conf_matrix_rf = metrics.confusion_matrix(y_test, y_pred_rf)
        print(class_report_rf)
        print(conf_matrix_rf)
        
        # 绘制混淆矩阵（图7：随机森林模型的混淆矩阵）
        # 这张图直观地显示了随机森林模型的预测结果与实际结果的对比
        # 通常比逻辑回归模型更准确，因为随机森林是集成学习方法
        plot_confusion_matrix(conf_matrix_rf, ['0', '1'], 'Confusion Matrix for Random Forest', 'figure6.svg')
        
        # 绘制ROC曲线（图8：随机森林模型的ROC曲线）
        # 这张图展示了模型在不同阈值下的真正例率和假正例率
        # AUC值越接近1，表示模型性能越好
        y_pred_rf_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
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
        plt.close()
        
        # 拿到主要特征
        feature_importances = best_rf.feature_importances_
        # 创建dataframe,储存重要影响因子
        features_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': feature_importances
        })
        # 对影响因子进行排序
        features_df = features_df.sort_values(by='Importance', ascending=False)
        # 绘图（图9：特征重要性条形图）
        # 这张图显示了各个特征对模型预测的重要性排序
        # 帮助我们理解哪些因素对员工离职影响最大
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=features_df.head(15))
        plt.title('Feature Importance in Random Forest Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.savefig('figure7.svg')
        plt.close()
        
        return best_rf, selected_features

if __name__ == "__main__":
    model, features = optimized_workflow()
    print(f"最终选择的特征: {list(features)}")