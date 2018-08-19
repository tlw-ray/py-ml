import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 1.9 估算房屋价格

# 1.9.2.02 加载房屋价格数据库
housing_data = datasets.load_boston()

# 1.9.2.03 通过suffle函数把数据的顺序打乱
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# 1.9.2.04 把数据分为训练数据集和测试数据集
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 1.9.2.05 选一个最大深度为4的决策树,限定决策树不变成任意深度
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train, y_train)

# 1.9.2.06 用带AdaBoost算法的决策树回归模型进行拟合
ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ab_regressor.fit(X_train, y_train)

# 1.9.2.07 评价决策树回归器的训练效果
y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)
print("\n#### Decision Tree performance ####")
print("Mean squared error = ", round(mse, 2))
print("Explained variance score = ", round(evs, 2))

# 1.9.2.08 评价AdaBoost算法改善的效果
y_pred_ab = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)
print("\n#### AdaBoost performance ####")
print("Mean squared error = ", round(mse, 2))
print("Explained variance score = ", round(evs, 2))

# 1.10 计算特征的相对重要性

# 1.10.02 定义画图函数
def plot_feature_importances(feature_importances, title, feature_names):

    # 将重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # 将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))

    # 让X坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # 画条形图
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

# 1.10.01 画出特征的相对重要性
plot_feature_importances(dt_regressor.feature_importances_, 'Decision Tree regressor', housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_, 'AdaBoost regressor', housing_data.feature_names)




