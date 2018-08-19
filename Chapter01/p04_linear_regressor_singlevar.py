import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pickle
# 1.4.2.01 加载数据
filename = 'data_singlevar.txt'
X = []
Y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        Y.append(yt)

# 1.4.2.02 数据分为训练数据集与测试数据集
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# 训练数据
X_train = np.array(X[:num_training]).reshape((num_training, 1))
Y_train = np.array(Y[:num_training])

# 测试数据
X_test = np.array(X[num_training:]).reshape((num_test, 1))
Y_test = np.array(Y[num_training:])

# 1.4.2.03 创建线性回归对象
linear_regressor = linear_model.LinearRegression()

# 用训练数据集训练模型
linear_regressor.fit(X_train, Y_train)

# 1.4.2.04 查看如何拟合
Y_train_pred = linear_regressor.predict(X_train)
print(Y_train_pred)
plt.scatter(X_train, Y_train, color='green')
plt.plot(X_train, Y_train_pred, color='black', linewidth=1)
plt.title('Training data')
plt.show()

# 1.4.2.07 用模型对测试数据集进行预测并绘制
Y_test_pred = linear_regressor.predict(X_test)
print(Y_test_pred)
plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, Y_test_pred, color='black', linewidth=1)
plt.title('Test data')
plt.show()

print("\nY_test = ", Y_test)
print("\nY_test_pred = ", Y_test_pred)

# 1.5.2 计算回归准确性
print("Mean absolute error = ", round(sm.mean_absolute_error(Y_test, Y_test_pred), 2))
print("Mean squared error = ", round(sm.mean_squared_error(Y_test, Y_test_pred), 2))
print("Median absolute error = ", round(sm.median_absolute_error(Y_test, Y_test_pred), 2))
print("Explained variance score = ", round(sm.explained_variance_score(Y_test, Y_test_pred), 2))
print("R2 score = ", round(sm.r2_score(Y_test, Y_test_pred), 2))

# 1.6.01 保存模型数据
output_model_file = 'saved_model.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(linear_regressor, f)

# 1.6.02 加载模型数据
with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)

Y_test_pred_new = model_linregr.predict(X_test)
print("\nY_test = ", Y_test)
print("\nY_test_pred_new = ", Y_test_pred_new)
print("\nNew mean absolute error = ", round(sm.mean_absolute_error(Y_test, Y_test_pred_new), 2))

# 1.7 创建岭回归器
ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)
ridge_regressor.fit(X_train, Y_train)
Y_test_pred_ridge = ridge_regressor.predict(X_test)