import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# 1.4.2.01 加载数据
# filename = 'data_singlevar.txt'
# X = []
# Y = []
# with open(filename, 'r') as f:
#     for line in f.readlines():
#         xt, yt = [float(i) for i in line.split(',')]
#         X.append(xt)
#         Y.append(yt)

X = [0, 0, 0, 0, 1, 2]
Y = [0, 1, 2, 0, 0, 0]

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
plt.figure()
plt.scatter(X_train, Y_train, color = 'green')
plt.plot(X_train, Y_train_pred, color = 'black', linewidth = 1)
plt.title('Training data')
plt.show()

# 1.4.2.07 用模型对测试数据集进行预测并绘制

Y_test_pred = linear_regressor.predict(X_test)
print(Y_test_pred)
plt.scatter(X_test, Y_test, color = 'green')
plt.plot(X_test, Y_test_pred, color = 'black', linewidth = 1)
plt.title('Test data')
plt.show()