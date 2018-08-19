import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# 训练集
X = [0, 1, 2, 3, 0, 1, 2, 3]
Y = [1, 2, 3, 4, -1, 0, 1, 2]

arr = np.array([1, 2, 3, 4]).reshape((4, 1))
print(arr)

# 测试集
X_TEST = [0.5, 1.5, 2.5]
Y_TEST = [1.5, 2.5, 3.5]

plt.figure()
plt.scatter(X, Y, color='red')
plt.title('base')
plt.plot(X, Y, color='black', linewidth=1)
plt.show()

x_train = np.array(X).reshape((len(X), 1))
y_train = np.array(Y).reshape((len(Y), 1))
x_in = np.array(X_TEST).reshape((len(X_TEST), 1))

linear_regression = linear_model.LinearRegression()
linear_regression.fit(x_train, y_train)
y_out = linear_regression.predict(x_in)
plt.scatter(x_train, y_train, color='green')
plt.plot(x_in, y_out, color='black', linewidth=1)
plt.title('Training data')
plt.show()