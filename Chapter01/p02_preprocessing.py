import numpy as np
from sklearn import preprocessing

# 1.2.2.01 均值移除
data = np.array([
    [3, -1.5, 2, -5.4],
    [0, 4, -0.3, 2.1],
    [1, 3.3, -1.9, -4.3]
])

print("\nData = ", data)

data_standardized = preprocessing.scale(data)
print("\nScale = ", data_standardized)
print("\nMean = ", data_standardized.mean(axis=0))
print("Std deviation = ", data_standardized.std(axis=0))

# 1.2.2.02 范围缩放
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print("\nMin max scaled data = ", data_scaled)

# 1.2.2.03 归一化
data_normalized = preprocessing.normalize(data, norm='l1')
print("\nL1 normalized data = ", data_normalized)

# 1.2.2.04 二值化
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("\nBinarized data = ", data_binarized)

# 1.2.2.05 独热编码
encoder = preprocessing.OneHotEncoder()
encoder.fit([
    [0, 2, 1, 12],
    [1, 3, 5, 3],
    [2, 3, 2, 12],
    [1, 2, 4, 3]
])
print("\nn_values_ = ", encoder.n_values_)
print("\nfeature_indices_is = ", encoder.feature_indices_)
encoded_vector = encoder.transform([
    [0, 3, 1, 3]
]).toarray()
print("\nEncoder vector = ", encoded_vector)
encoded_vector = encoder.transform([
    [1, 3, 2, 3]
]).toarray()
print("\nEncoder vector = ", encoded_vector)
encoded_vector = encoder.transform([
    [2, 3, 4, 3]
]).toarray()
print("\nEncoder vector = ", encoded_vector)
encoded_vector = encoder.transform([
    [2, 3, 5, 3]
]).toarray()
print("\nEncoder vector = ", encoded_vector)
