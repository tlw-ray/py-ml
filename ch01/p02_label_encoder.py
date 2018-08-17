from sklearn import preprocessing

# 1.3 标记编码方法
label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']

label_encoder.fit(input_classes)
print("\nClass mapping: ")
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)

# 1.3.06 标记编码转换
labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print("\nLabels = ", labels)
print("Encoded labels = ", list(encoded_labels))

# 1.3.07 编码数字转换为文字
encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("\nEncoded labels = ", encoded_labels)
print("Decoded labels = ", list(decoded_labels))

