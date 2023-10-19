import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epochs = 10  # 计算马氏距离时的训练轮数
lr = 0.18  # 学习率
num_feature = 4  # 特征数

# 欧氏距离实现KNN分类
def my_KNN_1(X, X_train, Y_train, K):  # X为所要进行预测的数据集，K为近邻数
    preds = []
    for sample in X:
        distance = []
        for i in X_train:
            distance.append(np.linalg.norm(sample - i))  # 计算欧氏距离
        label_order =np.argsort(distance)
        num_label = np.zeros(3)  # 记录K个近邻点中三个label的数量
        for i in range(K):
            num_label[Y_train[label_order[i]]] += 1
        preds.append(np.argsort(-num_label)[0])  # 得到3类中数量最多的label并作为预测结果
    return preds
# 马氏距离实现KNN分类
def train_A(X_train, Y_train):
    num = X_train.shape[0]  # 计算样本个数
    np.random.seed(10)
    A = np.random.randn(num_feature, num_feature)  # 初始化正交基A

    for epoch in range(epochs):
        derivative = np.zeros((num_feature, num_feature))  # 初始化导数
        f = 0  # 目标函数
        for i in range(num):
            sigma_k_deno = 0  # p_ij, p_ik的分母
            for k in range(num):
                if k == i:
                    pass
                else:
                    sigma_k_deno += np.exp(-(np.linalg.norm((X_train[i] - X_train[k]) @ A)**2))
            for j in range(num):
                if Y_train[i] != Y_train[j] or i == j:
                    pass
                else:
                    sigma_k_mol = np.zeros((num_feature, num_feature))  # 关于k求和的分子之和（分母相同）
                    for k in range(num):
                        if k == i or k == j:
                            pass
                        else:
                            sigma_k_mol += np.exp(-(np.linalg.norm((X_train[i] - X_train[k]) @ A)**2)) * (
                                    np.outer(X_train[i] - X_train[k], X_train[i] - X_train[k])
                                    - np.outer(X_train[i] - X_train[j], X_train[i] - X_train[j]))
                    p_ij = np.exp(-(np.linalg.norm((X_train[i] - X_train[j]) @ A)**2)) / sigma_k_deno
                    derivative += p_ij * sigma_k_mol / sigma_k_deno
                    f += p_ij
        print(f"Epoch{epoch}: 目标函数f的值为{f}")
        A -= 2 * lr * A @ derivative
    return A
def my_KNN_2(X, X_train, Y_train, A, K):
    preds = []
    for sample in X:
        distance = []
        for i in X_train:
            distance.append(np.sqrt(abs((sample - i) @ A @ (sample - i))))  # 计算欧氏距离
        label_order =np.argsort(distance)
        num_label = np.zeros(3)  # 记录K个近邻点中三个label的数量
        for i in range(K):
            num_label[Y_train[label_order[i]]] += 1
        preds.append(np.argsort(-num_label)[0])  # 得到3类中数量最多的label并作为预测结果
    return preds

# 读取文件
X_train = pd.read_csv("train.csv").drop("label", axis=1).to_numpy()
Y_train = pd.read_csv("train.csv")["label"].to_numpy()
X_val = pd.read_csv("val.csv").drop("label", axis=1).to_numpy()
Y_val = pd.read_csv("val.csv")["label"].to_numpy()
X_test = pd.read_csv("test_data.csv").to_numpy()

accuray_1 = []
accuray_2 = []
A = train_A(X_train, Y_train)
for K in range(1, 10):
    pred_1 = my_KNN_1(X_val, X_train, Y_train, K)
    pred_2 = my_KNN_2(X_val, X_train, Y_train, A, K)
    acc_1 = 0
    acc_2 = 0
    for i in range(Y_val.shape[0]):
        if pred_1[i] == Y_val[i]:
            acc_1 += 1
        if pred_2[i] == Y_val[i]:
            acc_2 += 1
    accuray_1.append(acc_1 / Y_val.shape[0])
    accuray_2.append(acc_2 / Y_val.shape[0])

X = np.arange(1, 10)
plt.xlabel('The value of K')
plt.ylabel('accuracy')
plt.plot(X, accuray_1, label="Euclidean distance")
plt.plot(X, accuray_2, label="Mahalanobis distance")
plt.legend()
plt.show()

# 在测试集上进行运行
K_1 = 4  # 根据可视化结果得到使用欧式距离是选取K = 4预测结果最好
K_2 = 3  # 根据可视化结果得到使用马氏距离是选取K = 5预测结果最好
pred_test_1 = my_KNN_1(X_test, X_train, Y_train, K_1)
pred_test_2 = my_KNN_2(X_test, X_train, Y_train, A, K_2)
# 将预测结果写入文件
dataframe = pd.DataFrame({"label": pred_test_1})
dataframe.to_csv("task1_test_prediction.csv", index=False, sep=',')
dataframe = pd.DataFrame({"label": pred_test_2})
dataframe.to_csv("task2_test_prediction.csv", index=False, sep=',')