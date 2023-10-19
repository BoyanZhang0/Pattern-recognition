import numpy as np
import pandas as pd

M = 1599  # 红酒的样本数
N = 4898  # 白酒的样本数
num_train = int((M + N) * 0.8)  # 选取数据集的80%作为训练集
num_test = M + N - num_train  # 选取数据集的20%作为测试集
num_feature = 12  # 样本的特征数量
num_label = 2  # 假设红酒的label为0，白酒的label为1,一共有两个标签
learning_rate = 0.01  # 学习率
epochs = 2000  # 训练次数

# PCA降维技术
def PCA(ds):  # ds为所要降维的数据集
    # 求平均值
    m = np.zeros(num_feature)
    num_sample = ds.shape[0]
    for i in ds:
        m += i
    m /= num_sample
    # 求S矩阵
    S = np.zeros((num_feature, num_feature))
    for i in ds:
        S += np.outer(np.array(i - m), np.array(i - m))

    eigenvalue, feature_vector = np.linalg.eig(S)
    # 确定应将数据降至几维，即需要选取多少个特征值
    sigma = 0
    for i in eigenvalue:
        sigma = sigma + abs(i)
    count = 0  # 用于记录需要选取多少个特征值，才能使保留的信息超过0.99
    eigenvalue_sigma = 0
    for i in range(len(eigenvalue)):
        eigenvalue_sigma = eigenvalue_sigma + eigenvalue[i]
        count += 1
        if eigenvalue_sigma / sigma > 0.99:
            break
    print("根据所求得的特征值，选取%s个特征值即可使保留的信息超过0.99" % count)

    W = feature_vector[:count, ]  # 得到投影矩阵
    m = np.outer(np.ones(num_sample).reshape(-1, 1), m.reshape(-1, num_feature))  # 用于中心化每一个样本数据
    ds = (ds - m) @ W.T  # 对ds进行降维
    return ds
# LDA降维技术
def LDA(ds, label):  # ds为数据集的feature， label为数据集的label
    m, n = 0, (M + N)  # m为类间平均值， n为总样本数
    mi, ni = np.zeros((num_label, num_feature)), np.zeros(num_label)   # mi为类内平均值， ni为每个类中的样本数
    for i in range(n):  # 求和
        m = m + ds[i]
        mi[int(label[i])] += ds[i]
        ni[int(label[i])] += 1
    # 取平均值
    m /= n
    for i in range(num_label):
        mi[i] /= ni[i]
    # 计算类内，类间散度矩阵
    SW, SB = np.zeros((num_feature, num_feature)), np.zeros((num_feature, num_feature))
    for i in range(n):
        SW += np.outer((ds[i] - mi[int(label[i])]), (ds[i] - mi[int(label[i])]))
    for i in range(num_label):
        SB += ni[i] * np.outer((mi[i] - m), (mi[i] - m))

    eigenvalue, feature_vector = np.linalg.eig(np.dot(np.linalg.pinv(SW), SB))
    print(f"由于一共有{num_label}个label，故降维至{num_label - 1}维并选取{num_label - 1}个特征值及特征向量")
    W = feature_vector[(num_label - 1):num_label, ]
    ds = ds @ W.T
    return ds
# sigmoid函数
def sigmoid(x):  # 由于会出现np.exp(-x)值越界情况，故进行手动调整
    x_ravel = x.ravel()  # 将numpy数组展平
    length = len(x_ravel)
    y = []
    for i in range(length):
        if x_ravel[i] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[i])))
        else:
            y.append(np.exp(x_ravel[i]) / (np.exp(x_ravel[i]) + 1))
    return np.array(y).reshape(x.shape)
# 逻辑回归模型构建函数
def func_log(feature, label, lr, epochs):  # X_train为feature, Y_train为label
    num_train = feature.shape[0]
    beta = np.zeros(feature.shape[1]).reshape(-1, 1)  # 初始化模型参数
    for i in range(epochs):
        z = np.dot(feature, beta)
        h = sigmoid(z)
        error = h - label
        beta = beta - lr * np.dot(feature.T, error) / num_train

    return beta
# 预测函数
def decision(beta, feature, value): # X_val为feature, value为阈值
    Y_pred = []
    log = feature @ beta
    for i in log:
        if sigmoid(i) > value:
            Y_pred.append(1)
        else:
            Y_pred.append(0)
    return np.array(Y_pred)

# 读取文件数据
red = pd.read_csv("winequality-red.csv")
white = pd.read_csv("winequality-white.csv")
# 将每个样本个体的数据提取成数组形式
ds = []
label = []
for n in range(M):  # 提取红葡萄酒数据
    data = red.iloc[n]
    arr = []
    temp = ""
    for i in data[0]:
        if i != ";":
            temp = temp + i
        else:
            temp = float(temp)
            arr.append(temp)
            temp = ""
    arr.append(float(temp))
    ds.append(arr)
    label.append(0)
for n in range(N):  # 提取白葡萄酒数据
    data = white.iloc[n]
    arr = []
    temp = ""
    for i in data[0]:
        if i != ";":
            temp = temp + i
        else:
            temp = float(temp)
            arr.append(temp)
            temp = ""
    arr.append(float(temp))
    ds.append(arr)
    label.append(1)
ds = np.array(ds)
label = np.array(label)
data = np.c_[ds, label]
np.random.seed(2)
np.random.shuffle(data)  # 打乱样本顺序便于进行模型拟合
feature, label = data[:, :num_feature], data[:, num_feature:]  # 将数据集的feature与label分割为两个矩阵

print("对数据集进行PCA降维：")
feature_pca = PCA(feature)
print("对数据集进行LDA降维：")
feature_lda = LDA(feature, label)

# 将数据集中80%作为训练集，20%作为测试集
train_feature_pca, test_feature_pca = feature_pca[:num_train, ], feature_pca[num_train:, ]
train_feature_lda, test_feature_lda = feature_lda[:num_train, ], feature_lda[num_train:, ]
train_label, test_label = label[:num_train, ], label[num_train:, ]  #由于降维不影响label，故PCA,LDA以及原数据集都使用相同分割后的label数组

# 进行逻辑回归
beta_pca = func_log(train_feature_pca, train_label, learning_rate, epochs)
beta_lda = func_log(train_feature_lda, train_label, learning_rate, epochs)
# 进行预测
value_pca = 0.2  # pca设置阈值为0.2
value_lda = 0.75  # lda设置阈值为0.75
pred_pca = decision(beta_pca, test_feature_pca, value_pca)
pred_lda = decision(beta_lda, test_feature_lda, value_lda)
# 分别计算PCA和LDA过后预测的准确率
total = num_test
correct_pca = 0
correct_lda = 0
for i in range(total):
    if pred_pca[i] == test_label[i]:
        correct_pca += 1
    if pred_lda[i] == test_label[i]:
        correct_lda += 1
print("PCA降维后正确率为:", correct_pca / total)
print("LDA降维后正确率为:", correct_lda / total)

# 原数据集进行模型构建及预测
total = num_test
train_feature, test_feature = feature[:num_train, ], feature[num_train:, ]
beta_origin = func_log(train_feature, train_label, learning_rate, epochs)
value_origin = 0.5  # 设置阈值为0.5
pred = decision(beta_origin, test_feature, value_origin)
correct = 0
for i in range(total):
    if pred[i] == test_label[i]:
        correct += 1
print("使用原数据集进行模型拟合并预测后，正确率为:", correct / total)

# 写入PCA模型的预测结果
dataframe = pd.DataFrame({"class": pred_pca})
dataframe.to_csv("pred_pca_result.csv", index=False, sep=',')
#写入LDA模型的预测结果
dataframe = pd.DataFrame({"class": pred_lda})
dataframe.to_csv("pred_lda_result.csv", index=False, sep=',')
#写入原数据集模型的预测结果
dataframe = pd.DataFrame({"class": pred})
dataframe.to_csv("pred_origin_data_result.csv", index=False, sep=',')



