import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_input = 784  # 输入层节点数
num_hidden = 256  # 隐藏层节点数
num_output = 10  # 输出层节点数
lr = 0.01  # 学习率
epochs = 10  # 训练轮数

def convert(imgf, labelf, outf, n):  # 将原文件读取成.csv文件
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(p_ix) for p_ix in image)+"\n")
    f.close()
    o.close()
    l.close()

class FCNN:
    def __init__(self, num_input, num_hidden, num_output, lr):
        self.input = num_input
        self.hidden = num_hidden
        self.output = num_output
        self.W1 = np.random.normal(0.0, pow(self.hidden, -0.5), (self.hidden, self.input))  # 隐藏层权重
        self.W2 = np.random.normal(0.0, pow(self.hidden, -0.5), (self.output, self.hidden))  # 输出层权重
        self.lr = lr
        self.hidden_inputs = 0  # 此处仅用于定义为类变量
        self.hidden_outputs = 0  # 此处仅用于定义为类变量
        self.final_inputs = 0  # 此处仅用于定义为类变量
        self.final_outputs = 0  # 此处仅用于定义为类变量

    def forward(self, inputs_list):
        # 进行前向传播
        inputs = np.array(inputs_list, ndmin=2).T
        self.hidden_inputs = np.dot(self.W1, inputs)
        self.hidden_outputs = my_sigmoid(self.hidden_inputs)
        self.final_inputs = np.dot(self.W2, self.hidden_outputs)
        self.final_outputs = my_sigmoid(self.final_inputs)

    def backward(self, inputs_list, targets_list):
        # 进行反向传播
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        output_errors = targets - self.final_outputs
        hidden_errors = np.dot(self.W2.T, output_errors)

        self.W2 += self.lr * np.dot((output_errors * self.final_outputs * (1.0 - self.final_outputs)), np.transpose(self.hidden_outputs))
        self.W1 += self.lr * np.dot((hidden_errors * self.hidden_outputs * (1.0 - self.hidden_outputs)), np.transpose(inputs))
        return np.linalg.norm(output_errors)

def my_sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 读取文件
# 先将原文件转化为.csv文件
convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "mnist_train.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "mnist_test.csv", 10000)
# 读取.csv文件
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

model = FCNN(num_input, num_hidden, num_output, lr)
Loss = []
print("开始在训练集上进行模型训练")
for epoch in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')  # 读取特征
        inputs = np.asfarray(all_values[1:]) / 255.0  # 对灰度图进行归一化
        # 独热编码，方便后续求导进行梯度下降
        targets = np.zeros(num_output)
        targets[int(all_values[0])] = 1
        # 前向预测
        model.forward(inputs)
        # 调用后向算法，并记录损失值
        loss = model.backward(inputs, targets)
        Loss.append(loss)
    print(f"Epoch{epoch}:loss is {loss}")

# 将训练过程中的损失可视化
x = np.arange(epochs)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x, Loss, label='FCNN_acc')  # 训练过程中的损失
plt.legend()
plt.show()

# 在测试集上进行预测
correct = 0
preds = []
for record in test_data_list:
    # 以下预测过程于训练过程处理方式相同
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = np.asfarray(all_values[1:]) / 255.0
    model.forward(inputs)
    pred = model.final_outputs
    preds.append(pred)
    label = np.argmax(pred)  # 将pred中值最大的元素下标为预测值

    if label == correct_label:
        correct += 1
print("accuray is ", correct / 10000)

# 将预测结果写入文件
dataframe = pd.DataFrame({"label": preds})
dataframe.to_csv("pred_test.csv", index=False, sep=',')