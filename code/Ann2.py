import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
data = pd.read_csv('../data/resault.csv', sep=',')

# 提取输入和输出数据
inputs = data.iloc[:, :6].values
outputs = data.iloc[:, 6:].values

# 归一化数据
scaler = MinMaxScaler()
inputs_scaled = scaler.fit_transform(inputs)
outputs_scaled = scaler.fit_transform(outputs)

from sklearn.model_selection import train_test_split

# 划分训练集和测试集
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs_scaled, outputs_scaled, test_size=0.2, random_state=42)

import torch
import torch.nn as nn

# 转换数据为PyTorch的Tensor类型
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_outputs = torch.tensor(train_outputs, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_outputs = torch.tensor(test_outputs, dtype=torch.float32)

# 定义神经网络模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建神经网络模型实例
model = NeuralNet()

import torch.optim as optim

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt

# 训练参数
num_epochs = 4000
train_loss_history = []

# 开始训练
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_inputs)
    loss = criterion(outputs, train_outputs)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 记录训练损失
    train_loss_history.append(loss.item())
    
    # 打印当前迭代结果
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 绘制迭代结果
plt.plot(range(num_epochs), train_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
