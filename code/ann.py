import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv('../data/train.csv', sep=',')
input_data = data.iloc[:, :6].values
output_data = data.iloc[:, [6, 9]].values

# 数据归一化
scaler = MinMaxScaler()
input_data = scaler.fit_transform(input_data)
output_data = scaler.fit_transform(output_data)

# train_size = int(0.8 * len(data))
# train_input = input_data[:train_size]
# train_output = output_data[:train_size]
# test_input = input_data[train_size:]
# test_output = output_data[train_size:]
train_input, test_input, train_output, test_output = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# 将numpy数组转换为PyTorch张量
train_input = torch.from_numpy(train_input).float()
train_output = torch.from_numpy(train_output).float()
test_input = torch.from_numpy(test_input).float()
test_output = torch.from_numpy(test_output).float()

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

input_size = 6
output_size = 2
model = NeuralNet(input_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# num_epochs = 500
# train_losses = []

# for epoch in range(num_epochs):
#     inputs = torch.tensor(train_input, dtype=torch.float32)
#     labels = torch.tensor(train_output, dtype=torch.float32)

#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

#     train_losses.append(loss.item())
#     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


# plt.plot(range(1, num_epochs+1), train_losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()
num_epochs = 1000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # 在训练集上进行前向传播、计算损失和反向传播
    model.train()
    train_outputs_pred = model(train_input)
    train_loss = criterion(train_outputs_pred, train_output)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # 在测试集上进行前向传播和计算损失
    model.eval()
    test_outputs_pred = model(test_input)
    test_loss = criterion(test_outputs_pred, test_output)
    
    # 记录训练集和测试集的损失
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

# 绘制训练集和测试集误差图表
plt.plot(range(num_epochs), train_losses, label='Training Error')
plt.plot(range(num_epochs), test_losses, label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Error')
plt.legend()
plt.show()

test_inputs = torch.tensor(test_input, dtype=torch.float32)
with torch.no_grad():
    test_outputs = model(test_inputs).numpy()

# 反归一化
test_outputs = scaler.inverse_transform(test_outputs)
test_output_original = scaler.inverse_transform(test_output)

# 选择多个点进行预测和绘图
num_points = 10
selected_indices = np.random.choice(len(test_outputs), num_points, replace=False)

selected_outputs = test_output_original[selected_indices]
selected_outputs_pred = test_outputs[selected_indices]

# 绘制真实值

x = np.arange(num_points)
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, selected_outputs[:, 0], width, label='True')
ax.bar(x + width/2, selected_outputs_pred[:, 0], width, label='Predicted')
ax.set_xlabel('Data Point')
ax.set_ylabel('Value')
ax.set_title('True vs Predicted (Current)')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.bar(x - width/2, selected_outputs[:, 1], width, label='True')
ax.bar(x + width/2, selected_outputs_pred[:, 1], width, label='Predicted')
ax.set_xlabel('Data Point')
ax.set_ylabel('Value')
ax.set_title('True vs Predicted ($\Delta$T)')
ax.legend()
plt.show()

print(train_input)
