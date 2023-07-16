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

# # 数据归一化
# scaler = MinMaxScaler()
# input_data = scaler.fit_transform(input_data)
# output_data = scaler.fit_transform(output_data)
# 归一化数据
input_data = (input_data - np.min(input_data, axis=0)) / (np.max(input_data, axis=0) - np.min(input_data, axis=0))
output_data = (output_data - np.min(output_data, axis=0)) / (np.max(output_data, axis=0) - np.min(output_data, axis=0))

# train_size = int(0.8 * len(data))
# train_input = input_data[:train_size]
# train_output = output_data[:train_size]
# test_input = input_data[train_size:]
# test_output = output_data[train_size:]
train_input, test_input, train_output, test_output = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
print(train_input)