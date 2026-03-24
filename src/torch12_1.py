#####简单RNNCell的使用示例#####
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

#cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) 
#batch_first=True表示输入和输出的形状为(batch_size, seq_len, input_size)和(batch_size, hidden_size)，
# 而不是(seq_len, batch_size, input_size)和(hidden_size)。  
#inputs = torch.randn(batch_size, seq_len, input_size) #输入数据，形状为(batch_size, seq_len, input_size)，因为batch_first=True

inputs = torch.randn(seq_len, batch_size, input_size) #输入数据，形状为(seq_len, batch_size, input_size
hidden = torch.zeros(num_layers,batch_size, hidden_size) #初始隐藏状态，形状为(batch_size, hidden_size)

out, hidden = cell(inputs, hidden) #代替下边的循环，直接调用cell对象，输入为整个序列的输入数据和初始隐藏状态，输出为整个序列的输出和最终隐藏状态
# for idx, input in enumerate(dataset):
#     print('=' * 20, idx, '=' * 20) #打印分隔线和当前时间步的索引
#     print('Input size: ', input.shape)

#     hidden = cell(input, hidden) #更新隐藏状态，输入为当前时间步的输入和上一个时间步的隐藏状态

#     print('Hidden size: ', hidden.shape)
#     print(hidden)

print('Output size: ', out.shape)
print('Output:', out)
print('Hidden size: ', hidden.shape)
print('Hidden:', hidden)

