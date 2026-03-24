import torch

input_size = 4
hidden_size = 4
batch_size = 1
num_layers = 2

idx2char = ['e', 'h', 'l', 'o'] #索引到字符的映射
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1,0,0,0], #e
                  [0,1,0,0], #h
                  [0,0,1,0], #l
                  [0,0,0,1]] #o 
x_one_hot = [one_hot_lookup[x] for x in x_data] #将输入数据转换为one-hot编码
inputs  = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)#将输入数据转换为三维张量，
#形状为(seq_len, batch_size, input_size)，其中seq_len是序列长度，batch_size是批次大小，input_size是输入特征数。
labels = torch.LongTensor(y_data)
#labels = torch.LongTensor(y_data).view(-1, 1) #将标签数据转换为二维张量，
#形状为(seq_len, 1)，其中seq_len是序列长度，1表示每个时间步的标签只有一个类别。
####简单RNNCell的使用示例#####
# class Model(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size):
#         super(Model, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size
#         self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

#     def forward(self, input, hidden):
#         hidden = self.rnncell(input, hidden) #输入为当前时间步的输入和上一个时间步的隐藏状态，输出为当前时间步的隐藏状态
#         return hidden
    
#     def init_hidden(self): #构造H0
#         return torch.zeros(self.batch_size, self.hidden_size) #初始化隐藏状态，形状为(batch_size, hidden_size)
    
# net = Model(input_size, hidden_size, batch_size)

# criterion = torch.nn.CrossEntropyLoss() #定义损失函数，使用交叉熵损失函数，适用于多分类问题。
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01) #定义优化器，使用Adam优化器，学习率为0.01。net.parameters()返回模型的可训练

#####训练
# for epoch in range(15):
#     loss = 0
#     optimizer.zero_grad() #优化器使用前要梯度清零
#     hidden = net.init_hidden() #初始化隐藏状态
#     print('Predicted string: ', end='')
#     for input, label in zip(inputs, labels): #inputs(seqlen, batch_size, input_size)，labels(seq_len, 1)，zip函数将它们按时间步进行配对
#         #input(batch_size, input_size)，label(1)
#         hidden = net(input, hidden) #前向传播，输入当前时间步的输入和隐藏状态，得到当前时间步的隐藏状态
#         loss += criterion(hidden, label) #计算损失，hidden是模型的输出，label是当前时间步的标签
#         _, idx = hidden.max(dim=1) #获取模型输出中最大值的索引，即预测的类别
#         print(idx2char[idx.item()], end='') #根据索引获取对应的字符，并打印出来
#     loss.backward() #反向传播，计算梯度
#     optimizer.step() #更新模型参数
#     print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item())) #打印当前轮次和损失值
######使用RNN层的示例#####
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size) #初始化隐藏状态，形状为(num_layers, batch_size, hidden_size)
        out,_ = self.rnncell(input, hidden) #输入为当前时间步的输入和上一个时间步的隐藏状态，输出为当前时间步的隐藏状态和最后一个时间步的隐藏状态
        return out.view(-1, self.hidden_size) #将输出张量调整为二维张量，形状为(seq_len, hidden_size)，其中seq_len是序列长度，hidden_size是隐藏状态的特征数。
    
    
net = Model(input_size, hidden_size, batch_size,num_layers)

criterion = torch.nn.CrossEntropyLoss() #定义损失函数，使用交叉熵损失函数，适用于多分类问题。
optimizer = torch.optim.Adam(net.parameters(), lr=0.01) #定义优化器，使用Adam优化器，学习率为0.01。net.parameters()返回模型的可训练

####RNN
for epoch in range(15):
    optimizer.zero_grad() #优化器使用前要梯度清零
    outputs = net(inputs)
    loss = criterion(outputs, labels) #计算损失，outputs是模型的输出，labels是标签
    loss.backward() #反向传播，计算梯度
    optimizer.step() #更新模型参数

    _,idx = outputs.max(dim=1) #获取模型输出中最大值的索引，即预测的类别
    idx = idx.data.numpy() #将索引转换为numpy数组
    print('Predicted string: ', ''.join([idx2char[x] for x in idx]), end='') #根据索引获取对应的字符，并打印出来
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item())) #打印当前轮次和损失值

