####加了嵌入层的RNN模型
import torch

num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
batch_size = 1
seq_len = 5
num_layers = 2

idx2char = ['e', 'h', 'l', 'o'] #索引到字符的映射
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

inputs = torch.LongTensor(x_data).view(1,-1)#将输入数据转换为二维张量，形状为(1, seq_len)，其中1表示批次大小，seq_len是序列长度。
labels = torch.LongTensor(y_data)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size) #嵌入层，将输入的索引转换为对应的嵌入向量，输入大小为input_size，输出大小为embedding_size
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                 hidden_size=hidden_size, 
                                 num_layers=num_layers,
                                 batch_first=True)
        #RNN层，输入大小为embedding_size，隐藏状态大小为hidden_size，层数为num_layers，
        # batch_first=True表示输入和输出的形状为(batch_size, seq_len, input_size)和(batch_size, hidden_size),
        # 而不是(seq_len, batch_size, input_size)和(hidden_size)
        self.fc = torch.nn.Linear(hidden_size, num_class) #全连接层，输入大小为hidden_size，输出大小为num_class
    

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size) #初始化隐藏状态，形状为(num_layers, batch_size, hidden_size)，
        #其中num_layers是RNN层的层数，x.size(0)是输入的批次大小，hidden_size是隐藏状态的特征数。
        x = self.emb(x) #将输入的索引转换为对应的嵌入向量，输入形状为(batch_size, seq_len)，输出形状为(batch_size, seq_len, embedding_size)
        x,_ = self.rnn(x, hidden) #RNN层的前向传播，输入形状为(batch_size, seq_len, embedding_size)，
        #输出形状为(batch_size, seq_len, hidden_size)
        x = self.fc(x) #全连接层，输入形状为(batch_size, seq_len, hidden_size)，输出形状为(batch_size, seq_len, num_class)
        return x.view(-1, num_class) #将输出张量调整为二维张量，形状为(batch_size*seq_len, num_class)，其中batch_size*seq_len是总的时间步数，num_class是类别数。

net = Model()

criterion = torch.nn.CrossEntropyLoss() #定义损失函数，使用交叉熵损失函数，适用于多分类问题。
optimizer = torch.optim.Adam(net.parameters(), lr=0.01) #定义优化器，使用Adam优化器，学习率为0.01。net.parameters()返回模型的可训练

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
