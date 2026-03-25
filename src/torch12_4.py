####RNN分类器示例：根据名字预测国家(练习版)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import gzip 



###1、常量与配置
HIDDEN_SIZE= 100
BATCH_SIZE= 256
N_LAYER= 2
N_EPOCHS= 100
N_CHARS= 128
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_GPU else "cpu")
###2、数据集准备



class NameDataset(Dataset):
    def _init__(self, is_train_set=True):
        filename = '/Users/nuomicir/Documents/USTC_R/文献/ai/深度学习/PyTorch深度学习实践/names_train.csv.gz' if is_train_set else '/Users/nuomicir/Documents/USTC_R/文献/ai/深度学习/PyTorch深度学习实践/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows] #提取名字数据，假设名字在第一列
        self.len = len(self.names) #数据集的长度
        self.countries = [row[1] for row in rows] #提取国家数据，假设国家在第二列
        self.country_list = sorted(list(set(self.countries))) #获取唯一的国家列表，并进行排序
        self.country_dict = self.getCountryDict() #获取国家到索引的映射字典
        self.country_num = len(self.country_list) #国家的数量

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]] #返回名字和对应的国家索引
    
    def __len__(self):
        return self.len #返回数据集的长度
    
    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0): #枚举国家列表，获取每个国家的索引和名称
            country_dict[country_name] = idx #将国家名称和索引添加到字典中
        return country_dict
    
    def idx2country(self, index):
        return self.country_list[index] #根据索引获取对应的国家名称
    
    def getCountriesNum(self):
        return self.country_num #返回国家的数量

def time_since(since):
    s = time.time() - since
    m = match.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0") #如果使用GPU，指定设备为cuda:0
        tensor = tensor.to(device) #将张量移动到GPU上进行计算
        return tensor #如果不使用GPU，直接返回原始张量

def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1): #枚举训练数据加载器，获取每个批次的名字和国家数据
        inputs, seq_lengths, target = make_tensors(names, countries) #将名字和国家数据转换为张量，inputs是名字的索引张量，seq_lengths是名字的长度张量，target是国家的索引张量
        outputs = classifier(inputs, seq_lengths) #将输入的名字索引张量和名字长度张量传入分类器模型，得到输出张量，形状为(batch_size, output_size)
        loss = criterion(outputs, target) #计算损失，outputs是模型的输出，target是国家的索引张量
        optimizer.zero_grad() #优化器使用前要梯度清零
        loss.backward() #反向传播，计算梯度
        optimizer.step() #更新模型参数

        total_loss += loss.item() #累积损失值
        if i % 10 == 0: #每10个批次打印一次损失值和训练进度
            print(f'[{time_since(start)}] Epoch {epoch}', end='') #打印当前时间和轮次
            print(f'[{i * len(inputs)}/{len(trainset)}]', end='') #打印当前训练进度，i * len(inputs)是当前已经处理的样本数量，len(trainset)是总的样本数量
            print(f'loss:={total_loss / (i * len(inputs))}')   #打印平均
        return total_loss
    
def teatModel():
    correct = 0
    total = len(testset) #获取测试集的总样本数量
    print("evaluating trained model...")
    with torch.no_grad(): #在评估模型时，不需要计算梯度，因此使用torch.no_grad()上下文管理器来禁用梯度计算，以节省内存和计算资源。
        for i, (names, countries) in enumerate(testloader, 1): #枚举测试数据加载器，获取每个批次的名字和国家数据
            inputs, seq_lengths, target = make_tensors(names, countries) #将名字和国家数据转换为张量，inputs是名字的索引张量，seq_lengths是名字的长度张量，target是国家的索引张量
            output = classifier(inputs, seq_lengths) #将输入的名字索引张量和名字长度张量传入分类器模型，得到输出张量，形状为(batch_size, output_size)
            pred = output.max(dim=1,keepdim=True)[1] #获取模型输出中最大值的索引，即预测的类别，pred的形状为(batch_size, 1)
            correct += pred.eq(target.view_as(pred)).sum().item() #计算正确预测的数量，pred.eq(target.view_as(pred))会返回一个布尔张量，表示每个预测是否正确。sum()会计算出正确预测的数量，item()将其转换为一个数值。

        percent = '%.2f' % (100 * correct / total) #计算准确率，correct是正确预测的数量，total是总的样本数量
        print(f'Test set: Accuracy: {correct}/{total} ({percent}%)') #打印测试集的准确率

    return correct / total #返回准确率


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_sizen, n_layers=1, bidirectional= True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1 #双向RNN的方向数为2，单向RNN的方向数为1

        self.embedding = torch.nn.Embedding(input_size, hidden_size) #嵌入层，将输入的索引转换为对应的嵌入向量，输入大小为input_size，输出大小为hidden_size
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional) #GRU层，输入大小为hidden_size，隐藏状态大小为hidden_size，层数为n_layers，是否双向由bidirectional参数决定  

        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size) #全连接层，输入大小为hidden_size * n_directions，输出大小为output_size

    def __init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, 
                             batch_size, self.hidden_size) #初始化隐藏状态，形状为(n_layers * n_directions, batch_size, 
        #hidden_size)，其中n_layers是RNN层的层数，n_directions是RNN的方向数
        return creat_tensor(hidden) #将隐藏状态张量移动到GPU上进行计算
    
    def forward(self, input, seq_lengths):
        input = input.t() #转置输入张量，形状从(batch_size, seq_len)变为(seq_len, batch_size)，因为RNN层要求输入的形状为(seq_len, batch_size, input_size)
        batch_size = input.size(1) #获取批次大小，即输入张量的第二维大小
        hidden = self.__init_hidden(batch_size) #初始化隐藏状态，形状为(n_layers * n_directions, batch_size, hidden_size)   
        embedding = self.embedding(input) #将输入的索引转换为对应的嵌入向量，输入形状为(seq_len, batch_size)，输出形状为(seq_len, batch_size, hidden_size)

        #pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths) #将嵌入向量序列打包成一个PackedSequence对象，seq_lengths是每个序列的长度，用于告诉RNN层每个序列的实际长度，以便正确处理填充部分
        output, hidden = self.gru(gru_input, hidden) #GRU层的前向传播，输入为打包后的嵌入向量序列和初始隐藏状态，输出为打包后的输出序列和最终隐藏状态
        if self.n_directions == 2: #如果是双向RNN
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1) #将正向和反向的最后一个隐藏状态拼接在一起，形状为(batch_size, hidden_size * 2)
        else:
            hidden_cat = hidden[-1] #如果是单向RNN，直接使用最后一个隐藏状态，形状为(batch_size, hidden_size)
        fc_output = self.fc(hidden_cat) #全连接层的前向传播，输入形状为(batch_size, hidden_size * n_directions)，输出形状为(batch_size, output_size)
        return fc_output #返回全连接层的输出，形状为(batch_size, output_size)
    
    def make_tensors(name, countriez):
        sequences_and_lengths = [name2list(name) for name in names] #将名字转换为索引列表，并记录每个名字的长度
        name_sequences = [sl[0] for sl in sequences_and_lengths] #提取名字的索引列表
        seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths]) #提取名字的长度，并转换为LongTensor
        countries = countries.long() #将国家标签转换为LongTensor

        #make tensor of name, Batch x SeqLen
        seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True) #对名字的长度进行排序，得到排序后的长度和对应的索引
        seq_tensor = seq_tensor[perm_idx] #根据排序后的索引对名字的索引列表进行重新排序，得到排序后的名字索引张量
        countries = countries[perm_idx] #根据排序后的索引对国家标签进行重新排序，得到排序后的国家标签张量

        return creat_tensor(seq_tensor), \
               creat_tensor(countries),  \
               creat_tensor(seq_lengths) #将名字索引张量和国家标签张量移动到GPU上进行计算，并返回它们以及名字的长度




if __name__ == '__main__':
    trainset = NameDataset(is_train_set=True) #创建训练数据集实例
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True) #创建训练数据加载器，batch_size是批次大小，shuffle=True表示在每个epoch开始时打乱数据
    testset = NameDataset(is_train_set=False) #创建测试数据集实例
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False) #创建测试数据加载器，batch_size是批次大小，shuffle=False表示不打乱数据

    N_COUNTRY = trainset.getCountriesNum() #获取国家的数量，作为类别数, 决定了模型输出的维度，最终的大小
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASSES, N_LAYERS)
    #创建RNN分类器实例，N_CHARS是输入特征数，HIDDEN_SIZE是隐藏层大小，N_CLASSES是类别数，N_LAYERS是RNN层数
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device) #将模型移动到GPU上进行计算
    
    criterion = torch.nn.CrossEntropyLoss() #定义损失函数，使用交叉熵损失函数，适用于多分类问题。
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001) 

    start = time.time()
    print('Training for %d epochs...' % N_EPOCHS)
    acc_list = []#用于记录每轮训练后的准确率
    for epoch in range(1, N_EPOCH + 1):
        #Train cycle
        trainModel() #函数封装
        acc = testModel() #函数封装，测试模型性能，返回准确率
        acc_list.append(acc) #将当前轮次的准确率添加到列表中

    epoch = np.arange(1, len(acc_list) + 1, 1) #生成一个从1到len(acc_list)的整数数组，表示每轮训练的轮次
    acc_list = np.array(acc_list) #将准确率列表转换为numpy数组，方便后续绘图
    plt.plot(epoch, acc_list) #绘制轮次与准确率的关系图
    plt.xlabel('Epoch') #设置x轴标签
    plt.ylabel('Accuracy') #设置y轴标签
    plt.grid() #显示网格线
    plt.show() #显示图形