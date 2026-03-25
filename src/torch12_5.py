#### 12.5 RNN 进阶：变长序列、双向 GRU、名字分类器
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import gzip
import csv

# --- 1. 常量与配置 ---
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYERS = 2
N_EPOCHS = 100
N_CHARS = 128  # ASCII 字符数量
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_GPU else "cpu")

# --- 2. 数据集准备 ---
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        # 请确保路径正确，建议放在代码同级目录
        filename = '/Users/nuomicir/Documents/USTC_R/文献/ai/深度学习/PyTorch深度学习实践/names_train.csv.gz' if is_train_set else '/Users/nuomicir/Documents/USTC_R/文献/ai/深度学习/PyTorch深度学习实践/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = sorted(list(set(self.countries)))
        self.country_dict = {country: i for i, country in enumerate(self.country_list)}
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]
    
    def __len__(self):
        return self.len

    def getCountriesNum(self):
        return self.country_num

# --- 3. 工具函数 ---
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def name2list(name):
    """将名字转为 ASCII 码列表"""
    return [ord(c) for c in name], len(name)

def make_tensors(names, countries):
    """关键：将名字列表转为 Tensor 并进行填充"""
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()

    # 制作填充后的 Tensor (Batch x Max_Seq_Len)
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # 排序：按照长度降序排列（pack_padded_sequence 的要求）
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return seq_tensor.to(device), seq_lengths.to(device), countries.to(device)

# --- 4. 模型定义 ---
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return hidden.to(device)
    
    def forward(self, input, seq_lengths):
        # input shape: (B, S) -> 转置为 (S, B) 适配 GRU
        input = input.t()
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)
        
        embedded = self.embedding(input) # (S, B, H)

        # 打包变长序列
        gru_input = pack_padded_sequence(embedded, seq_lengths.cpu())
        output, hidden = self.gru(gru_input, hidden)

        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
            
        return self.fc(hidden_cat)

# --- 5. 训练与测试逻辑 ---
def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}] loss: {total_loss / (i * len(inputs)):.6f}')
    return total_loss

def testModel():
    correct = 0
    total = len(testset)
    print("Evaluating trained model...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy: {correct}/{total} ({percent}%)')
    return correct / total

# --- 6. 主程序 ---
if __name__ == '__main__':
    # 数据加载
    trainset = NameDataset(is_train_set=True)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = NameDataset(is_train_set=False)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    N_COUNTRY = trainset.getCountriesNum()
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYERS).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    acc_list = []
    
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    # 绘图
    plt.plot(range(1, N_EPOCHS + 1), acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()