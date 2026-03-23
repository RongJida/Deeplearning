import torch
import numpy as np
from torch.utils.data import Dataset #抽象类不可实例化，只能继承
from torch.utils.data import DataLoader

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0] #xy是一个N行9列的矩阵，N是样本数量，9是feature（其中8列是特征列，1列是目标列），所以拿到的shape是[N,9]的元组
        #通过取第0个元素就知道数据集一共有多少个，所以对于下边len函数非常简单，直接return即可
        self.x_data = torch.from_numpy(xy[:,:-1]) #从第一列开始，最后一列不要。对于这个文件，取出来的是前边八列。
        self.y_data = torch.from_numpy(xy[:, [-1]]) #所有行，-1这一列，[]输出来是矩阵
        
    def __getitem__(self, index): #加上索引
        return self.x_data[index],  self.y_data[index]

    def __len__(self): #检查len长度
        return self.len
if __name__ == '__main__':     
    dataset = DiabetesDataset('/Users/nuomicir/Documents/USTC_R/文献/ai/深度学习/PyTorch深度学习实践/diabetes.csv.gz')
    train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

    class Model(torch.nn.Module):
        def __init__(self):
         super(Model, self).__init__()
         self.linear1 = torch.nn.Linear(8, 6)
         self.linear2 = torch.nn.Linear(6, 4)
         self.linear3 = torch.nn.Linear(4, 1)
         self.sigmoid = torch.nn.Sigmoid() #没有需要调整的参数，所以括号里空。用来构建计算图。

        def forward(self, x):
         x = self.sigmoid(self.linear1(x))
         x = self.sigmoid(self.linear2(x))
         x = self.sigmoid(self.linear3(x))
         return x
    
    model = Model()
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        for i,data in enumerate(train_loader, 0): #直接对train_loader做迭代，enumerate是查看当前是第几次迭代
        #prepare data
            input, labels = data #inputs是x，labels是y。都是张量。
        #Forward
            y_pred = model(input)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
        #Backward
            optimizer.zero_grad()
            loss.backward()
        #update
            optimizer.step()
            if i % 10 == 0: # 没必要全打出来，每10个批次打印一次就好
                print(f"Epoch {epoch}, Batch {i}, Loss {loss.item():.4f}")

    print('w = ',model.linear3.weight.data)
    print('b = ',model.linear3.bias.data)

