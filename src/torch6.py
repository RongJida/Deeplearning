import torchvision
#train_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True, download=True) #MNIST数据集
#root:数据集在的路径。train:是训练集还是测试集。download：是否需要联网下载。
#train_set = torchvision.datasets.CIFAR10(...) #CIFAR10数据集
import torch
import torch.nn.functional as F
###数据准备
x_data = torch.Tensor([[1.0], [2.0], [3.0]]) #torch张量
y_data = torch.Tensor([[0], [0], [1]]) 
###设计模型
class LogisticRegressModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressModel,self).__init__()
        self.linear = torch.nn.Linear(1, 1) #1个输入特征，1个输出特征
    
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x)) #logistic和线性回归的差别在此!!!将线性输出映射到[0,1]概率
        return y_pred
model = LogisticRegressModel()
###构造损失函数和优化器   
criterion = torch.nn.BCELoss(size_average=False) #进行二分类交叉熵损失BCEloss计算，是否需要求均值。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
###训练循环
for epoch in range(1000):
    y_pred = model(x_data) #前馈
    loss = criterion(y_pred, y_data) #计算损失，也是前馈
    print(epoch, loss.item())

    optimizer.zero_grad() #梯度归零
    loss.backward() #反向传播
    optimizer.step() #权重更新
print('w = ',model.linear.weight.item())
print('b = ',model.linear.bias.item())

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200) #从0-10之内，生成均匀分布的200个点
x_t = torch.Tensor(x).view(200, 1) #200行一列的矩阵
y_t = model(x_t) #预测概率
y = y_t.data.numpy() #numpy的n维数组

plt.plot(x, y)
plt.plot([0, 10],[0.5, 0.5], c ='r')
plt.xlabel('Hours')
plt.ylabel('Probaility pf Pass')
plt.grid()
plt.show()