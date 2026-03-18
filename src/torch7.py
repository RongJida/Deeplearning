import numpy as np
import torch
xy = np.loadtxt('/Users/nuomicir/Documents/USTC_R/文献/ai/深度学习/PyTorch深度学习实践/diabetes.csv.gz',delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1]) #从第一列开始，最后一列不要。对于这个文件，取出来的是前边八列。
y_data = torch.from_numpy(xy[:, [-1]]) #所有行，-1这一列，[]输出来是矩阵

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

#ReLU
class Model_ReLU(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.activate = torch.nn.ReLU() #没有需要调整的参数，所以括号里空。用来构建计算图。

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x)) #如果最后一步计算y_pred，需要改成sigmoid。因为ReLU对小于0的部分都输出为o
        return x
model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    #Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    #Backward
    optimizer.zero_grad()
    loss.backward()
    #Update
    optimizer.step()

print('w = ',model.linear3.weight.data)
print('b = ',model.linear3.bias.data)
#注意这块的print和前边的代码都不同。因为本次代码输入的数据是8维的
