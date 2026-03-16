import torch
####准备数据集
x_data = torch.Tensor([[1.0], [2.0], [3.0]]) #torch张量
y_data = torch.Tensor([[2.0], [4.0], [6.0]]) 
#(self, *args, **kwargs)中,*args是把输入变成一个词组，**kwargs是把输入变成词典
####设计模型（计算图）
class LinearModel(torch.nn.Module): #从父类nn.Module继承下来的Module会自行计算backward，但如果是自己构造的model，需要加入backward函数计算
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) #构造对象，包括权重w和偏置b。继承自Module，nn是neural network。1，1是输入为一维，输出为一维

    def forward(self, x): #前馈
        y_pred = self.linear(x)
        return y_pred
    
model = LinearModel() #实例化，可以被调用callable
####构造损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False) #损失函数，继承自nn.Module，不求mini-batch的均值。根据计算需求进行调整

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #优化器，来自class torch.optim.SGD。parameters自行将所有需要计算的权重找出。
                                                        #lr，学习率
####训练
for epoch in range(100):
    y_pred = model(x_data) #前馈
    loss = criterion(y_pred, y_data) #计算损失，也是前馈
    print(epoch, loss)

    optimizer.zero_grad() #梯度归零
    loss.backward() #反向传播
    optimizer.step() #权重更新

print('w = ',model.linear.weight.item())
print('b = ',model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ',y_test.data)