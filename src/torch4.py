import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0]) #[]套一层，代表一维张量；[[]]套两层，代表二维张量
w.requires_grad = True #创建的Tensor默认不会计算其梯度

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2
print('Predict (before training)', 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y) #第一步，先算损失
        l.backward() #第二步，反向传播
        print('\tgrad', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data #张量的计算会有计算图出现，内存暴涨。我们只需要标量计算，所以要.data
                                            #第三步，梯度下降，序的更新
        w.grad.data.zero_() #权重里梯度的数据全部清零，因为本次算法结构里不想让梯度累加。但有些算法，根据设计需求需要累加。
    print("progress:", epoch, l.item())
print("predict (after training)", 4, forward(4).item())