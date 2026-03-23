import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad() # 优化器使用前要梯度清零

        #forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics,每300个批次打印一次
        running_loss += loss.item() # loss.item()是一个数值，表示当前批次的损失值。running_loss是一个累积变量，用于统计当前300个批次的总损失。
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) #torch.max返回两个值，第一个是最大值，第二个是最大值的索引。这里我们只需要索引，所以用_来忽略第一个值。
            total += labels.size(0) #labels.size(0)是当前批次的样本数量。total是一个累积变量，用于统计测试集中所有样本的总数量。
            correct += (predicted == labels).sum().item() #predicted == labels会返回一个布尔张量，表示每个预测是否正确。sum()会计算出正确预测的数量，item()将其转换为一个数值。

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch == 9: # 训练10轮后测试模型性能
            test()