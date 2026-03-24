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

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels) #批归一化层，输入通道数为channels
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels) #批归一化层，输入通道数为channels
        #ResidualBlock要求输入和输出的通道数相同，如果输入和输出的通道数不同，则需要使用一个1x1卷积层来调整输入的通道数，使其与输出的通道数一致。
        #self.shortcut = torch.nn.Conv2d(channels, channels, kernel_size=1) if channels != channels else None
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x))) #卷积层1 + 批归一化 + ReLU激活函数
        y = self.bn2(self.conv2(y)) #卷积层2 + 批归一化 + ReLU激活函数
        return F.relu(x + y) 
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16) #第一个残差块，输入和输出的通道数都是16
        self.rblock2 = ResidualBlock(32) #第二个残差块，输入和输出的通道数都是32

        self.fc = torch.nn.Linear(512, 10) #全连接层，输入特征数为512=32*4*4，输出特征数为10

    def forward(self, x):
        in_size = x.size(0) #获取输入的批次大小
        x = self.mp(F.relu(self.conv1(x))) #卷积层1 + ReLU激活函数 + 最大池化
        x = self.rblock1(x) #第一个残差块
        x = self.mp(F.relu(self.conv2(x))) #卷积层2 + ReLU激活函数 + 最大池化
        x = self.rblock2(x) #第二个残差块
        x = x.view(in_size, -1) #展平操作，将特征图展平为一维向量，in_size是批次大小，-1表示自动计算
        x = self.fc(x) #全连接层
        return x
model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#检查是否有可用的GPU，如果有则使用GPU，否则使用CPU。
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train() #将模型设置为训练模式。这会启用诸如Dropout和BatchNorm等特定于训练的行为。
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) #将输入数据和标签移动到与模型相同的设备上，以确保计算在同一设备上进行。
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
    model.eval() #将模型设置为评估模式。这会禁用诸如Dropout和BatchNorm等特定于训练的行为，以确保在评估时模型的行为一致。
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            inputs, labels = images.to(device), labels.to(device) #将输入数据和标签移动到与模型相同的设备上，以确保计算在同一设备上进行。
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) #torch.max返回两个值，第一个是最大值，第二个是最大值的索引。这里我们只需要索引，所以用_来忽略第一个值。
            total += labels.size(0) #labels.size(0)是当前批次的样本数量。total是一个累积变量，用于统计测试集中所有样本的总数量。
            correct += (predicted == labels).sum().item() #predicted == labels会返回一个布尔张量，表示每个预测是否正确。sum()会计算出正确预测的数量，item()将其转换为一个数值。

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch == 9: # 训练10轮后测试模型性能
            test()