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

###CNN卷积层
# in_channels, out_channels= 5, 10
# width, height = 100, 100
# kernel_size = 3
# batch_size = 1

# input = torch.randn(batch_size, in_channels, width, height)
# conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
# output = conv_layer(input)
# print(f'输入张量形状: {input.shape}')
# print(f'输出张量形状: {output.shape}')
# print(f'卷积层权重形状: {conv_layer.weight.shape}')

###padding=1
# input = [3,4,6,5,7,
#          2,4,6,8,2,
#          1,6,7,8,4,
#          9,7,4,6,2,
#          3,7,5,4,1]
# input = torch.Tensor(input).view(1,1,5,5) #输入张量形状: torch.Size([1, 1, 5, 5])，表示批次大小为1，输入通道数为1，宽度和高度均为5。
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False) #输出张量形状: torch.Size([1, 1, 3, 3])，表示批次大小为1，输出通道数为1，宽度和高度均为3。
# kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3) #卷积层权重形状: torch.Size([1, 1, 3, 3])，表示输出通道数为1，输入通道数为1，卷积核的宽度和高度均为3。
# conv_layer.weight.data = kernel.data
# output = conv_layer(input)
# print(f'输入张量形状: {input.shape}')
# print(f'输出张量形状: {output.shape}')

###stride=2
# input = [3,4,6,5,7,
#          2,4,6,8,2,
#          1,6,7,8,4,
#          9,7,4,6,2,
#          3,7,5,4,1]
# input = torch.Tensor(input).view(1,1,5,5) #输入张量形状: torch.Size([1, 1, 5, 5])，表示批次大小为1，输入通道数为1，宽度和高度均为5。
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False) #输出张量形状: torch.Size([1, 1, 2, 2])，表示批次大小为1，输出通道数为1，宽度和高度均为2。
# kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3) #卷积层权重形状: torch.Size([1, 1, 3, 3])，表示输出通道数为1，输入通道数为1，卷积核的宽度和高度均为3。
# conv_layer.weight.data = kernel.data
# output = conv_layer(input)
# print(f'输入张量形状: {input.shape}')
# print(f'输出张量形状: {output.shape}')

###Max Pooling Layer
# input = [3,4,6,5,
#          2,4,6,8,
#          1,6,7,8,
#          9,7,4,6,]
# input = torch.Tensor(input).view(1,1,4,4) #输入张量形状: torch.Size([1, 1, 4, 4])，表示批次大小为1，输入通道数为1，宽度和高度均为4。
# maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2) #输出张量形状: torch.Size([1, 1, 2, 2])，表示批次大小为1，输出通道数为1，宽度和高度均为2。
# output = maxpooling_layer(input)
# print(f'输入张量形状: {input.shape}')
# print(f'输出张量形状: {output.shape}')

###设计计算图
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.fc = torch.nn.Linear(320,10)

    def forward(self, x):
        batch_size = x.size(0) #获取输入张量的批次大小，x.size(0)返回输入张量在第0维的大小，即批次大小。
        x = F.relu(self.pooling(self.conv1(x))) #第一层卷积后进行池化和ReLU激活函数处理。首先，输入张量通过self.conv1进行卷积操作，然后通过self.pooling进行池化操作，最后通过F.relu函数应用ReLU激活函数。
        x = F.relu(self.pooling(self.conv2(x))) #第二层卷积后进行池化和ReLU激活函数处理。首先，输入张量通过self.conv2进行卷积操作，然后通过self.pooling进行池化操作，最后通过F.relu函数应用ReLU激活函数。
        x = x.view(batch_size, -1) #将卷积层的输出张量展平为二维张量，以便输入到全连接层中。x.view(batch_size, -1)将输入张量x的形状调整为(batch_size, -1)，其中-1表示自动计算该维度的大小，使得总元素数量保持不变。
        x = self.fc(x)
        return x
model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#检查是否有可用的GPU，如果有则使用GPU，否则使用CPU。
model.to(device) #将模型移动到指定的设备上（GPU或CPU）。model.to(device)将模型的参数和缓冲区移动到指定的设备上，以便在该设备上进行计算。

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