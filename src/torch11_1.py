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

class InceptionA(torch.nn.Module):
    def __init__(self,in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)
        
        self.branch3x3dbl_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5_1 = self.branch5x5_1(x)
        branch5x5_2 = self.branch5x5_2(branch5x5_1)

        branch3x3dbl_1 = self.branch3x3dbl_1(x)
        branch3x3dbl_2 = self.branch3x3dbl_2(branch3x3dbl_1)
        branch3x3dbl_3 = self.branch3x3dbl_3(branch3x3dbl_2)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5_2, branch3x3dbl_3, branch_pool]
        return torch.cat(outputs, 1)
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5) #输入通道数为1，输出通道数为10，卷积核大小为5x5
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5) #输入通道数为88，输出通道数为20，卷积核大小为5x5

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2) #池化层，使用2x2的窗口进行最大池化
        self.fc = torch.nn.Linear(1408, 10) #全连接层，输入特征数为1408，输出特征数为10

    def forward(self, x):
        in_size = x.size(0) #获取输入的批次大小
        x = F.relu(self.mp(self.conv1(x))) #卷积层1 + ReLU激活函数 + 最大池化
        x = self.incep1(x) #Inception模块1
        x = F.relu(self.mp(self.conv2(x))) #卷积层2 + ReLU激活函数 + 最大池化
        x = self.incep2(x) #Inception模块2
        ####
        #print(f'Inception模块输出形状: {x.shape}') #打印Inception模块的输出形状，记得将上个部分的self.fc= torch.nn.Linear(1408, 10)注释掉，
        # 还有这个部分下边的x = self.fc(x)
        # 否则会报错，因为全连接层的输入特征数不匹配。算出来是1408=88*4*4，88是Inception模块的输出通道数，4x4是特征图的宽度和高度。
        ####
        x = x.view(in_size, -1) #将特征图展平为一维向量
        x = self.fc(x) #全连接层
        return x    
    
model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#检查是否有可用的GPU，如果有则使用GPU，否则使用CPU。
model.to(device)
#######
# 测试代码片段，匹配{x.shape}
# if __name__ == '__main__':

#     # 模拟一个 MNIST 批次：1张图片，1个通道，28x28
#     test_input = torch.randn(1, 1, 28, 28) 
#     try:
#         output = model(test_input)
#         print(f"最终输出形状: {output.shape}")
#     except Exception as e:
#         print(f"哎呀，报错了: {e}")

########
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
