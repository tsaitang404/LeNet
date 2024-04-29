#! /usr/bin/python3
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = nn.functional.avg_pool2d(x, 2)
        x = torch.tanh(self.conv2(x))
        x = nn.functional.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def test_model(model, test_loader):
    # 使用训练好的模型对实测数据进行预测
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # 显示图片和预测结果
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    for idx in range(5):
        ax = axes[idx]
        img = images[idx][0] * 0.5 + 0.5  # 反归一化
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Predicted: {predicted[idx].item()}, Actual: {labels[idx].item()}')
        ax.axis('off')

    plt.show()

if __name__ == "__main__":
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    # 初始化模型
    model = LeNet()

    # 加载模型参数
    model.load_state_dict(torch.load("lenet_model.pth"))

    # 测试模型
    test_model(model, test_loader)

