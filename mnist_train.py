#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

def load_mnist_data(train_batch_size=64, test_batch_size=1000):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 300 == 299:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 300))
                running_loss = 0.0
    print('训练完成')


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

if __name__ == "__main__":
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader, test_loader = load_mnist_data()
    train_model(model, train_loader, criterion, optimizer)
    save_model(model, "lenet_model.pth")

