#! /usr/bin/python3
import torch
from mnist_init import download_mnist_dataset, display_dataset_info, save_img
from mnist_train import LeNet, load_mnist_data, train_model, save_model
from mnist_test import test_model

def main():
    print("1. 初始化mnist数据集")
    print("2. 训练模型")
    print("3. 测试LeNet模型")
    print("0. 退出")
    choice = input("请选择(1/2/3/0): ")

    if choice == "1":
        train_data, test_data = download_mnist_dataset()
        display_dataset_info(train_data, test_data)
        saveDirTrain = './DataImages-Train'
        saveDirTest = './DataImages-Test'
        save_img(train_data, saveDirTrain)
        save_img(test_data, saveDirTest)
    elif choice == "2":
        model = LeNet()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        train_loader, test_loader = load_mnist_data()
        train_model(model, train_loader, criterion, optimizer)
        save_model(model, "lenet_model.pth")
    elif choice == "3":
        model = LeNet()
        model.load_state_dict(torch.load("lenet_model.pth"))
        test_loader = load_mnist_data(test_batch_size=10000)[1]
        test_model(model, test_loader)
    elif choice == "0":
        return 0
    else:
        print("无效选项")

if __name__ == "__main__":
    main()

