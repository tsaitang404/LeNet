#! /usr/bin/python3
# mnist_dataset.py

from torchvision import datasets
import os
from tqdm import tqdm

def download_mnist_dataset():
    # 使用 torchvision 中的 datasets 下载数据集。
    train_data = datasets.MNIST(root="./data/", train=True, download=True)
    test_data = datasets.MNIST(root="./data/", train=False, download=True)
    return train_data, test_data

def display_dataset_info(train_data, test_data):
    # 显示数据集内容
    print(f"len(train_data): {len(train_data)}, len(test_data): {len(test_data)}")
    print(train_data[0])
    print(train_data[0][0])

def save_img(data, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in tqdm(range(len(data))):
        img, label = data[i]
        img.save(os.path.join(save_path, str(i) + '-label-' + str(label) + '.png'))

if __name__ == "__main__":
    train_data, test_data = download_mnist_dataset()
    display_dataset_info(train_data, test_data)
    saveDirTrain = './DataImages-Train'
    saveDirTest = './DataImages-Test'
    save_img(train_data, saveDirTrain)
    save_img(test_data, saveDirTest)

