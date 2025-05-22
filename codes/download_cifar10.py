import torch
import torchvision
import torchvision.transforms as transforms
import os

def download_cifar10():
    # 设置数据保存路径
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载训练集
    print("正在下载CIFAR-10训练集...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True,
        download=True, 
        transform=transform
    )
    
    # 下载测试集
    print("正在下载CIFAR-10测试集...")
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False,
        download=True, 
        transform=transform
    )
    
    print("CIFAR-10数据集下载完成！")
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    print(f"数据保存在: {data_dir}")

if __name__ == "__main__":
    download_cifar10() 