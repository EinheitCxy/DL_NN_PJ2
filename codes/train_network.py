import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    data_path: str = 'codes/data'
    checkpoint_dir: str = 'checkpoints'
    num_workers: int = 2

def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    """获取可用的设备"""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS (Metal Performance Shaders)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device

def get_transforms():
    """获取数据预处理转换"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform_train, transform_test

def load_data(config: TrainingConfig):
    """加载数据集"""
    transform_train, transform_test = get_transforms()
    
    trainset = torchvision.datasets.CIFAR10(
        root=config.data_path, 
        train=True, 
        download=False,
        transform=transform_train)
    trainloader = DataLoader(
        trainset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers)
    
    testset = torchvision.datasets.CIFAR10(
        root=config.data_path, 
        train=False, 
        download=False,
        transform=transform_test)
    testloader = DataLoader(
        testset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers)
    
    return trainloader, testloader

def generate_alexnet():
    """生成AlexNet模型"""
    return nn.Sequential(
        # 第一层卷积：32x32 -> 16x16
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # 第二层卷积：16x16 -> 8x8
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # 第三层卷积：8x8 -> 4x4
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # 第四层卷积：4x4 -> 4x4
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        
        # 第五层卷积：4x4 -> 2x2
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Flatten(),
        # 2x2x512 = 2048
        nn.Linear(2048, 1024), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 512), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 10)
    )

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/len(val_loader), 100.*correct/total

def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def save_checkpoint(model, optimizer, epoch, val_acc, config: TrainingConfig):
    """保存检查点"""
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, f'{config.checkpoint_dir}/best_model.pth')

def train(config: TrainingConfig):
    """主训练函数"""
    # 设置设备
    device = get_device()
    
    # 设置随机种子
    set_seed()
    
    # 加载数据
    trainloader, testloader = load_data(config)
    
    # 创建模型
    model = generate_alexnet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    
    # 训练记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_acc = 0
    
    # 训练循环
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, testloader, criterion, device)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f'Learning rate changed from {old_lr:.6f} to {new_lr:.6f}')
        
        # 记录结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, config)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)

def main():
    """主函数"""
    config = TrainingConfig()
    train(config)

if __name__ == '__main__':
    main() 