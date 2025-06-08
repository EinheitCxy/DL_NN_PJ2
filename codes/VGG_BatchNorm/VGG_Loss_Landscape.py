import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(3))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X, y in train_loader:
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    plt.imshow(np.transpose(X[0].numpy(), (1, 2, 0)))
    plt.title(f"Label: {y[0].item()}")
    plt.show()
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # 记录loss
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()
            # 记录梯度（以最后一层为例）
            loss.backward()
            grad.append(model.classifier[-1].weight.grad.clone().cpu().numpy())
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        model.eval()
        train_acc = get_accuracy(model, train_loader, device)
        val_acc = get_accuracy(model, val_loader, device)
        train_accuracy_curve[epoch] = train_acc
        val_accuracy_curve[epoch] = val_acc
        axes[1].plot(val_accuracy_curve, label='val_acc')
        axes[1].plot(train_accuracy_curve, label='train_acc')
        axes[1].legend()
        plt.show()

    return losses_list, grads


# Train your model
# feel free to modify
epo = 20
loss_save_path = ''
grad_save_path = ''

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
if len(losses_list) > 0:
    loss_array = np.array(losses_list, dtype=object)
    # 处理 ragged array
    min_curve = np.array([np.min(epoch_losses) for epoch_losses in loss_array])
    max_curve = np.array([np.max(epoch_losses) for epoch_losses in loss_array])

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(min_curve, max_curve):
    plt.figure(figsize=(10, 5))
    plt.plot(min_curve, label='min loss')
    plt.plot(max_curve, label='max loss')
    plt.fill_between(range(len(min_curve)), min_curve, max_curve, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Landscape')
    plt.legend()
    plt.show()