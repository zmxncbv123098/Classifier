import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
import mlflow

from dataset import CustomDataset, get_transform, labels
from model import Net


def test_dataset(loader, epoch, net, batch_size, loader_name):
    was_training = net.training
    net.eval()
    correct = 0
    total = 0
    data_loss = 0.0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            data_loss += loss.item()
    print('Accuracy of the network on the %s %s images and epoch %s: %d %%' %
          (len(loader) * batch_size, loader_name, epoch + 1, 100 * correct / total))

    net.train(mode=was_training)
    return (100 * correct / total), (data_loss / len(loader))


def train(loader, epoch, net):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


""" Load Annotations """

folder = 'annotations/'
filename = 'instances_train2017.json'
ann_filepath = os.path.join(folder, filename)

""" PARAMS """
batch_size = 4
epochs = 100
learning_rate = 0.001
momentum = 0.9
redundancy = 0

dataset = CustomDataset(get_transform(train=True), labels, ann_filepath, redundancy)
dataset_val = CustomDataset(get_transform(train=False), labels, ann_filepath, redundancy)

"""  Split Dataset  """
validation_split = .2
shuffle_dataset = True
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices[:len(train_indices)-(len(train_indices) % 4)])
valid_sampler = SubsetRandomSampler(val_indices[:len(val_indices)-(len(val_indices) % 4)])

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=4)
validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=4)

"""  Device  """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# net = Net()
# net = net.to(device)

net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 5)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

with mlflow.start_run():
    mlflow.log_param("Batch size", batch_size)
    mlflow.log_param("Epochs", epochs)
    mlflow.log_param("Learning Rate", learning_rate)
    mlflow.log_param("Momentum", momentum)
    mlflow.log_param("Dataset Size", dataset_size)
    mlflow.log_param("Adjustment", redundancy)

    for epoch in range(epochs):  # loop over the dataset multiple times
        best_model_wts = copy.deepcopy(net.state_dict())
        best_acc = 0.0
        train(train_loader, epoch, net)

        # Check-in metric
        train_accuracy, train_loss = test_dataset(train_loader, epoch, net, batch_size, "train")
        val_accuracy, val_loss = test_dataset(validation_loader, epoch, net, batch_size, "val")
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(net.state_dict())
        mlflow.log_metric(key="Validation Loss", value=val_loss, step=epoch)
        mlflow.log_metric(key="Train Loss", value=train_loss, step=epoch)
        mlflow.log_metric(key="Validation Accuracy", value=val_accuracy, step=epoch)
        mlflow.log_metric(key="Train Accuracy", value=train_accuracy, step=epoch)

        if epoch > 0 and epoch % 5 == 0:
            PATH = os.path.join("backup", "net_%s.pth" % str(epoch + 1))
            # torch.save(net.state_dict(), PATH)
            time.sleep(120)

    print('Finished Training')

    """ Save model """
    PATH = os.path.join("net.pth")
    # torch.save(net.state_dict(), PATH)
