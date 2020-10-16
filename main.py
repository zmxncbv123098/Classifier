from dataset import CustomDataset, get_transform
from model import Net

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import time


def validation(val_loader, epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 4385 val images and epoch %s: %d %%' % (epoch + 1, 100 * correct / total))
    return 100 * correct / total


dataset = CustomDataset('', get_transform(train=True))
dataset_val = CustomDataset('', get_transform(train=False))
inverted_labels = {1: 'bird', 2: 'cat', 3: 'dog', 4: 'horse', 5: 'sheep'}

batch_size = 4
epochs = 100
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

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                sampler=valid_sampler)

""" Device """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = Net()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

with mlflow.start_run():
    mlflow.log_param("Epochs", epochs)

    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        accuracy = validation(validation_loader, epoch)
        mlflow.log_metric(key="Loss", value=epoch_loss / len(train_loader), step=epoch)
        mlflow.log_metric(key="Accuracy of the network", value=accuracy, step=epoch)

        PATH = os.path.join("backup", "net_%s.pth" % str(epoch + 1))
        torch.save(net.state_dict(), PATH)
        time.sleep(180)

    print('Finished Training')

    """ Save model """
    PATH = os.path.join("net.pth")
    torch.save(net.state_dict(), PATH)
