from dataset import CustomDataset, get_transform, labels
from model import Net
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler


def test_model(PATH, dataset, batch_size, sampler):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler)
    net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %s val images: %d %%' % (len(sampler),
                                                                   100 * correct / total))


dataset = CustomDataset(get_transform(True), labels)

batch_size = 4
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

valid_sampler = SubsetRandomSampler(val_indices)

test_model("cifar_net.pth", dataset, batch_size, valid_sampler)
