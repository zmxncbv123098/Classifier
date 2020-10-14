from dataset import CustomDataset
from net import Net

import torch

dataset = CustomDataset('')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

PATH = './cifar_net.pth'
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

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
