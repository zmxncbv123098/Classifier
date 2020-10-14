from dataset import CustomDataset
from net import Net

import torch
import torch.nn as nn
import torch.optim as optim


dataset = CustomDataset('')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
inverted_labels = {1: 'bird', 2: 'cat', 3: 'dog', 4: 'horse', 5: 'sheep'}
""" Device """
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


net = Net()
net = net  # .to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data  # [0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

""" Save model """
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
