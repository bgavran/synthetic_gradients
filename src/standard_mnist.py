import torch
import torch.nn as nn
from torch.autograd import Variable
from src.data_loader import MNIST

gpu = True

input_size = 784
hidden_size = 500
num_classes = 10
batch_size = 100
lr = 0.001

mnist = MNIST(batch_size)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
if gpu:
    net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # this optimizer only updates net's parameters

num_epochs = 2
for epoch in range(num_epochs):
    for step, (images, labels) in enumerate(mnist.train_loader):
        if gpu:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()  # backward accumulates gradients in all the leaves
        optimizer.step()  # updates only net's parameters

        if step % 100 == 0:
            print("Epoch ", epoch + 1, " Step ", step, " Loss ", loss.data[0])


def test_network(network):
    correct = 0
    total = 0
    for images, labels in mnist.test_loader:
        if gpu:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28 * 28))
        outputs = network(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)  # add amount in one batch
        correct += (predicted == labels).sum()

    print("Accuracy on 10000 test images:", 100 * correct / total)


test_network(net)
