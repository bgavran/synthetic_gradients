import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.data_loader import MNIST

from src.modules import OneLayer, GradModule, SquaredDifferenceLoss

gpu = True

max_epochs = 10

input_size = 784
hidden_size = 32
out_size = 10

batch_size = 128
lr = 0.001

mnist = MNIST(batch_size)

squared_loss_sum = SquaredDifferenceLoss()
ce_loss = nn.CrossEntropyLoss()

module1 = OneLayer(input_size, hidden_size)
module1_opt = torch.optim.Adam(module1.parameters(), lr=lr)

module2 = OneLayer(hidden_size, out_size)
module2_opt = torch.optim.Adam(module2.parameters(), lr=lr)

grad_module1 = GradModule(hidden_size, module1.w.data.shape)
grad_module1_opt = torch.optim.Adam(grad_module1.parameters(), lr=lr)

if gpu:
    module1 = module1.cuda()
    module2 = module2.cuda()
    grad_module1 = grad_module1.cuda()


def network_output(images):
    return module2(module1(images))


def test_network():
    print("TESTING..........................................................")
    correct = 0
    total = 0
    for images, labels in mnist.test_loader:
        if gpu:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28 * 28))
        outputs = network_output(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)  # add amount in one batch
        correct += (predicted == labels).sum()

    print("Accuracy on 10000 test images:", 100 * correct / total)


def print_fn(step):
    return step % 100 == 0


for epoch in range(max_epochs):
    for step, (images, labels) in enumerate(mnist.train_loader):
        if gpu:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        module1_opt.zero_grad()
        module1_output = module1(images)
        mod1_generated_grad = grad_module1(module1_output)
        module1.set_grads(mod1_generated_grad)
        module1_opt.step()

        # Training last (second) layer
        module2_opt.zero_grad()
        module2_output = module2(module1(images))
        mod2_cost = ce_loss(module2_output, labels)
        mod2_cost.backward(retain_graph=True)
        module2_opt.step()

        # Training the gradient module
        grad_module1_opt.zero_grad()
        mod1_true_grad = torch.autograd.grad(mod2_cost, module1.w, retain_graph=True)[0]
        mod1_true_grad.volatile = False  # why do I have to set this?
        grad_mod1_cost = squared_loss_sum(mod1_true_grad, mod1_generated_grad)
        grad_mod1_cost.backward()
        grad_module1_opt.step()

        if print_fn(step):
            try:
                print("epoch", epoch, "   step", step)
                text = "mod2_cost {:.2f}, grad_mod1_cost {:.2f}, mod1_generated_grads norm {:.2f}"
                print(text.format(float(mod2_cost.data.cpu().numpy()),
                                  float(grad_mod1_cost.data.cpu().numpy()),
                                  float(mod1_generated_grad.norm().cpu().data.numpy())))
            except NameError:
                pass
            test_network()
