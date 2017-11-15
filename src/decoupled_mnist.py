import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.data_loader import MNIST

from src.modules import OneLayer, GradModule, SquaredDifferenceLoss

gpu = True

max_epochs = 4

input_size = 784
hidden_size_1 = 256
hidden_size_2 = 256
out_size = 10

batch_size = 128
lr = 0.001

mnist = MNIST(batch_size)

squared_loss_sum = SquaredDifferenceLoss()
ce_loss = nn.CrossEntropyLoss()

module1 = OneLayer(input_size, hidden_size_1)
module1_opt = torch.optim.Adam(module1.parameters(), lr=lr)

module2 = OneLayer(hidden_size_1, hidden_size_2)
module2_opt = torch.optim.Adam(module2.parameters(), lr=lr)

module3 = OneLayer(hidden_size_2, out_size)
module3_opt = torch.optim.Adam(module3.parameters(), lr=lr)

grad_module1 = GradModule(hidden_size_1, module1.w.data.shape)
grad_module1_opt = torch.optim.Adam(grad_module1.parameters(), lr=lr)

grad_module2 = GradModule(hidden_size_2, module2.w.data.shape)
grad_module2_opt = torch.optim.Adam(grad_module2.parameters(), lr=lr)

if gpu:
    module1 = module1.cuda()
    module2 = module2.cuda()
    module3 = module3.cuda()
    grad_module1 = grad_module1.cuda()
    grad_module2 = grad_module2.cuda()


def network_output(images):
    return module3(module2(module1(images)))


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

    print("Accuracy on 10000 test images:", 100 * correct / total, "\n\n")


def print_fn(step):
    return step % 100 == 0


for epoch in range(max_epochs):
    for step, (images, labels) in enumerate(mnist.train_loader):
        if gpu:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Training first layer with synthetic gradients
        module1_opt.zero_grad()
        module1_output = module1(images)
        mod1_generated_grad = grad_module1(module1_output)
        module1.set_grads(mod1_generated_grad)
        module1_opt.step()

        # Training second layer with synthetic gradients
        module2_opt.zero_grad()
        module2_output = module2(module1_output)
        mod2_generated_grad = grad_module2(module2_output)
        module2.set_grads(mod2_generated_grad)
        module2_opt.step()

        # Training third (last) layer with real gradients
        module3_opt.zero_grad()
        module3_output = module3(module2_output)
        mod3_cost = ce_loss(module3_output, labels)
        mod3_cost.backward(retain_graph=True)
        module3_opt.step()

        # Training gradient module 2
        grad_module2_opt.zero_grad()
        mod2_true_grad_output, mod2_true_grad_w = torch.autograd.grad(mod3_cost, [module2_output, module2.w],
                                                                      retain_graph=True)
        mod2_true_grad_w.volatile = False  # why do I have to set this?
        grad_mod2_cost = squared_loss_sum(mod2_true_grad_w, mod2_generated_grad)
        grad_mod2_cost.backward(retain_graph=True)
        grad_module2_opt.step()

        # Training gradient module 1
        grad_module1_opt.zero_grad()
        mod1_true_grad_w = torch.autograd.grad(module2_output, module1.w,
                                               grad_outputs=mod2_true_grad_output, retain_graph=True)[0]
        mod1_true_grad_w.volatile = False  # why do I have to set this?
        grad_mod1_cost = squared_loss_sum(mod1_true_grad_w, mod1_generated_grad)
        grad_mod1_cost.backward()
        grad_module1_opt.step()

        if print_fn(step):
            try:
                print("epoch", epoch, "   step", step)
                text = "grad_mod1_cost {:.2f}, " \
                       "grad_mod2_cost {:.2f}, " \
                       "mod3_cost {:.2f}, " \
                       "mod1_generated_grads norm {:.2f}, " \
                       "mod2_generated_grads norm {:.2f} "
                print(text.format(float(grad_mod1_cost.data.cpu().numpy()),
                                  float(grad_mod2_cost.data.cpu().numpy()),
                                  float(mod3_cost.data.cpu().numpy()),
                                  float(mod1_generated_grad.norm().cpu().data.numpy()),
                                  float(mod2_generated_grad.norm().cpu().data.numpy())))
            except NameError:
                pass
            test_network()
