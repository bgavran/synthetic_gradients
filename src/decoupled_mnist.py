import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from src.data_loader import MNIST
from src.modules import OneLayer, GradModule, SquaredDifferenceLoss
from src.plot import *

gpu = True

max_epochs = 2

input_size = 784
hidden_size_1 = 128
hidden_size_2 = 128
out_size = 10

batch_size = 128
lr = 0.001

mnist = MNIST(batch_size)

squared_loss_sum = SquaredDifferenceLoss()


def optimize_module(module_optimizer, module_generated_sample, module_true_sample):
    module_optimizer.zero_grad()
    module_true_sample.volatile = False  # why do I have to set this?
    module_cost = squared_loss_sum(module_true_sample, module_generated_sample)
    module_cost.backward(retain_graph=True)
    module_optimizer.step()
    return module_cost


module1 = OneLayer(input_size, hidden_size_1, nonlinearity=nn.ReLU)
module1_opt = torch.optim.Adam(module1.parameters(), lr=lr)
plot_mod1_gen_grad_norm = Plot("Module 1 norm of generated gradients")

grad_module1 = GradModule(hidden_size_1, module1.w.data.shape)
grad_module1_opt = torch.optim.Adam(grad_module1.parameters(), lr=lr)
plot_grad_mod1_loss = Plot("Gradient Module 1 loss")

module2 = OneLayer(hidden_size_1, hidden_size_2, nonlinearity=nn.ReLU)
module2_opt = torch.optim.Adam(module2.parameters(), lr=lr)
plot_mod2_gen_grad_norm = Plot("Module 2 norm of generated gradients")

grad_module2 = GradModule(hidden_size_2, module2.w.data.shape)
grad_module2_opt = torch.optim.Adam(grad_module2.parameters(), lr=lr)
plot_grad_mod2_loss = Plot("Gradient Module 2 loss")

module3 = OneLayer(hidden_size_2, out_size)
module3_opt = torch.optim.Adam(module3.parameters(), lr=lr)
plot_mod3_loss = Plot("Module 3 loss")

ce_loss = nn.CrossEntropyLoss()

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


print("Initialized.")
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

        mod2_output_true_grad, mod2_w_true_grad = torch.autograd.grad(mod3_cost,
                                                                      [module2_output] + list(module2.parameters()),
                                                                      grad_outputs=None,
                                                                      retain_graph=True)
        grad_mod2_cost = optimize_module(grad_module2_opt,
                                         mod2_generated_grad,
                                         mod2_w_true_grad)

        mod1_output_true_grad, mod1_w_true_grad = torch.autograd.grad(module2_output,
                                                                      [module1_output] + list(module1.parameters()),
                                                                      grad_outputs=mod2_output_true_grad,
                                                                      retain_graph=True)
        grad_mod1_cost = optimize_module(grad_module1_opt,
                                         mod1_generated_grad,
                                         mod1_w_true_grad)
        # grad_mod1_cost = Variable(torch.randn(1,))
        # grad_mod2_cost = Variable(torch.randn(1,))

        if print_fn(step):
            print("epoch", epoch, "   step", step)
            stp = len(mnist.train_loader.dataset) * epoch + step * batch_size
            plot_mod3_loss.update(stp, mod3_cost)
            plot_grad_mod1_loss.update(stp, grad_mod1_cost)
            plot_grad_mod2_loss.update(stp, grad_mod2_cost)
            plot_mod1_gen_grad_norm.update(stp, mod1_generated_grad.norm())
            plot_mod2_gen_grad_norm.update(stp, mod2_generated_grad.norm())
            try:
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
