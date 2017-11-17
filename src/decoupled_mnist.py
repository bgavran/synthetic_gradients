import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from src.data_loader import MNIST
from src.modules import Affine, Module
from src.plot import *

gpu = True

max_epochs = 10

input_size = 784
hidden_size_1 = 128
hidden_size_2 = 128
out_size = 10

batch_size = 64
lr = 0.001

mnist = MNIST(batch_size)

########
layer1 = Affine(input_size, hidden_size_1, nonlinearity=nn.ReLU)
layer1_opt = torch.optim.Adam(layer1.parameters(), lr=lr)
plot_mod1_gen_grad_norm = Plot("Module 1 norm of generated gradients")

grad_module1 = Module(hidden_size_1, hidden_size_1, lr=lr, name="Gradient Module 1", reshape_and_sum=False)

########
layer2 = Affine(hidden_size_1, hidden_size_2, nonlinearity=nn.ReLU)
layer2_opt = torch.optim.Adam(layer2.parameters(), lr=lr)
plot_mod2_gen_grad_norm = Plot("Module 2 norm of generated gradients")

grad_module2 = Module(hidden_size_2, hidden_size_2, lr=lr, name="Gradient Module 2", reshape_and_sum=False)

########
layer3 = Affine(hidden_size_2, out_size)
layer3_opt = torch.optim.Adam(layer3.parameters(), lr=lr)

inp_module_3 = Module(input_size,
                      hidden_size_2,
                      zero_init=False,
                      reshape_and_sum=False,
                      name="Input Module 3")

plot_mod3_loss = Plot("Module 3 loss")

ce_loss = nn.CrossEntropyLoss()

########

if gpu:
    layer1 = layer1.cuda()
    layer2 = layer2.cuda()
    layer3 = layer3.cuda()
    grad_module1 = grad_module1.cuda()
    grad_module2 = grad_module2.cuda()
    inp_module_3 = inp_module_3.cuda()


def network_output(images):
    return layer3(layer2(layer1(images)))


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


print("Initialized.")
for epoch in range(max_epochs):
    for step, (images, labels) in enumerate(mnist.train_loader):
        if gpu:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Training first layer with synthetic gradients
        layer1_opt.zero_grad()
        layer1_output = layer1(images)
        layer1_generated_grad = grad_module1(layer1_output)
        layer1.set_grads(layer1_output, layer1_generated_grad)
        layer1_opt.step()

        # Training second layer with synthetic gradients
        layer2_opt.zero_grad()
        layer2_output = layer2(layer1_output)
        layer2_generated_grad = grad_module2(layer2_output)
        layer2.set_grads(layer2_output, layer2_generated_grad)
        layer2_opt.step()

        # Training third (last) layer with synthetic input and real gradients
        layer3_opt.zero_grad()
        inp_mod3_generated_sample = inp_module_3(images)
        layer3_output = layer3(inp_mod3_generated_sample)
        mod3_cost = ce_loss(layer3_output, labels)
        mod3_cost.backward(retain_graph=True)
        layer3_opt.step()

        ###
        grad_mod1_cost = Variable(torch.randn(1, ))
        grad_mod2_cost = Variable(torch.randn(1, ))
        # inp_mod3_cost = Variable(torch.randn(1, ))
        ###

        # Training input module 3
        inp_mod3_cost = inp_module_3.optimize_itself(inp_mod3_generated_sample, layer2_output)

        mod2_output_true_grad = torch.autograd.grad(mod3_cost,
                                                    [inp_mod3_generated_sample],
                                                    grad_outputs=None,
                                                    retain_graph=True)[0]
        grad_mod2_cost = grad_module2.optimize_itself(layer2_generated_grad, mod2_output_true_grad)

        mod1_output_true_grad = torch.autograd.grad(layer2_output,
                                                    [layer1_output],
                                                    grad_outputs=layer2_generated_grad,
                                                    retain_graph=True)[0]
        grad_mod1_cost = grad_module1.optimize_itself(layer1_generated_grad, mod1_output_true_grad)

        if step % 100 == 0:
            print("epoch", epoch, "   step", step)
            stp = len(mnist.train_loader.dataset) * epoch + step * batch_size
            plot_mod3_loss.update(stp, mod3_cost)
            grad_module1.plot.update(stp, grad_mod1_cost)
            grad_module2.plot.update(stp, grad_mod2_cost)
            plot_mod1_gen_grad_norm.update(stp, layer1_generated_grad.norm())
            plot_mod2_gen_grad_norm.update(stp, layer2_generated_grad.norm())
            inp_module_3.plot.update(stp, inp_mod3_cost)
            try:
                text = "grad_mod1_cost {:.2f}, " \
                       "grad_mod2_cost {:.2f}, " \
                       "mod3_cost {:.2f}, " \
                       "mod1_generated_grads norm {:.2f}, " \
                       "mod2_generated_grads norm {:.2f}, " \
                       "inp_mod3_cost {:.2f} "
                print(text.format(float(grad_mod1_cost.data.cpu().numpy()),
                                  float(grad_mod2_cost.data.cpu().numpy()),
                                  float(mod3_cost.data.cpu().numpy()),
                                  float(layer1_generated_grad.norm().cpu().data.numpy()),
                                  float(layer2_generated_grad.norm().cpu().data.numpy()),
                                  float(inp_mod3_cost.data.cpu().numpy())
                                  ))
            except NameError:
                pass
            test_network()
