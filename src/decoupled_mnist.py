import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from src.data_loader import MNIST
from src.modules import Net, Module
from src.plot import *

gpu = True

max_epochs = 10

input_size = 784
hidden_size_1 = 32
hidden_size_2 = 32
out_size = 10

batch_size = 64
lr = 0.001

mnist = MNIST(batch_size)

########

layer1 = Net([input_size, hidden_size_1], nonlinearity=nn.ReLU, lr=lr, name="Network 1")
layer2 = Net([hidden_size_1, hidden_size_2], nonlinearity=nn.ReLU, lr=lr, name="Network 2")
layer3 = Net([hidden_size_2, out_size], lr=lr, name="Network 3")

########

inp_module_2 = Module([input_size, hidden_size_1], zero_init=False, name="Input Module 2")
inp_module_3 = Module([input_size, hidden_size_2], zero_init=False, name="Input Module 3")

########

grad_module1 = Module([hidden_size_1, hidden_size_1], lr=lr, name="Gradient Module 1")
grad_module2 = Module([hidden_size_2, hidden_size_2], lr=lr, name="Gradient Module 2")

########

plot_mod3_loss = Plot("Module 3 loss")

ce_loss = nn.CrossEntropyLoss()

if gpu:
    layer1 = layer1.cuda()
    layer2 = layer2.cuda()
    layer3 = layer3.cuda()

    grad_module1 = grad_module1.cuda()
    grad_module2 = grad_module2.cuda()

    inp_module_2 = inp_module_2.cuda()
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
        layer1_output = layer1(images)
        layer1_generated_grad = grad_module1(layer1_output)
        layer1.update_grads(layer1_output, layer1_generated_grad)
        layer1.opt.step()

        # Training second layer with synthetic gradients
        inp_mod2_generated_sample = inp_module_2(images)
        layer2_output = layer2(inp_mod2_generated_sample)
        layer2_generated_grad = grad_module2(layer2_output)
        layer2.update_grads(layer2_output, layer2_generated_grad)
        layer2.opt.step()

        # Training third (last) layer with synthetic input and real gradients
        inp_mod3_generated_sample = inp_module_3(images)
        layer3_output = layer3(inp_mod3_generated_sample)
        mod3_cost = ce_loss(layer3_output, labels)
        mod3_cost.backward(retain_graph=True)
        layer3.opt.step()

        ###
        # grad_mod1_cost = Variable(torch.randn(1, ))
        # grad_mod2_cost = Variable(torch.randn(1, ))
        # inp_mod3_cost = Variable(torch.randn(1, ))
        # inp_mod2_cost = Variable(torch.randn(1, ))
        ###

        # # Training input module 3
        inp_mod3_cost = inp_module_3.optimize_itself(inp_mod3_generated_sample, layer2_output)

        # Training input module 2
        inp_mod2_cost = inp_module_2.optimize_itself(inp_mod2_generated_sample, layer1_output)

        # Training gradient module 2
        mod2_output_true_grad = torch.autograd.grad(mod3_cost,
                                                    [inp_mod3_generated_sample],
                                                    grad_outputs=None,
                                                    retain_graph=True)[0]
        grad_mod2_cost = grad_module2.optimize_itself(layer2_generated_grad, mod2_output_true_grad)

        # Training gradient module 1
        mod1_output_true_grad = torch.autograd.grad(layer2_output,
                                                    [inp_mod2_generated_sample],
                                                    grad_outputs=layer2_generated_grad,
                                                    retain_graph=True)[0]
        grad_mod1_cost = grad_module1.optimize_itself(layer1_generated_grad, mod1_output_true_grad)

        if step % 100 == 0:
            print("epoch", epoch, "   step", step)
            stp = len(mnist.train_loader.dataset) * epoch + step * batch_size
            plot_mod3_loss.update(stp, mod3_cost)
            grad_module1.plot.update(stp, grad_mod1_cost)
            grad_module2.plot.update(stp, grad_mod2_cost)
            layer1.plot.update(stp, layer1_generated_grad.norm())
            layer2.plot.update(stp, layer2_generated_grad.norm())
            inp_module_3.plot.update(stp, inp_mod3_cost)
            inp_module_2.plot.update(stp, inp_mod2_cost)
            try:
                text = "grad_mod1_cost {:.2f}, " \
                       "grad_mod2_cost {:.2f}, " \
                       "mod3_cost {:.2f}, " \
                       "mod1_generated_grads norm {:.2f}, " \
                       "mod2_generated_grads norm {:.2f}, " \
                       "inp_mod3_cost {:.2f} " \
                       " + other..."
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
