import numbers
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from src.plot import *


class Identity:
    def __call__(self, x):
        return x


class Net(nn.Module):
    def __init__(self, sizes, nonlinearity=Identity, lr=0.001, name="Network"):
        super().__init__()
        self.sizes = sizes
        self.layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(sizes, sizes[1:])])
        self.nonlinearity = nonlinearity()
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.plot = Plot(name + " incoming gradient norm")

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.nonlinearity(x)
        return x

    def update_grads(self, net_output, previous_grad):
        self.opt.zero_grad()
        grad_list = torch.autograd.grad(net_output, self.parameters(), grad_outputs=previous_grad)
        for param, grad in zip(self.parameters(), grad_list):
            param.grad = grad


class Module(nn.Module):
    def __init__(self, sizes,
                 nonlinearity=Identity,
                 lr=0.001,
                 zero_init=True,
                 name="Gradient Module"):
        super().__init__()
        self.nonlinearity = nonlinearity()
        self.sizes = sizes
        self.layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(sizes, sizes[1:])])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.plot = Plot(name)
        if zero_init:
            self.zero_init()

    def zero_init(self):
        for param in self.parameters():
            param.data.zero_()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.nonlinearity(x)
        return x

    def optimize_itself(self, module_generated_sample, module_true_sample):
        self.optimizer.zero_grad()
        if module_true_sample.volatile is True:
            module_true_sample.volatile = False  # why do I have to set this?
        module_cost = squared_loss_sum(module_true_sample, module_generated_sample)
        module_cost.backward(retain_graph=True)
        self.optimizer.step()
        return module_cost


class SquaredDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grads_true, grads_predicted):
        out = torch.mean((grads_true - grads_predicted) ** 2)
        return out


squared_loss_sum = SquaredDifferenceLoss()
