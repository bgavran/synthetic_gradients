import numbers
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from src.plot import *


class Identity:
    def __call__(self, x):
        return x


class Affine(nn.Module):
    def __init__(self, input_size, output_size, nonlinearity=Identity):
        super().__init__()
        self.nonlinearity = nonlinearity()
        self.w = nn.Parameter(torch.nn.init.xavier_normal(torch.randn(input_size + 1, output_size)))

    def forward(self, x):
        ones = Variable(torch.ones(x.data.shape[0], 1)).cuda()
        out = torch.cat((x, ones), dim=1)
        out = out @ self.w
        out = self.nonlinearity(out)
        return out

    def set_grads(self, layer_output, input_grad):
        grad = torch.autograd.grad(layer_output, self.w, grad_outputs=input_grad)[0]
        self.w.grad = grad


class Module(nn.Module):
    def __init__(self, input_size,
                 output_shape,
                 nonlinearity=Identity,
                 lr=0.001,
                 zero_init=True,
                 name="Gradient Module",
                 reshape_and_sum=True):
        super().__init__()
        self.nonlinearity = nonlinearity()
        self.input_size = input_size
        self.output_shape = output_shape
        self.output_size = int(np.prod(output_shape))
        self.reshape_and_sum = reshape_and_sum

        self.w = nn.Parameter(torch.nn.init.xavier_normal(torch.randn(input_size + 1, self.output_size), gain=10))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.plot = Plot(name)
        if zero_init:
            self.zero_init()

    def zero_init(self):
        for param in self.parameters():
            param.data.zero_()

    def forward(self, x):
        ones = Variable(torch.ones(x.data.shape[0], 1)).cuda()
        out = torch.cat((x, ones), dim=1)
        out @= self.w
        out = self.nonlinearity(out)

        if self.reshape_and_sum:
            out = out.view(-1, *self.output_shape)  # reshape to output_shape
            out = out.sum(dim=0)  # sum batch
        return out

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


class Net(nn.Module):
    def __init__(self, sizes, nonlinearity=Identity):
        super().__init__()
        self.sizes = sizes
        self.layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(sizes, sizes[1:])])
        self.nonlinearity = nonlinearity()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.nonlinearity(x)
        return x
