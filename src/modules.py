import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Identity:
    def __call__(self, x):
        return x


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


class OneLayer(nn.Module):
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

    def set_grads(self, grad):
        self.w.grad = grad


class GradModule(nn.Module):
    def __init__(self, input_size, output_shape, nonlinearity=Identity):
        super().__init__()
        self.nonlinearity = nonlinearity()
        self.input_size = input_size
        self.output_shape = output_shape
        self.output_size = int(np.prod(output_shape))

        self.w = nn.Parameter(torch.nn.init.xavier_normal(torch.randn(input_size + 1, self.output_size), gain=10))
        self.zero_init()

    def zero_init(self):
        for param in self.parameters():
            param.data.zero_()

    def forward(self, x):
        ones = Variable(torch.ones(x.data.shape[0], 1)).cuda()
        out = torch.cat((x, ones), dim=1)
        out @= self.w
        out = self.nonlinearity(out)

        out = out.view(-1, *[int(p) for p in self.output_shape]).sum(dim=0)  # reshape to output_shape and sum batch
        return out


class SquaredDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grads_true, grads_predicted):
        out = torch.mean((grads_true - grads_predicted) ** 2)
        return out
