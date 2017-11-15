import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class OneLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = nn.Parameter(torch.nn.init.xavier_normal(torch.randn(input_size + 1, output_size)))

    def forward(self, x):
        ones = Variable(torch.ones(x.data.shape[0], 1)).cuda()
        out = torch.cat((x, ones), dim=1)
        out = out @ self.w
        return out

    def set_grads(self, grad):
        self.w.grad = grad


class GradModule(nn.Module):
    def __init__(self, input_size, output_shape):
        super().__init__()
        self.input_size = input_size
        self.output_shape = output_shape
        self.output_size = int(np.prod(output_shape))

        self.w = nn.Parameter(torch.nn.init.xavier_normal(torch.randn(input_size + 1, self.output_size)))
        self.zero_init()

    def zero_init(self):
        for param in self.parameters():
            param.data.zero_()

    def forward(self, x):
        ones = Variable(torch.ones(x.data.shape[0], 1)).cuda()
        out = torch.cat((x, ones), dim=1)
        out @= self.w
        out = out.view(-1, *[int(p) for p in self.output_shape])
        out = out.sum(dim=0)
        return out


class SquaredDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grads_true, grads_predicted):
        return torch.sum((grads_true - grads_predicted) ** 2)
