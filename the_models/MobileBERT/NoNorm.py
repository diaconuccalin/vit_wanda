import torch
from torch import nn


class NoNorm(nn.Module):
    def __init__(self, feat_size):
        super(NoNorm, self).__init__()
        self.bias = nn.Parameter(torch.zeros(feat_size))
        self.weight = nn.Parameter(torch.ones(feat_size))

    def forward(self, input_tensor):
        return input_tensor * self.weight + self.bias
