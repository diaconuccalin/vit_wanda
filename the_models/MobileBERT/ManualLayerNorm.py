import torch
from torch import nn


class ManualLayerNorm(nn.Module):
    def __init__(self, feat_size, eps=1e-6):
        super(ManualLayerNorm, self).__init__()
        self.bias = nn.Parameter(torch.zeros(feat_size))
        self.weight = nn.Parameter(torch.ones(feat_size))
        self.eps = eps

    def forward(self, input_tensor):
        mean = input_tensor.mean(-1, keepdim=True)
        std = input_tensor.std(-1, keepdim=True)
        return self.weight * (input_tensor - mean) / (std + self.eps) + self.bias
