from timm.models.layers import trunc_normal_
from torch import nn


def init_weights(m):
    """
    Initialize the weights of the model.
    For linear layers:
        - weights: truncated normal distribution, with standard deviation of 0.02
        - bias: constant 0
    For layer normalization layers:
        - weight: constant 1
        - bias: constant 0
    """

    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
