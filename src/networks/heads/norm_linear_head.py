import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLinearHead(nn.Module):
    """NormLinearHead"""
    def __init__(self, input, output):
        super(NormLinearHead, self).__init__()
        self.input = input
        self.output = output
        self.weight = nn.Parameter(torch.Tensor(output, input))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.reset_parameters()

    def forward(self, x):
        # pooling layer from torchvision
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # norm layer
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        x_normalized = F.normalize(x, p=2, dim=1)
        out = x_normalized.matmul(weight_normalized.t())
        return x, out

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
